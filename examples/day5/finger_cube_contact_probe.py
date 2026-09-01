"""Day 5 实验 4：构造并检测双指与方块的接触。

本实验不要求机器人先运动到桌面上的方块，而是直接把 cube 放到 home
姿态的两指之间，用来隔离验证以下链路：

    夹爪闭合命令 -> 手指运动 -> 碰撞求解 -> data.contact
                  -> ContactSensor 判断左右手指是否接触 cube

实验中暂时关闭重力，避免 cube 在夹爪闭合前掉落。这只是接触检测的
单元实验，不代表完成真实抓取；真实抓取还需要重力、抬升和持续性。
"""

import mujoco
import numpy as np

from panda_mujoco.contact import ContactSensor
from panda_mujoco.simulation import PandaScene


def place_cube_between_fingers(
    scene: PandaScene,
    offset: float = 0.0,
) -> None:
    """把自由方块放到夹爪中心，并让方块坐标轴与夹爪对齐。

    若只改世界位置而不改姿态，倾斜的夹爪可能先碰到 cube 的角，左右
    接触不再对称。这里同时复制位置和旋转，构造容易解释的对称场景。
    """

    model = scene.model
    data = scene.data

    # cube 是 free joint：qpos 占 7 位（xyz+wxyz），qvel 占 6 位。
    # 先通过名称得到 joint ID，再分别查询两种状态数组的起始地址。
    cube_joint_id = model.joint("cube_joint").id
    cube_qpos_adr = int(model.jnt_qposadr[cube_joint_id])
    cube_dof_adr = int(model.jnt_dofadr[cube_joint_id])

    # ee_center_body 位于两根手指的几何中心。这里读取的是当前时刻在
    # 世界坐标系中的位置 xpos 和 3×3 旋转矩阵 xmat。
    ee_body = data.body("ee_center_body")

    # finger_joint1 在模型中有自己的局部移动轴，但夹爪当前是倾斜的，
    # 所以不能简单地沿世界坐标 x 或 y 移动方块。
    finger_joint_id = model.joint("finger_joint1").id

    # data.xaxis 保存关节轴经过机器人当前姿态旋转后，在世界坐标系中的方向。
    opening_axis = data.xaxis[finger_joint_id].copy()


    # free joint 的姿态要写成 wxyz 四元数，因此用 MuJoCo 工具函数把
    # ee 的旋转矩阵转换成四元数，避免手工换算顺序错误。
    cube_quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(cube_quaternion, ee_body.xmat)

    # free joint 的 qpos 布局是 xyz + wxyz。这里直接设置实验初态，
    # 从夹爪中心出发，沿手指开合方向移动offset米。
    cube_position = ee_body.xpos + offset * opening_axis

    data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = cube_position
    data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = cube_quaternion

    # 重新放置后必须清除原来的线速度和角速度，否则 cube 会带着旧动量运动。
    data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0

    # 直接写 qpos 后，派生世界坐标不会自动刷新。mj_forward 不推进时间，
    # 只根据新状态重算 body/geom 坐标和接触候选，使读数立即有效。
    mujoco.mj_forward(model, data)


def print_contact_state(
    scene: PandaScene,
    sensor: ContactSensor,
    label: str,
) -> None:
    """打印当前时刻的双指实际位置和接触分类快照。"""

    # ContactSensor 不修改仿真，只读取本物理步的 data.contact。
    left_count, right_count = sensor.finger_contacts(scene.data)

    # ctrl[7] 是目标命令，而 qpos[7:9] 是实际手指位置；有 cube 阻挡时，
    # 即使 ctrl=0，实际 qpos 也会停在 cube 半宽附近。
    left_qpos = float(scene.data.qpos[7])
    right_qpos = float(scene.data.qpos[8])

    print(f"\n{label}")
    print(f"finger qpos: left={left_qpos:.6f}, right={right_qpos:.6f}")
    print(f"all contacts: {scene.data.ncon}")
    print(f"left-cube contacts: {left_count}")
    print(f"right-cube contacts: {right_count}")
    # 双侧接触只证明当前一帧左右手指都碰到 cube，不等于稳定抓取成功。
    print(f"bilateral contact: {left_count > 0 and right_count > 0}")


def main() -> None:
    # reset 后：机械臂在 home、夹爪完全张开、cube 位于桌面上方初态。
    scene = PandaScene()
    model = scene.model
    data = scene.data

    # 隔离接触识别：只修改运行时 model，不改 XML。关闭重力后，放到
    # 两指之间的自由 cube 不会在夹爪闭合前掉走。
    model.opt.gravity[:] = 0.0

    # 把 cube 移到夹爪中心、对齐姿态并清除旧速度。
    place_cube_between_fingers(scene)

    # 构造时一次性收集 left_finger/right_finger/cube 所属的 geom ID。
    sensor = ContactSensor(model)

        # 状态1：夹爪张开，方块位于正中间。
    place_cube_between_fingers(scene, offset=0.0)
    print_contact_state(scene, sensor, "OPEN CENTERED")

    # 状态2：方块向左手指移动21 mm。
    # 方块半宽20 mm，夹爪半开口40 mm，因此偏移约20 mm时开始接触。
    place_cube_between_fingers(scene, offset=0.021)
    print_contact_state(scene, sensor, "LEFT ONLY")

    # 状态3：方块向相反方向移动21 mm。
    place_cube_between_fingers(scene, offset=-0.021)
    print_contact_state(scene, sensor, "RIGHT ONLY")

    # 状态4：方块重新放回正中间，再闭合夹爪。
    place_cube_between_fingers(scene, offset=0.0)
    data.ctrl[7] = 0.0

    num_steps = int(round(1.0 / model.opt.timestep))
    for _ in range(num_steps):
        mujoco.mj_step(model, data)

    print_contact_state(scene, sensor, "CLOSED BILATERAL")


if __name__ == "__main__":
    main()
