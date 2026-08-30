"""审计 Panda + cube 场景模型（Day 2 交付：模型审计脚本）。

定位：只读"体检报告生成器"——不控制机械臂、不开可视化窗口、不推进仿真时间。
做四件事：
    1. 编译 XML，打印 nq/nv/nu 维度及其来源拆解；
    2. 核对路线图后续代码要按名字引用的全部对象名（查不到 = -1 = 审计失败）；
    3. 打印关节/执行器清单（id、类型、限位、地址、传动对象）；
    4. 做一次 mj_forward，打印末端 site 和 cube 的世界坐标，并输出隐患警告。
退出码：全部通过返回 0，存在未解析名称返回 1（便于 pytest/CI 自动验收）。

用法：
    python examples/inspect_panda.py              # 审计默认模型
    python examples/inspect_panda.py --model 路径  # 审计其他场景
"""

import argparse  # 命令行参数解析：支持 --model 覆盖默认路径
import sys  # 仅用于最后把 main() 的返回值变成进程退出码
from pathlib import Path  # 路径对象，跨平台拼接

import mujoco

# 默认模型路径：从本脚本自身位置反推项目根，而不是写死 "D:\..."——
#   parents[0]=examples/  parents[1]=panda_mujoco_manipulation/  parents[2]=Mujoco Project/
# 好处：无论从哪个目录启动脚本都能找到模型；模型搬家后只需改这一处。
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "model"
    / "franka_emika_panda"
    / "scene_with_cube.xml"
)

# 路线图后续任务（home/关节控制/夹爪/接触检测）要按名字用到的对象名。
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]  # 7 个手臂关节 joint1..joint7
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]  # 2 个手指滑动关节（米制行程）
ACTUATORS = [f"actuator{i}" for i in range(1, 9)]  # 8 个执行器 = 7 臂 + 1 腱

# "接口契约"表：{MuJoCo对象类型: 该类型下必须存在的名字}。
# XML 里写的名字是给人看的，引擎内部只用整数 id；这份表声明了
# 未来代码依赖的全部名字。任何人改坏 XML（改名/删对象），跑本脚本立刻暴露。
REQUIRED_NAMES = {
    mujoco.mjtObj.mjOBJ_JOINT: ARM_JOINTS + FINGER_JOINTS + ["cube_joint"],
    mujoco.mjtObj.mjOBJ_ACTUATOR: ACTUATORS,
    mujoco.mjtObj.mjOBJ_SITE: ["ee_center_site"],  # 末端参考点（场景中唯一的 site）
    mujoco.mjtObj.mjOBJ_BODY: ["hand", "left_finger", "right_finger", "cube"],
    mujoco.mjtObj.mjOBJ_GEOM: ["floor", "cube_geom"],  # Day3 接触检测按 geom 名配对
    mujoco.mjtObj.mjOBJ_TENDON: ["split"],  # 把夹爪驱动力五五分给两指的腱
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，目前只有 --model 一项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="scene XML to audit (default: Menagerie panda scene_with_cube)",
    )
    return parser.parse_args()


def check_names(model) -> list[str]:
    """逐个查询 REQUIRED_NAMES，收集所有查不到的（mj_name2id 返回 -1 的）。

    返回失败清单（空列表 = 全部通过）。
    注意是"收集全部"而不是遇错即停——一次运行暴露所有问题。
    """
    failures = []
    for obj_type, names in REQUIRED_NAMES.items():
        for name in names:
            # mj_name2id：名字 → 整数 id；名字不存在时返回 -1。
            if mujoco.mj_name2id(model, obj_type, name) == -1:
                # 把枚举名 "mjOBJ_SITE" 修成可读的 "site"。
                label = obj_type.name.removeprefix("mjOBJ_").lower()
                failures.append(f"{label} '{name}'")
    return failures


def joint_type_name(model, jid: int) -> str:
    """把关节类型整数（model.jnt_type 里存的）翻译成可读名。

    mjtJoint(...).name 形如 "mjJNT_HINGE"，去掉前缀得到 "HINGE"。
    """
    return mujoco.mjtJoint(model.jnt_type[jid]).name.removeprefix("mjJNT_")


def joint_range(model, jid: int) -> str:
    """返回关节限位的可读字符串；自由关节（cube）无限位。"""
    if not model.jnt_limited[jid]:  # limited=0 表示该关节不设限
        return "unlimited"
    low, high = model.jnt_range[jid]  # 限位下界/上界（铰链=弧度，滑轨=米）
    return f"[{low:+.4f}, {high:+.4f}]"


def actuator_target_name(model, aid: int) -> str:
    """查出执行器 aid 的传动对象——它到底驱动"谁"。

    MuJoCo 执行器不直接连关节，而是连一个"传动装置"（transmission）：
    - mjTRN_JOINT ：直接驱动某个关节（actuator1-7 都属此类）；
    - mjTRN_TENDON：驱动一条腱（actuator8 → tendon "split"，同时拉两根手指）。
    actuator_trnid[aid, 0] 存的就是传动目标的对象 id。
    """
    trn_type = mujoco.mjtTrn(model.actuator_trntype[aid])
    target_id = model.actuator_trnid[aid, 0]
    obj_type = (
        mujoco.mjtObj.mjOBJ_TENDON
        if trn_type == mujoco.mjtTrn.mjTRN_TENDON
        else mujoco.mjtObj.mjOBJ_JOINT
    )
    target = mujoco.mj_id2name(model, obj_type, target_id)  # id 反查回名字
    return f"{target} [{trn_type.name.removeprefix('mjTRN_')}]"


def print_dimensions(model) -> None:
    """打印 nq/nv/nu 及来源拆解（Day 2 验收要求"能解释这三个数"）。

    nq = qpos 数组长度（位置型状态）；nv = 速度自由度 dof 数。
    cube 的自由关节：位置 xyz(3) + 四元数 wxyz(4) = 7 个 qpos，
    但旋转本只占 3 个自由度，故 nv 只加 6 —— nq-nv=1 的来源就是四元数。
    """
    hinges = len(ARM_JOINTS) + len(FINGER_JOINTS)  # 9 个单值关节
    print("== Dimensions ==")
    print(
        f"nq={model.nq}: {hinges} hinge joints x 1 qpos"
        " + cube free joint x 7 (pos xyz + quat wxyz)"
    )
    print(
        f"nv={model.nv}: {hinges} hinge joints x 1 dof"
        " + cube free joint x 6 (translation + rotation)"
    )
    print(f"nu={model.nu}: 7 arm position actuators + 1 tendon actuator for both fingers")


def print_joints(model) -> None:
    """打印全部关节表：id、类型、在状态数组中的地址、限位。"""
    print("\n== Joints ==")
    print(f"{'id':>3}  {'name':<14} {'type':<6} {'qposadr':>7} {'dofadr':>6}  range")
    for jid in range(model.njnt):  # 遍历模型全部关节（共 10 个）
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)  # id→名字
        # qposadr：该关节的数值在全局 data.qpos 中的起始下标。
        # 例：cube_joint 的 qposadr=9 → data.qpos[9:16] 是 cube 位姿（7 个数）。
        # 以后直接读写 data.qpos 时，全靠这张地址表定位。
        print(
            f"{jid:>3}  {name:<14} {joint_type_name(model, jid):<6} "
            f"{model.jnt_qposadr[jid]:>7} {model.jnt_dofadr[jid]:>6}  {joint_range(model, jid)}"
        )


def print_actuators(model) -> None:
    """打印全部执行器表：id、合法控制量范围、传动目标。"""
    print("\n== Actuators ==")
    print(f"{'id':>3}  {'name':<10} {'ctrlrange':>22}  target")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        # ctrlrange = data.ctrl 允许写入的区间。注意量纲不统一：
        # actuator1-7 是弧度，actuator8 是 0~255 的无量纲开合度。
        low, high = model.actuator_ctrlrange[aid]
        print(
            f"{aid:>3}  {name:<10} [{low:>9.4f}, {high:>9.4f}]  {actuator_target_name(model, aid)}"
        )


def print_forward_state(model, data) -> None:
    """做一次正运动学，打印末端 site 与 cube 的世界坐标。

    mj_forward 只"由当前关节角重算一切派生量"（含各 site 世界坐标），
    时间纹丝不动；mj_step 才是推进物理。体检只拍照，不做运动。
    data.site("...") / data.joint("...") 是 MuJoCo 命名访问，免查 id。
    """
    mujoco.mj_forward(model, data)
    ee = data.site("ee_center_site").xpos  # 末端 site 的世界坐标 (x, y, z)
    cube = data.joint("cube_joint").qpos  # cube 的 7 维 qpos：xyz + 四元数 wxyz
    print("\n== World poses after mj_forward (default qpos0, no home set yet) ==")
    print(f"ee_center_site pos : ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})")
    print(f"cube pos           : ({cube[0]:+.4f}, {cube[1]:+.4f}, {cube[2]:+.4f})")
    print(f"cube quat (wxyz)   : ({', '.join(f'{v:+.3f}' for v in cube[3:])})")


def print_warnings(model) -> None:
    """打印写控制代码前必须知道的隐患（现场读模型，不硬编码结论）。"""
    print("\n== Warnings ==")
    # 隐患1：joint4 的默认关节角（qpos0）是否落在自身限位内。
    # 做法是"读出来再判断"，结论不写死——模型改了，警告会自动跟着变。
    j4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint4")
    low, high = model.jnt_range[j4]
    q0 = model.qpos0[model.jnt_qposadr[j4]]  # 默认关节角数组中 joint4 的值
    if not low <= q0 <= high:
        print(
            f"joint4 default qpos0={q0:+.4f} is OUTSIDE its range {joint_range(model, j4)};"
            " an explicit home pose in reset_to_home is required (Day 3)."
        )
    # 隐患2：夹爪执行器控制量是 0~255 而非弧度，与手臂执行器量纲不同，勿混用。
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    low, high = model.actuator_ctrlrange[gripper]
    print(f"actuator8 (gripper tendon) ctrlrange is [{low:.0f}, {high:.0f}], not radians.")


def main() -> int:
    """编排整个审计流程；返回 0=通过 / 1=失败（给 pytest、CI 判定用）。"""
    args = parse_args()
    print(f"MuJoCo {mujoco.__version__}")
    print(f"Model : {args.model}")
    if not args.model.exists():
        print(f"ERROR: model file not found: {args.model}")
        return 1

    # MjModel：编译后的静态结构（关节/执行器/限位……只读，不会变）。
    model = mujoco.MjModel.from_xml_path(str(args.model))
    # MjData：运行时可变状态（qpos、qvel、ctrl、各 site 世界坐标……）。
    data = mujoco.MjData(model)

    failures = check_names(model)
    print_dimensions(model)
    print_joints(model)
    print_actuators(model)
    print_forward_state(model, data)
    print_warnings(model)

    print()
    if failures:
        print("AUDIT FAILED - unresolved names:")
        for item in failures:
            print(f"  - {item}")
        return 1  # 非零退出码 = 审计失败
    print("AUDIT PASSED - all required names resolved.")
    return 0


if __name__ == "__main__":  # 直接运行本文件才执行；被 import 时不执行
    sys.exit(main())
