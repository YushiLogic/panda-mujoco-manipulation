
"""Day 5 实验 2：验证自由方块落地后的稳定性。

本实验保持机器人不动，只观察 cube，从而把“方块物理是否正常”与夹爪
控制分开。cube 初始中心 z=0.05 m，半边长 0.02 m；落在 z=0 地面上
后，中心理论高度应接近 0.02 m。

自由关节的 qpos 占 7 位（xyz+wxyz），qvel 占 6 位（线速度+角速度），
所以代码分别通过 jnt_qposadr 和 jnt_dofadr 查询两种数组的起始地址。
"""

import mujoco
import numpy as np

from panda_mujoco.simulation import PandaScene

def main() -> None:
    # PandaScene 构造时已把机器人和 cube 重置到确定的初始状态。
    scene = PandaScene()
    model = scene.model
    data = scene.data

    # 先用稳定的关节名称获得整数 ID，再用 ID 查询它在大数组中的地址。
    cube_joint_id = model.joint("cube_joint").id

    # qpos_adr 指向位置/姿态起点；dof_adr 指向速度起点。当前模型中二者
    # 恰好都是 9，但动态查询比硬编码 9 更能适应模型结构变化。
    qpos_adr = int(model.jnt_qposadr[cube_joint_id])
    dof_adr = int(model.jnt_dofadr[cube_joint_id])

    # timestep 是一次 mj_step 对应的仿真秒数；duration 是总仿真时长。
    physics_dt = float(model.opt.timestep)
    duration = 6.0

    # 例如 6 s / 0.002 s = 3000 个物理步。
    num_steps = int(round(duration / physics_dt))

    # 每个物理步都记录高度，但只每隔 1 s 打印一次，避免终端输出过密。
    sample_every = int(round(1.0 / physics_dt))

    # 保存完整高度历史，最后检查最后 1 s 是否仍有弹跳或下沉。
    z_history = []

    for step in range(num_steps + 1):
        # free joint 的 qpos 从 qpos_adr 开始依次为 xyz、wxyz，所以 +2
        # 对应 cube 中心的世界坐标 z。
        cube_z = float(data.qpos[qpos_adr + 2])
        z_history.append(cube_z)

        if step % sample_every == 0:
            # free joint 的 qvel 前 3 位单位 m/s，后 3 位单位 rad/s。
            # copy() 固定当前时刻快照，避免视图随下一次 mj_step 改变。
            linear_velocity = data.qvel[dof_adr:dof_adr + 3].copy()
            angular_velocity = data.qvel[dof_adr + 3:dof_adr + 6].copy()

            # 三维向量二范数分别表示线速率和角速率的大小。
            linear_speed = float(np.linalg.norm(linear_velocity))
            angular_speed = float(np.linalg.norm(angular_velocity))

            print(
                f"time={data.time:4.1f} s | "
                f"z={cube_z:.6f} m | "
                f"linear_speed={linear_speed:.3e} m/s | "
                f"angular_speed={angular_speed:.3e} rad/s"
            )

        if step < num_steps:
            # 先记录 t=0，再推进；末尾条件避免多走一步到 duration+dt。
            mujoco.mj_step(model, data)

    # np.ptp 等于 max(z)-min(z)。最后 1 s 的 ptp 越接近 0，说明越稳定。
    last_second_steps = int(round(1.0 / physics_dt))
    final_z_values = np.asarray(z_history[-last_second_steps:])

    final_z = float(data.qpos[qpos_adr + 2])
    final_linear_velocity = data.qvel[dof_adr:dof_adr + 3]
    final_angular_velocity = data.qvel[dof_adr + 3:dof_adr + 6]

    z_variation = float(np.ptp(final_z_values))
    # 线速度和角速度单位不同，必须分别报告，不能合成一个“6D speed”。
    final_linear_speed = float(np.linalg.norm(final_linear_velocity))
    final_angular_speed = float(np.linalg.norm(final_angular_velocity))

    print("\nFinal stability summary")
    print(f"final_z={final_z:.6f} m")
    print(f"last_1s_z_variation={z_variation:.3e} m")
    print(f"final_linear_speed={final_linear_speed:.3e} m/s")
    print(f"final_angular_speed={final_angular_speed:.3e} rad/s")
    # isfinite 用来发现 NaN/Inf；出现非有限值通常表示数值仿真已失稳。
    print(f"all_finite={np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))}")


if __name__ == "__main__":
    main()
