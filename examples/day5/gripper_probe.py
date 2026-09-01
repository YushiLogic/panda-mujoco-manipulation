
"""Day 5 实验 1：验证夹爪命令与实际手指位置的关系。

本实验只让夹爪在空中开合，不接触方块，用来单独验证 actuator8 的
0～255 命令能否让两根手指到达 0～0.04 m，并由 equality 保持同步。

始终区分：data.ctrl[7] 是目标命令；data.qpos[7:9] 是实际位置（米）。
无物体阻挡时，单指目标位置近似为 0.04 * ctrl / 255。
"""

import mujoco

from panda_mujoco.simulation import PandaScene


def hold_gripper(scene: PandaScene, command: float, duration: float) -> None:
    """发送夹爪命令，并推进足够的物理时间让手指产生实际响应。

    Args:
        scene: 已加载并 reset 到 home 的 PandaScene。
        command: actuator8 命令；0=闭合，255=张开。
        duration: 命令保持的仿真秒数，不是程序墙钟时间。
    """

    # 第 8 个执行器在从 0 开始的 ctrl 数组中下标为 7。这里只改变目标
    # 命令，不会瞬间改变实际 qpos；手指需要后续 mj_step 才会运动。
    scene.data.ctrl[7] = command

    # 一次 mj_step 推进的仿真时间。本模型通常为 0.002 s，即 500 Hz。
    physics_dt = scene.model.opt.timestep

    # 例如 1 s / 0.002 s = 500 步。根据 timestep 计算而不写死 500，
    # 这样以后调整模型时间步后实验时长仍然正确。
    num_steps = int(duration / physics_dt)

    for _ in range(num_steps):
        # model 存静态物理规则，data 存当前状态；mj_step 根据 ctrl 和当前
        # 状态计算力、速度与下一时刻位置，并把结果写回 data。
        mujoco.mj_step(scene.model, scene.data)


def print_gripper_state(scene: PandaScene, command: float) -> None:
    """把目标命令和左右手指的实际响应放在同一行打印。"""

    # qpos 前 7 位是手臂关节；下标 7、8 是两个 slide 类型的手指关节。
    left = float(scene.data.qpos[7])
    right = float(scene.data.qpos[8])

    # 两指从中心分别向两侧移动，因此总开口近似为两根手指行程之和。
    opening_width = left + right
    # equality 约束要求左右同步；这个差值量化约束的数值误差。
    synchronization_error = abs(left - right)

    print(
        f"ctrl={command:6.1f} | "
        f"left={left:.5f} m | "
        f"right={right:.5f} m | "
        f"width={opening_width:.5f} m | "
        f"sync_error={synchronization_error:.2e} m"
    )


def main() -> None:
    # 构造时会加载模型、创建 MjData 并 reset_to_home；home 下夹爪张开。
    scene = PandaScene()

    # 每个二元组是 (命令, 保持秒数)。最后再次张开用于检查可重复性。
    experiments = [
        (255.0, 1.0),
        (128.0, 1.0),
        (0.0, 1.0),
        (255.0, 1.0),
    ]

    for command, duration in experiments:
        # 必须先推进仿真再读 qpos，否则看到的仍是上一个时刻的实际状态。
        hold_gripper(scene, command, duration)
        print_gripper_state(scene, command)


# 直接运行本文件时才做实验；被其他模块 import 时不会自动执行 main。
if __name__ == "__main__":
    main()
