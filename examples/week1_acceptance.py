"""第一周功能集成验收脚本。

这个脚本会把第一周完成的功能串起来进行检查：

1. 加载 Panda 场景；
2. 重复复位并检查结果；
3. 执行关节运动；
4. 控制夹爪开合；
5. 检查仿真状态是否有效。

目前先实现第 1、2 项。
"""
import time

import mujoco
import numpy as np

from panda_mujoco.gripper import command_gripper, get_gripper_width
from panda_mujoco.joint_trajectory import JointPath
from panda_mujoco.simulation import PANDA_HOME_QPOS, PandaScene

def make_joint1_target(joint1_angle: float) -> np.ndarray:
    """生成一组只改变 joint1、其他关节保持 home 的目标位姿。"""

    # PANDA_HOME_QPOS 前 7 项是机械臂的 7 个关节角。
    # 必须使用 copy()，否则修改 target[0] 时可能影响原始 home 数据。
    target = PANDA_HOME_QPOS[:7].copy()

    # 数组下标 0 对应 joint1。
    target[0] = joint1_angle

    return target

def check_reset_repeatability(scene: PandaScene, repeats: int = 10) -> None:
    """检查连续复位多次后，仿真状态是否完全一致。"""

    # PandaScene 创建时已经自动执行了一次 reset_to_home()。
    # 这里把第一次得到的标准状态保存下来，作为后续比较的参考答案。
    reference_qpos = scene.data.qpos.copy()
    reference_qvel = scene.data.qvel.copy()
    reference_ctrl = scene.data.ctrl.copy()

    for index in range(repeats):
        # 再次复位场景。
        scene.reset_to_home()

        # 将本次复位结果与第一次保存的标准状态比较。
        np.testing.assert_allclose(scene.data.qpos, reference_qpos)
        np.testing.assert_allclose(scene.data.qvel, reference_qvel)
        np.testing.assert_allclose(scene.data.ctrl, reference_ctrl)

        print(f"reset {index + 1:2d}/{repeats}: PASSED")

def move_to_joint_target(
    scene: PandaScene,
    target: np.ndarray,
    ticks_per_move: int = 100,
    control_every: int = 10,
    settle_steps: int = 250,
) -> float:
    """让机械臂平滑移动到一组关节目标，并返回最终最大关节误差。"""

    # 读取机械臂当前的 7 个实际关节角，作为本次运动的起点。
    start = scene.data.qpos[:7].copy()

    # 创建只有一段的轨迹：
    # 当前实际位置 start → 目标位置 target。
    path = JointPath(
        goals=[start, target],
        ticks_per_move=ticks_per_move,
    )

    # path.finished() 为 False 时，说明轨迹还没有走完。
    while not path.finished():
        # 取得当前控制拍对应的插值目标。
        current_target = path.target()

        # 将 7 个关节的目标角写入前 7 个执行器。
        # 注意：ctrl 是“希望关节到哪里”，并不等于直接修改实际 qpos。
        scene.data.ctrl[:7] = current_target

        # 一个控制拍包含 10 个物理仿真步。
        # MuJoCo 会在这些物理步中计算力、加速度、速度和实际位置。
        for _ in range(control_every):
            mujoco.mj_step(scene.model, scene.data)

        # 轨迹规划器前进一个控制拍。
        path.advance()

    # 循环退出时，显式写入最终目标。
    scene.data.ctrl[:7] = target

    # 给机械臂一小段稳定时间，让实际关节角追上目标角。
    for _ in range(settle_steps):
        mujoco.mj_step(scene.model, scene.data)

    # 计算每个关节的绝对误差。
    joint_errors = np.abs(target - scene.data.qpos[:7])

    # 取 7 个关节中最大的那个误差，作为本次运动的验收指标。
    max_error = float(np.max(joint_errors))

    return max_error
def check_joint_targets(scene: PandaScene) -> None:
    """依次执行三个关节目标，并检查最终跟踪误差。"""

    targets = [
        make_joint1_target(0.3),   # 目标1：joint1 转到 +0.3 rad
        make_joint1_target(-0.3),  # 目标2：joint1 转到 -0.3 rad
        make_joint1_target(0.0),   # 目标3：joint1 回到 home
    ]

    # enumerate(..., start=1) 让编号从 1 开始。
    for index, target in enumerate(targets, start=1):
        max_error = move_to_joint_target(scene, target)

        target_joint1 = target[0]
        actual_joint1 = scene.data.qpos[0]

        print(
            f"target {index}/3 | "
            f"joint1 target={target_joint1:+.3f} rad | "
            f"actual={actual_joint1:+.3f} rad | "
            f"max error={max_error:.6f} rad | "
            f"time={scene.data.time:.2f} s"
        )

        # 如果最大关节误差超过 0.02 rad，就认为没有正确到达。
        assert max_error < 0.02, (
            f"Target {index} tracking error is too large: "
            f"{max_error:.6f} rad"
        )

        # 仿真状态不能出现 NaN 或无穷大。
        assert np.all(np.isfinite(scene.data.qpos))
        assert np.all(np.isfinite(scene.data.qvel))
        assert np.all(np.isfinite(scene.data.ctrl))

def move_gripper(
    scene: PandaScene,
    target_width: float,
    duration: float = 1.0,
) -> tuple[float, float]:
    """发送夹爪宽度命令，运行仿真，并返回实际宽度和误差。"""

    # 将目标宽度转换为 actuator8 的控制量，并写入 data.ctrl。
    command_gripper(scene, target_width)

    # MuJoCo 的默认仿真步长是 0.002 s，计算需要多少步才能运行指定的 duration。
    physics_dt = scene.model.opt.timestep
    # 计算 duration 秒需要运行多少个物理步。
    number_of_steps = int(duration / physics_dt)

    #只有调用mj_step()，MuJoCo 才会计算力、加速度、速度和实际位置。
    for _ in range(number_of_steps):
        mujoco.mj_step(scene.model, scene.data)

    #从两根手指的实际qpos计算当前总开口宽度
    actual_width = get_gripper_width(scene)

    #计算实际宽度与目标宽度的误差
    width_error = abs(actual_width - target_width)

    return actual_width, width_error

def check_gripper_targets(scene: PandaScene) -> None:
    """依次检查夹爪闭合、半开和完全张开。"""

    target_widths = [
        0.00,  # 完全闭合
        0.04,  # 半开
        0.08,  # 完全张开
    ]

    for index, target_width in enumerate(target_widths, start=1):
        actual_width, width_error = move_gripper(
            scene=scene,
            target_width=target_width,
        )

        print(
            f"gripper {index}/3 | "
            f"target={target_width:.5f} m | "
            f"actual={actual_width:.5f} m | "
            f"error={width_error:.2e} m | "
            f"time={scene.data.time:.2f} s"
        )

        # 实际宽度与目标宽度的误差不得超过 0.5 mm。
        assert width_error < 0.0005, (
            f"Gripper target {index} error is too large: "
            f"{width_error:.6f} m"
        )

        # 确认仿真状态没有出现 NaN 或 Inf。
        assert np.all(np.isfinite(scene.data.qpos))
        assert np.all(np.isfinite(scene.data.qvel))
        assert np.all(np.isfinite(scene.data.ctrl))

def check_long_term_stability(
    scene: PandaScene,
    simulation_duration: float = 300.0,
    settle_duration: float = 2.0,
) -> None:
    """保持 home 状态运行较长时间，检查数值稳定性和物体漂移。"""

    # 从统一的 home 状态开始。
    scene.reset_to_home()

    physics_dt = scene.model.opt.timestep

    # 先运行 2 秒，让方块落到地面、机械臂稳定下来。
    settle_steps = int(settle_duration / physics_dt)

    for _ in range(settle_steps):
        mujoco.mj_step(scene.model, scene.data)

    # 稳定后记录方块位置，作为漂移比较的参考。
    reference_cube_pos = scene.cube_pos.copy()

    # 300 秒仿真时间对应的物理步数。
    number_of_steps = int(simulation_duration / physics_dt)

    # 每隔 1 秒仿真时间检查一次状态。
    check_every = max(1, int(1.0 / physics_dt))

    wall_start = time.perf_counter()

    for step in range(number_of_steps):
        mujoco.mj_step(scene.model, scene.data)

        if step % check_every == 0:
            # 任意一个数组中出现 NaN 或 Inf，都立即停止验收。
            assert np.all(np.isfinite(scene.data.qpos)), (
                "qpos contains NaN or Inf"
            )
            assert np.all(np.isfinite(scene.data.qvel)), (
                "qvel contains NaN or Inf"
            )
            assert np.all(np.isfinite(scene.data.ctrl)), (
                "ctrl contains NaN or Inf"
            )

    wall_elapsed = time.perf_counter() - wall_start

    final_cube_pos = scene.cube_pos.copy()

    # 方块稳定后的初始位置与最终位置之间的三维距离。
    cube_drift = float(
        np.linalg.norm(final_cube_pos - reference_cube_pos)
    )

    # 机械臂最终实际关节角与 home 控制目标之间的最大误差。
    max_joint_error = float(
        np.max(
            np.abs(
                scene.data.qpos[:7] - PANDA_HOME_QPOS[:7]
            )
        )
    )

    # 夹爪在 home 状态下应该保持完全张开，即 0.08 m。
    final_gripper_width = get_gripper_width(scene)
    gripper_error = abs(final_gripper_width - 0.08)

    print(
        f"simulated duration={simulation_duration:.1f} s | "
        f"wall time={wall_elapsed:.2f} s"
    )
    print(
        f"cube reference={reference_cube_pos} | "
        f"cube final={final_cube_pos}"
    )
    print(f"cube drift={cube_drift:.3e} m")
    print(f"max joint error={max_joint_error:.6f} rad")
    print(f"final gripper width={final_gripper_width:.6f} m")

    # 方块漂移不能超过 0.1 mm。
    assert cube_drift < 0.0001, (
        f"Cube drift is too large: {cube_drift:.6f} m"
    )

    # 机械臂最大关节误差不能超过 0.02 rad。
    assert max_joint_error < 0.02, (
        f"Joint error is too large: {max_joint_error:.6f} rad"
    )

    # 夹爪宽度误差不能超过 0.5 mm。
    assert gripper_error < 0.0005, (
        f"Gripper error is too large: {gripper_error:.6f} m"
    )

def main() -> None:
    """运行第一周集成验收。"""

    print("Loading Panda scene...")

    # 创建 PandaScene：
    # 1. 加载 scene_with_cube.xml；
    # 2. 创建 MjData；
    # 3. 自动复位到 home。
    scene = PandaScene()

    print(f"Scene: {scene.scene_path}")
    print(f"Initial simulation time: {scene.data.time:.3f} s")
    print(f"Initial end-effector position: {scene.ee_pos}")
    print(f"Initial cube position: {scene.cube_pos}")

    print("\nChecking reset repeatability...")
    check_reset_repeatability(scene)
    print("\nReset repeatability check: PASSED")

    print("\nChecking three joint targets...")
    check_joint_targets(scene)

    print("\nJoint target check: PASSED")

    print("\nChecking three gripper targets...")
    check_gripper_targets(scene)

    print("\nGripper target check: PASSED")

    print("\nChecking 300-second simulation stability...")
    check_long_term_stability(scene)

    print("\nLong-term stability check: PASSED")
    print("\nWeek 1 acceptance: PASSED")
if __name__ == "__main__":
    main()