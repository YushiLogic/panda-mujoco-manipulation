"""方块复位与可复现随机化工具。"""

from dataclasses import dataclass

import mujoco
import numpy as np

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
)
from panda_mujoco.simulation import PandaScene


CUBE_JOINT_NAME = "cube_joint"

# 为后续抓取实验选取一个较保守的桌面区域。
DEFAULT_X_RANGE = (0.42, 0.48)
DEFAULT_Y_RANGE = (-0.06, 0.06)

# 方块保持直立，只随机改变绕世界z轴的朝向。
DEFAULT_YAW_RANGE = (
    -np.deg2rad(30.0),
    np.deg2rad(30.0),
)

# 方块先生成在地面上方，随后由动力学自然落稳。
DEFAULT_SPAWN_HEIGHT = 0.05


@dataclass(frozen=True)
class CubePose:
    """一次方块位姿采样的结果。"""

    position: np.ndarray
    quaternion: np.ndarray
    yaw: float


@dataclass(frozen=True)
class CubeSettleResult:
    """方块落稳过程的最终结果。"""

    settled: bool
    steps: int
    simulation_duration: float
    final_position: np.ndarray
    linear_speed: float
    angular_speed: float


def get_cube_joint_addresses(
    model: mujoco.MjModel,
) -> tuple[int, int]:
    """返回方块free joint在qpos和qvel中的起始地址。"""

    joint_id = model.joint(CUBE_JOINT_NAME).id

    # 防止模型中同名关节被意外改成其他类型。
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
            f"{CUBE_JOINT_NAME} must be a free joint"
        )

    qpos_address = int(
        model.jnt_qposadr[joint_id]
    )
    qvel_address = int(
        model.jnt_dofadr[joint_id]
    )

    return qpos_address, qvel_address


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    """将绕世界z轴的yaw角转换为MuJoCo的wxyz四元数。"""

    yaw = float(yaw)

    if not np.isfinite(yaw):
        raise ValueError(
            "yaw must be finite"
        )

    half_yaw = 0.5 * yaw

    quaternion = np.array(
        [
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        ],
        dtype=float,
    )

    return quaternion


def sample_cube_pose(
    seed: int | None,
) -> CubePose:
    """根据seed采样一个直立的方块位姿。"""

    # 创建局部随机数生成器，不修改NumPy的全局随机状态。
    rng = np.random.default_rng(seed)

    x = rng.uniform(
        DEFAULT_X_RANGE[0],
        DEFAULT_X_RANGE[1],
    )
    y = rng.uniform(
        DEFAULT_Y_RANGE[0],
        DEFAULT_Y_RANGE[1],
    )
    yaw = rng.uniform(
        DEFAULT_YAW_RANGE[0],
        DEFAULT_YAW_RANGE[1],
    )

    position = np.array(
        [x, y, DEFAULT_SPAWN_HEIGHT],
        dtype=float,
    )

    quaternion = yaw_to_quaternion(yaw)

    return CubePose(
        position=position,
        quaternion=quaternion,
        yaw=float(yaw),
    )


def reset_cube(
    scene: PandaScene,
    pose: CubePose,
) -> None:
    """将方块复位到指定姿态，并清除残留速度。"""

    position = np.asarray(
        pose.position,
        dtype=float,
    )
    quaternion = np.asarray(
        pose.quaternion,
        dtype=float,
    )

    if position.shape != (3,):
        raise ValueError(
            "cube position must have shape (3,)"
        )

    if quaternion.shape != (4,):
        raise ValueError(
            "cube quaternion must have shape (4,)"
        )

    if not np.all(np.isfinite(position)):
        raise ValueError(
            "cube position must contain finite values"
        )

    if not np.all(np.isfinite(quaternion)):
        raise ValueError(
            "cube quaternion must contain finite values"
        )

    if not np.isfinite(pose.yaw):
        raise ValueError(
            "cube yaw must be finite"
        )

    quaternion_norm = np.linalg.norm(
        quaternion
    )

    if not np.isclose(
        quaternion_norm,
        1.0,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError(
            "cube quaternion must have unit norm"
        )

    qpos_address, qvel_address = (
        get_cube_joint_addresses(scene.model)
    )

    # free joint的qpos：xyz + wxyz。
    scene.data.qpos[
        qpos_address:qpos_address + 3
    ] = position

    scene.data.qpos[
        qpos_address + 3:qpos_address + 7
    ] = quaternion

    # free joint的qvel：三维线速度 + 三维角速度。
    scene.data.qvel[
        qvel_address:qvel_address + 6
    ] = 0.0

    # 根据新的qpos重新计算body位置、接触等派生量。
    # mj_forward不会推进仿真时间。
    mujoco.mj_forward(
        scene.model,
        scene.data,
    )


def settle_cube(
    scene: PandaScene,
    *,
    duration: float = 1.5,
    linear_speed_tolerance: float = 1e-4,
    angular_speed_tolerance: float = 1e-3,
) -> CubeSettleResult:
    """推进动力学，让方块下落并判断最终是否稳定。"""

    if (
        not np.isfinite(duration)
        or duration <= 0.0
    ):
        raise ValueError(
            "duration must be positive and finite"
        )

    if (
        not np.isfinite(linear_speed_tolerance)
        or linear_speed_tolerance <= 0.0
    ):
        raise ValueError(
            "linear_speed_tolerance must be "
            "positive and finite"
        )

    if (
        not np.isfinite(angular_speed_tolerance)
        or angular_speed_tolerance <= 0.0
    ):
        raise ValueError(
            "angular_speed_tolerance must be "
            "positive and finite"
        )

    _, qvel_address = get_cube_joint_addresses(
        scene.model
    )

    start_time = float(scene.data.time)
    end_time = start_time + duration
    steps = 0

    while scene.data.time < end_time:
        # 让机械臂在方块落稳期间保持当前姿态。
        apply_arm_bias_compensation(scene)

        # 推进一个MuJoCo物理时间步。
        mujoco.mj_step(
            scene.model,
            scene.data,
        )

        steps += 1

    cube_velocity = scene.data.qvel[
        qvel_address:qvel_address + 6
    ].copy()

    linear_speed = float(
        np.linalg.norm(
            cube_velocity[:3]
        )
    )

    angular_speed = float(
        np.linalg.norm(
            cube_velocity[3:]
        )
    )

    final_position = (
        scene.data.body("cube").xpos.copy()
    )

    if not np.all(
        np.isfinite(final_position)
    ):
        raise RuntimeError(
            "cube final position is not finite"
        )

    if not np.all(
        np.isfinite(cube_velocity)
    ):
        raise RuntimeError(
            "cube final velocity is not finite"
        )

    settled = (
        linear_speed
        <= linear_speed_tolerance
        and angular_speed
        <= angular_speed_tolerance
    )

    return CubeSettleResult(
        settled=settled,
        steps=steps,
        simulation_duration=(
            float(scene.data.time)
            - start_time
        ),
        final_position=final_position,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
    )
