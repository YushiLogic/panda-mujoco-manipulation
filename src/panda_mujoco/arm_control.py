"""Panda 手臂执行器控制工具。"""

import numpy as np

from panda_mujoco.kinematics import ARM_JOINT_NAMES
from panda_mujoco.simulation import PandaScene


ARM_ACTUATOR_NAMES = tuple(
    f"actuator{index}" for index in range(1, 8)
)


def _get_arm_actuator_ids(
    scene: PandaScene,
) -> np.ndarray:
    """取得七个手臂执行器的 ID。"""

    actuator_ids = np.array(
        [
            scene.model.actuator(actuator_name).id
            for actuator_name in ARM_ACTUATOR_NAMES
        ],
        dtype=int,
    )

    return actuator_ids


def _get_arm_dof_addresses(
    scene: PandaScene,
) -> np.ndarray:
    """取得七个手臂关节在 qvel/qfrc 中的地址。"""

    dof_addresses = np.array(
        [
            scene.model.jnt_dofadr[
                scene.model.joint(joint_name).id
            ]
            for joint_name in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    return dof_addresses


def command_arm_joint_positions(
    scene: PandaScene,
    target_positions: np.ndarray,
) -> None:
    """向七个手臂位置执行器发送目标关节角。

    本函数只写入 data.ctrl，不会直接修改 qpos，
    也不会推进仿真时间。
    """

    target_positions = np.asarray(
        target_positions,
        dtype=float,
    )

    if target_positions.shape != (7,):
        raise ValueError(
            "target_positions must have shape (7,)"
        )

    if not np.all(np.isfinite(target_positions)):
        raise ValueError(
            "target_positions must contain finite values"
        )

    actuator_ids = _get_arm_actuator_ids(scene)

    control_ranges = (
        scene.model.actuator_ctrlrange[actuator_ids]
    )

    lower_limits = control_ranges[:, 0]
    upper_limits = control_ranges[:, 1]

    if np.any(target_positions < lower_limits) or np.any(
        target_positions > upper_limits
    ):
        raise ValueError(
            "target_positions must lie within actuator "
            "control ranges"
        )

    scene.data.ctrl[actuator_ids] = target_positions


def apply_arm_bias_compensation(
    scene: PandaScene,
) -> None:
    """将当前手臂偏置力作为理想模型前馈补偿。

    应在每个 mj_step() 之前调用，因为 qfrc_bias
    会随当前 qpos 和 qvel 改变。
    """

    dof_addresses = _get_arm_dof_addresses(scene)

    scene.data.qfrc_applied[dof_addresses] = (
        scene.data.qfrc_bias[dof_addresses]
    )
