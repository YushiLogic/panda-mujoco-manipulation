"""把外部输入整理为安全的 7 维归一化动作。"""
from scipy.spatial.transform import Rotation
from panda_mujoco.gripper import GRIPPER_OPEN_WIDTH
import numpy as np


ACTION_DIM = 7
POSITION_STEP_M = 0.002
ROTATION_STEP_RAD = np.deg2rad(2.0)
GRIPPER_WIDTH_STEP_M = 0.002

def clip_normalized_action(action: np.ndarray) -> np.ndarray:
    """检查动作，并把每一维限制在 [-1, 1]。"""
    values = np.asarray(action, dtype=np.float64)

    if values.shape != (ACTION_DIM,):
        raise ValueError("action must have shape (7,)")

    if not np.all(np.isfinite(values)):
        raise ValueError("action must contain only finite values")

    return np.clip(values, -1.0, 1.0)


def position_delta_from_action(action: np.ndarray) -> np.ndarray:
    """把动作前三维换算成世界坐标系下的位置增量，单位 m。"""
    safe_action = clip_normalized_action(action)
    return POSITION_STEP_M * safe_action[:3]


def rotation_delta_from_action(action: np.ndarray) -> np.ndarray:
    """返回世界坐标系中的 3×3 增量旋转矩阵。"""
    safe_action = clip_normalized_action(action)
    rotation_vector = ROTATION_STEP_RAD * safe_action[3:6]
    return Rotation.from_rotvec(rotation_vector).as_matrix()


def gripper_target_from_action(
    current_target_width: float,
    action: np.ndarray,
) -> float:
    """计算新的夹爪目标开口，单位 m。"""
    if not np.isfinite(current_target_width):
        raise ValueError("current target width must be finite")

    safe_action = clip_normalized_action(action)
    new_width = (
        current_target_width
        + GRIPPER_WIDTH_STEP_M * safe_action[6]
    )
    return float(np.clip(new_width, 0.0, GRIPPER_OPEN_WIDTH))

def map_action_to_targets(
    action: np.ndarray,
    current_target_position: np.ndarray,
    current_target_rotation: np.ndarray,
    current_target_width: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """把一次 7 维动作映射为下一步的三个控制目标。"""
    safe_action = clip_normalized_action(action)
    position = np.asarray(current_target_position, dtype=np.float64)
    rotation = np.asarray(current_target_rotation, dtype=np.float64)

    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("target position or rotation has wrong shape")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        raise ValueError("target pose must contain only finite values")

    next_position = position + position_delta_from_action(safe_action)
    next_rotation = rotation_delta_from_action(safe_action) @ rotation
    next_width = gripper_target_from_action(
        current_target_width, safe_action
    )
    return next_position, next_rotation, next_width