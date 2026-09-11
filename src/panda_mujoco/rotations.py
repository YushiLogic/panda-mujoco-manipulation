"""旋转矩阵和末端姿态误差工具。"""

import numpy as np
from scipy.spatial.transform import Rotation


def _validated_rotation_matrix(
    rotation_matrix: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """检查并返回一个有效旋转矩阵的副本。"""

    rotation_matrix = np.asarray(
        rotation_matrix,
        dtype=float,
    )

    if rotation_matrix.shape != (3, 3):
        raise ValueError(
            f"{name} must have shape (3, 3)"
        )

    if not np.all(np.isfinite(rotation_matrix)):
        raise ValueError(
            f"{name} must contain finite values"
        )

    orthogonality_error = np.max(
        np.abs(
            rotation_matrix.T @ rotation_matrix
            - np.eye(3)
        )
    )

    determinant = np.linalg.det(rotation_matrix)

    if orthogonality_error > 1e-6:
        raise ValueError(
            f"{name} must be orthogonal"
        )

    if not np.isclose(
        determinant,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{name} must have determinant +1"
        )

    return rotation_matrix.copy()


def rotation_error(
    target_rotation: np.ndarray,
    current_rotation: np.ndarray,
) -> np.ndarray:
    """计算从当前姿态到目标姿态的世界坐标系旋转误差。

    Args:
        target_rotation:
            目标坐标系相对于世界坐标系的旋转矩阵，
            形状为 (3, 3)。

        current_rotation:
            当前坐标系相对于世界坐标系的旋转矩阵，
            形状为 (3, 3)。

    Returns:
        世界坐标系下的旋转误差向量，形状为 (3,)，
        向量方向表示旋转轴，向量模长表示旋转角，
        角度单位为弧度。
    """

    target_rotation = _validated_rotation_matrix(
        target_rotation,
        name="target_rotation",
    )

    current_rotation = _validated_rotation_matrix(
        current_rotation,
        name="current_rotation",
    )

    relative_rotation = (
        target_rotation @ current_rotation.T
    )

    error_vector = Rotation.from_matrix(
        relative_rotation
    ).as_rotvec()

    return error_vector