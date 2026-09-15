"""Panda抓取目标的几何定义。

本模块只负责根据方块位置计算抓取目标，不修改MuJoCo状态，
不写入ctrl，也不推进仿真时间。
"""

from dataclasses import dataclass

import numpy as np


# 向下抓取时，末端坐标系相对于世界坐标系的目标旋转矩阵。
#
# 第一列：末端局部+x轴指向世界+x
# 第二列：末端局部+y轴指向世界-y，作为夹爪张合轴
# 第三列：末端局部+z轴指向世界-z，作为向下接近轴
TOP_DOWN_GRASP_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
# 默认预抓取点位于抓取点上方10厘米。
DEFAULT_PREGRASP_DISTANCE = 0.10

# ee_center_site暂时设置在方块中心上方5毫米。
# 这是初始工程参数，后面要根据实际接触结果调整。
DEFAULT_GRASP_Z_OFFSET = 0.005

# 抓住方块后，末端向世界+z方向抬升8厘米。
DEFAULT_LIFT_HEIGHT = 0.08

@dataclass(frozen=True)
class GraspTargets:
    """一次抓取所需的三个目标位置和一个目标姿态。

    所有位置都使用世界坐标系，单位为米。
    rotation表示末端坐标系相对于世界坐标系的旋转矩阵。
    """

    pregrasp_position: np.ndarray
    grasp_position: np.ndarray
    lift_position: np.ndarray
    rotation: np.ndarray

def generate_grasp_targets(
    cube_position: np.ndarray,
    *,
    pregrasp_distance: float = DEFAULT_PREGRASP_DISTANCE,
    grasp_z_offset: float = DEFAULT_GRASP_Z_OFFSET,
    lift_height: float = DEFAULT_LIFT_HEIGHT,
) -> GraspTargets:
    """根据方块中心位置生成向下抓取目标。

    Args:
        cube_position:
            方块中心在世界坐标系中的位置，形状为(3,)，单位为米。

        pregrasp_distance:
            预抓取点沿接近方向反方向离开抓取点的距离。

        grasp_z_offset:
            ee_center_site相对于方块中心的世界z方向偏置。

        lift_height:
            抓取后沿世界+z方向抬升的距离。

    Returns:
        包含pregrasp、grasp、lift位置和目标旋转矩阵的GraspTargets。
    """

    # 将list或其他数组输入统一转换为浮点NumPy数组。
    cube_position = np.asarray(
        cube_position,
        dtype=float,
    )

    # 方块位置必须严格包含x、y、z三个数。
    if cube_position.shape != (3,):
        raise ValueError(
            "cube_position must have shape (3,)"
        )

    # NaN和Inf无法作为有效的几何位置。
    if not np.all(np.isfinite(cube_position)):
        raise ValueError(
            "cube_position must contain finite values"
        )

    # 预抓取距离必须为有限正数。
    if (
        not np.isfinite(pregrasp_distance)
        or pregrasp_distance <= 0.0
    ):
        raise ValueError(
            "pregrasp_distance must be positive and finite"
        )

    # 抓取高度偏置可以为正、零或负，但必须是有限数。
    if not np.isfinite(grasp_z_offset):
        raise ValueError(
            "grasp_z_offset must be finite"
        )

    # 抬升高度必须为有限正数。
    if (
        not np.isfinite(lift_height)
        or lift_height <= 0.0
    ):
        raise ValueError(
            "lift_height must be positive and finite"
        )

    # 使用副本，避免后续计算修改调用者传入的数组。
    grasp_position = cube_position.copy()

    # 抓取点在方块中心基础上增加高度偏置。
    grasp_position[2] += grasp_z_offset

    # 旋转矩阵第三列是末端局部+z轴，即接近方向。
    approach_direction = TOP_DOWN_GRASP_ROTATION[:, 2]

    # 接近方向朝下，所以沿它的反方向退回，
    # 就能得到位于抓取点上方的预抓取点。
    pregrasp_position = (
        grasp_position
        - pregrasp_distance * approach_direction
    )

    # 抬升使用世界坐标系+z方向，而不是末端局部坐标轴。
    world_up = np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    lift_position = (
        grasp_position
        + lift_height * world_up
    )

    return GraspTargets(
        pregrasp_position=pregrasp_position.copy(),
        grasp_position=grasp_position.copy(),
        lift_position=lift_position.copy(),
        rotation=TOP_DOWN_GRASP_ROTATION.copy(),
    )