"""Panda 末端运动学读取工具。

MuJoCo 会根据当前 qpos 自动完成正运动学计算。
本模块负责以清晰、可复用的接口读取末端位置和姿态。
"""

import numpy as np

from panda_mujoco.simulation import PandaScene


EE_SITE_NAME = "ee_center_site"


def get_ee_position(scene: PandaScene) -> np.ndarray:
    """返回末端 site 在 world frame 中的位置，形状为 (3,)。"""

    position = scene.data.site(EE_SITE_NAME).xpos.copy()

    return position


def get_ee_rotation_matrix(scene: PandaScene) -> np.ndarray:
    """返回末端坐标系相对于 world frame 的旋转矩阵。"""

    rotation = (
        scene.data
        .site(EE_SITE_NAME)
        .xmat
        .reshape(3, 3)
        .copy()
    )

    return rotation


def get_ee_pose(
    scene: PandaScene,
) -> tuple[np.ndarray, np.ndarray]:
    """返回末端位置和旋转矩阵。"""

    position = get_ee_position(scene)
    rotation = get_ee_rotation_matrix(scene)

    return position, rotation