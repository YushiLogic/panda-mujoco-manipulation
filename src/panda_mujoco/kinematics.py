"""Panda 末端运动学读取工具。

MuJoCo 会根据当前 qpos 自动完成正运动学计算。
本模块负责以清晰、可复用的接口读取末端位置和姿态。
"""
import mujoco
import numpy as np

from panda_mujoco.simulation import PandaScene


EE_SITE_NAME = "ee_center_site"
ARM_JOINT_NAMES = tuple(
    f"joint{index}" for index in range(1, 8)
)


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


def get_ee_jacobian(
    scene: PandaScene,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 Panda 末端相对于7个手臂关节的 Jacobian。

    Returns:
        linear_jacobian:
            末端线速度 Jacobian，形状为 (3, 7)。

        angular_jacobian:
            末端角速度 Jacobian，形状为 (3, 7)。
    """

    model = scene.model
    data = scene.data

    site_id = model.site(EE_SITE_NAME).id

    # mj_jacSite 输出的是相对于模型全部 nv 个速度自由度的结果。
    full_linear_jacobian = np.zeros((3, model.nv))
    full_angular_jacobian = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        full_linear_jacobian,
        full_angular_jacobian,
        site_id,
    )

    # 找出 joint1～joint7 在 qvel/Jacobian 中对应的列号。
    arm_dof_addresses = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = model.joint(joint_name).id
        dof_address = model.jnt_dofadr[joint_id]
        arm_dof_addresses.append(dof_address)

    # 只保留7个机械臂关节对应的列。
    linear_jacobian = full_linear_jacobian[
        :, arm_dof_addresses
    ].copy()

    angular_jacobian = full_angular_jacobian[
        :, arm_dof_addresses
    ].copy()

    return linear_jacobian, angular_jacobian