"""Day 15：审计 Panda 夹爪与方块的抓取几何。

本程序只读取 MuJoCo 模型和当前状态，不修改 qpos、ctrl，
也不推进仿真时间。我们要确认：

1. 方块实际尺寸；
2. ee_center_site 的世界位置和三个局部轴；
3. 左右手指的关节轴；
4. 两根手指是否沿相反方向运动；
5. 手指开合轴是否与末端局部 y 轴一致。
"""
import mujoco
import numpy as np

from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.simulation import PandaScene


def get_joint_axis_world(
    scene: PandaScene,
    joint_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """返回关节轴在局部坐标系和世界坐标系中的表达。"""

    model = scene.model
    data = scene.data

    # 通过关节名称取得关节ID。
    joint_id = model.joint(joint_name).id

    # MuJoCo记录了每个关节属于哪个body。
    body_id = int(model.jnt_bodyid[joint_id])

    # jnt_axis是在关节所属body局部坐标系中表达的。
    axis_local = model.jnt_axis[joint_id].copy()

    # data.xmat给出body相对于世界坐标系的旋转矩阵。
    body_rotation = data.xmat[body_id].reshape(3, 3).copy()

    # 将局部关节轴转换到世界坐标系。
    axis_world = body_rotation @ axis_local

    return axis_local, axis_world

def find_main_fingertip_pad_geom_id(
    scene: PandaScene,
    finger_body_name: str,
) -> int:
    """查找指定手指上的主要指腹box碰撞几何体。"""

    model = scene.model
    finger_body_id = model.body(finger_body_name).id

    # panda.xml中主要指腹碰撞块的半尺寸。
    expected_half_size = np.array(
        [0.0085, 0.004, 0.0085]
    )

    candidate_ids = []

    for geom_id in range(model.ngeom):
        belongs_to_finger = (
            model.geom_bodyid[geom_id]
            == finger_body_id
        )

        is_box = (
            model.geom_type[geom_id]
            == mujoco.mjtGeom.mjGEOM_BOX
        )

        has_expected_size = np.allclose(
            model.geom_size[geom_id],
            expected_half_size,
        )

        if (
            belongs_to_finger
            and is_box
            and has_expected_size
        ):
            candidate_ids.append(geom_id)

    if len(candidate_ids) != 1:
        raise RuntimeError(
            f"Expected one main pad on "
            f"{finger_body_name}, "
            f"found {len(candidate_ids)}"
        )

    return candidate_ids[0]


def get_geom_offset_in_ee_frame(
    scene: PandaScene,
    geom_id: int,
) -> np.ndarray:
    """返回geom中心相对于ee_center_site的局部坐标。"""

    data = scene.data

    ee_site = data.site("ee_center_site")
    ee_position = ee_site.xpos.copy()
    ee_rotation = ee_site.xmat.reshape(3, 3).copy()

    geom_position = data.geom_xpos[geom_id].copy()

    # 两个点的差首先是在世界坐标系中表达。
    offset_world = geom_position - ee_position

    # R把末端局部向量转换到世界坐标系；
    # R.T则把世界坐标系向量转换回末端坐标系。
    offset_ee = ee_rotation.T @ offset_world

    return offset_ee
def main() -> None:
    """打印并检查当前模型的抓取几何。"""

    scene = PandaScene()
    model = scene.model
    data = scene.data

    # 保存初始状态，用于证明本探针没有修改仿真。
    qpos_before = data.qpos.copy()
    ctrl_before = data.ctrl.copy()
    time_before = float(data.time)

    # -------------------------
    # 1. 方块尺寸
    # -------------------------

    cube_geom = model.geom("cube_geom")

    # MuJoCo中box的size是半尺寸。
    cube_half_size = cube_geom.size.copy()
    cube_full_size = 2.0 * cube_half_size

    # -------------------------
    # 2. 末端位姿和局部轴
    # -------------------------

    ee_position, ee_rotation = get_ee_pose(scene)

    # 旋转矩阵的三列，是末端三个局部轴在世界坐标系中的方向。
    ee_x_axis_world = ee_rotation[:, 0]
    ee_y_axis_world = ee_rotation[:, 1]
    ee_z_axis_world = ee_rotation[:, 2]

    # -------------------------
    # 3. 左右手指关节轴
    # -------------------------

    left_axis_local, left_axis_world = get_joint_axis_world(
        scene,
        "finger_joint1",
    )
    right_axis_local, right_axis_world = get_joint_axis_world(
        scene,
        "finger_joint2",
    )
    left_pad_geom_id = find_main_fingertip_pad_geom_id(
        scene,
        "left_finger",
    )
    right_pad_geom_id = find_main_fingertip_pad_geom_id(
        scene,
        "right_finger",
    )

    left_pad_offset_ee = get_geom_offset_in_ee_frame(
        scene,
        left_pad_geom_id,
    )
    right_pad_offset_ee = get_geom_offset_in_ee_frame(
        scene,
        right_pad_geom_id,
    )
    # -------------------------
    # 4. 定义向下抓取姿态
    # -------------------------

    grasp_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    grasp_x_axis_world = grasp_rotation[:, 0]
    grasp_y_axis_world = grasp_rotation[:, 1]
    grasp_z_axis_world = grasp_rotation[:, 2]


    np.set_printoptions(precision=6, suppress=True)

    print("Cube geometry")
    print("half size:", cube_half_size)
    print("full size:", cube_full_size)

    print("\nEnd-effector geometry")
    print("position:", ee_position)
    print("local x-axis in world:", ee_x_axis_world)
    print("local y-axis in world:", ee_y_axis_world)
    print("local z-axis in world:", ee_z_axis_world)

    print("\nFinger joint axes")
    print("left local axis: ", left_axis_local)
    print("right local axis:", right_axis_local)
    print("left world axis: ", left_axis_world)
    print("right world axis:", right_axis_world)

    print("\nTarget top-down grasp orientation")
    print("rotation matrix:")
    print(grasp_rotation)
    print("local x-axis in world:", grasp_x_axis_world)
    print("local y-axis in world:", grasp_y_axis_world)
    print("local z-axis in world:", grasp_z_axis_world)

    print("\nFingertip pad offsets in EE frame")
    print("left pad: ", left_pad_offset_ee)
    print("right pad:", right_pad_offset_ee)
    # -------------------------
    # 5. 自动验收
    # -------------------------

    # 方块三个方向的完整边长都应为4厘米。
    np.testing.assert_allclose(
        cube_full_size,
        np.array([0.04, 0.04, 0.04]),
        atol=1e-12,
    )

    # 末端旋转矩阵必须是合法旋转矩阵。
    np.testing.assert_allclose(
        ee_rotation.T @ ee_rotation,
        np.eye(3),
        atol=1e-9,
    )
    assert np.isclose(
        np.linalg.det(ee_rotation),
        1.0,
        atol=1e-9,
    )

    # 两根手指的局部关节轴都在各自body的+y方向。
    np.testing.assert_allclose(
        left_axis_local,
        np.array([0.0, 1.0, 0.0]),
    )
    np.testing.assert_allclose(
        right_axis_local,
        np.array([0.0, 1.0, 0.0]),
    )

    # 由于左右手指镜像安装，它们的世界运动轴方向相反。
    np.testing.assert_allclose(
        left_axis_world,
        -right_axis_world,
        atol=1e-9,
    )

    # 左手指的运动轴与末端局部+y轴一致。
    np.testing.assert_allclose(
        left_axis_world,
        ee_y_axis_world,
        atol=1e-9,
    )

    # 右手指的运动轴与末端局部+y轴相反。
    np.testing.assert_allclose(
        right_axis_world,
        -ee_y_axis_world,
        atol=1e-9,
    )
        # 目标姿态必须是合法旋转矩阵。
    np.testing.assert_allclose(
        grasp_rotation.T @ grasp_rotation,
        np.eye(3),
        atol=1e-12,
    )
    assert np.isclose(
        np.linalg.det(grasp_rotation),
        1.0,
        atol=1e-12,
    )

    # 局部+y轴是夹爪张合方向。
    np.testing.assert_allclose(
        grasp_y_axis_world,
        np.array([0.0, -1.0, 0.0]),
    )

    # 局部+z轴是向下接近方向。
    np.testing.assert_allclose(
        grasp_z_axis_world,
        np.array([0.0, 0.0, -1.0]),
    )
    np.testing.assert_allclose(
        left_pad_offset_ee,
        np.array([0.0, 0.0455, -0.0021]),
        atol=1e-9,
    )

    np.testing.assert_allclose(
        right_pad_offset_ee,
        np.array([0.0, -0.0455, -0.0021]),
        atol=1e-9,
    )
    # 本程序只允许读取，不能修改状态或推进时间。
    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_array_equal(data.ctrl, ctrl_before)
    assert data.time == time_before

    print("\nGrasp geometry audit: PASSED")


if __name__ == "__main__":
    main()