"""Day 16：验证单次方块复位与落稳过程。"""

import numpy as np

from panda_mujoco.cube_reset import (
    DEFAULT_SPAWN_HEIGHT,
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    DEFAULT_YAW_RANGE,
    CubeSettleResult,
    get_cube_joint_addresses,
    reset_cube,
    sample_cube_pose,
    settle_cube,
)
from panda_mujoco.simulation import PandaScene


TEST_SEED = 42

# 方块半高为0.02 m，所以理论落稳中心高度为0.02 m。
EXPECTED_SETTLED_HEIGHT = 0.02

# 接触模型允许非常小的弹性穿透。
HEIGHT_TOLERANCE = 5e-4

# 方块竖直下落时，xy方向不应出现明显漂移。
XY_DRIFT_TOLERANCE = 1e-4

BATCH_CASE_COUNT = 20


def check_sampling_reproducibility() -> None:
    """检查seed复现性和采样范围。"""

    first = sample_cube_pose(42)
    repeated = sample_cube_pose(42)
    different = sample_cube_pose(43)

    first_qpos = np.concatenate(
        [
            first.position,
            first.quaternion,
        ]
    )

    repeated_qpos = np.concatenate(
        [
            repeated.position,
            repeated.quaternion,
        ]
    )

    different_qpos = np.concatenate(
        [
            different.position,
            different.quaternion,
        ]
    )

    same_seed_identical = np.array_equal(
        first_qpos,
        repeated_qpos,
    )

    different_seed_different = (
        not np.array_equal(
            first_qpos,
            different_qpos,
        )
    )

    print("Sampling reproducibility")
    print(
        "same seed identical:",
        same_seed_identical,
    )
    print(
        "different seed different:",
        different_seed_different,
    )

    assert same_seed_identical
    assert different_seed_different

    # 检查两个代表样本是否都在规定范围内。
    for pose in (first, different):
        assert (
            DEFAULT_X_RANGE[0]
            <= pose.position[0]
            <= DEFAULT_X_RANGE[1]
        )

        assert (
            DEFAULT_Y_RANGE[0]
            <= pose.position[1]
            <= DEFAULT_Y_RANGE[1]
        )

        assert (
            pose.position[2]
            == DEFAULT_SPAWN_HEIGHT
        )

        assert (
            DEFAULT_YAW_RANGE[0]
            <= pose.yaw
            <= DEFAULT_YAW_RANGE[1]
        )

        assert np.isclose(
            np.linalg.norm(
                pose.quaternion
            ),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )

    print(
        "Sampling reproducibility: PASSED"
    )


def count_contacts_between(
    scene: PandaScene,
    first_geom_name: str,
    second_geom_name: str,
) -> int:
    """统计两个指定geom之间的接触点数量。"""

    first_geom_id = scene.model.geom(
        first_geom_name
    ).id

    second_geom_id = scene.model.geom(
        second_geom_name
    ).id

    count = 0

    for contact_index in range(
        scene.data.ncon
    ):
        contact = scene.data.contact[
            contact_index
        ]

        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        same_order = (
            geom1 == first_geom_id
            and geom2 == second_geom_id
        )

        reversed_order = (
            geom1 == second_geom_id
            and geom2 == first_geom_id
        )

        if same_order or reversed_order:
            count += 1

    return count


def check_batch_resets(
    scene: PandaScene,
) -> None:
    """连续验证20个seed的复位和落稳。"""

    _, qvel_address = (
        get_cube_joint_addresses(
            scene.model
        )
    )

    maximum_height_error = 0.0
    maximum_xy_drift = 0.0
    maximum_linear_speed = 0.0
    maximum_angular_speed = 0.0

    print(
        "\nTwenty-seed reset and settling check"
    )

    for seed in range(BATCH_CASE_COUNT):
        # 故意复用同一个scene。
        # 这样才能检查上一轮是否污染下一轮。
        scene.reset_to_home()

        pose = sample_cube_pose(seed)

        time_before_reset = float(
            scene.data.time
        )

        reset_cube(scene, pose)

        time_after_reset = float(
            scene.data.time
        )

        qvel_after_reset = scene.data.qvel[
            qvel_address:qvel_address + 6
        ].copy()

        initial_floor_contacts = (
            count_contacts_between(
                scene,
                "cube_geom",
                "floor",
            )
        )

        # -------------------------
        # reset瞬间的检查
        # -------------------------

        assert (
            time_after_reset
            == time_before_reset
        )

        np.testing.assert_array_equal(
            qvel_after_reset,
            np.zeros(6),
        )

        assert initial_floor_contacts == 0

        assert (
            DEFAULT_X_RANGE[0]
            <= pose.position[0]
            <= DEFAULT_X_RANGE[1]
        )

        assert (
            DEFAULT_Y_RANGE[0]
            <= pose.position[1]
            <= DEFAULT_Y_RANGE[1]
        )

        assert (
            pose.position[2]
            == DEFAULT_SPAWN_HEIGHT
        )

        assert (
            DEFAULT_YAW_RANGE[0]
            <= pose.yaw
            <= DEFAULT_YAW_RANGE[1]
        )

        assert np.isclose(
            np.linalg.norm(
                pose.quaternion
            ),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )

        # -------------------------
        # 动力学落稳
        # -------------------------

        result = settle_cube(
            scene,
            duration=1.5,
        )

        final_floor_contacts = (
            count_contacts_between(
                scene,
                "cube_geom",
                "floor",
            )
        )

        height_error = abs(
            result.final_position[2]
            - EXPECTED_SETTLED_HEIGHT
        )

        xy_drift = float(
            np.linalg.norm(
                result.final_position[:2]
                - pose.position[:2]
            )
        )

        # -------------------------
        # 落稳后的检查
        # -------------------------

        assert result.settled

        assert (
            height_error
            <= HEIGHT_TOLERANCE
        )

        assert (
            xy_drift
            <= XY_DRIFT_TOLERANCE
        )

        # 落稳后必须与地面形成接触。
        assert final_floor_contacts > 0

        assert np.all(
            np.isfinite(
                result.final_position
            )
        )

        maximum_height_error = max(
            maximum_height_error,
            height_error,
        )

        maximum_xy_drift = max(
            maximum_xy_drift,
            xy_drift,
        )

        maximum_linear_speed = max(
            maximum_linear_speed,
            result.linear_speed,
        )

        maximum_angular_speed = max(
            maximum_angular_speed,
            result.angular_speed,
        )

        print(
            f"seed={seed:02d} | "
            f"spawn=({pose.position[0]:.4f}, "
            f"{pose.position[1]:+.4f}) | "
            f"yaw={np.degrees(pose.yaw):+7.3f} deg | "
            f"initial contacts="
            f"{initial_floor_contacts} | "
            f"final contacts="
            f"{final_floor_contacts} | "
            f"z={result.final_position[2]:.7f} m | "
            f"settled={result.settled}"
        )

    print("\nBatch summary")
    print(
        "cases:",
        BATCH_CASE_COUNT,
    )
    print(
        "maximum height error:",
        maximum_height_error,
    )
    print(
        "maximum xy drift:",
        maximum_xy_drift,
    )
    print(
        "maximum linear speed:",
        maximum_linear_speed,
    )
    print(
        "maximum angular speed:",
        maximum_angular_speed,
    )

    print(
        "Twenty-seed reset and settling: PASSED"
    )


def main() -> None:
    """执行一次完整的reset和settle验证。"""

    # -------------------------
    # 1. Arrange：准备测试
    # -------------------------
    check_sampling_reproducibility()
    scene = PandaScene()

    qpos_address, qvel_address = (
        get_cube_joint_addresses(
            scene.model
        )
    )

    pose = sample_cube_pose(TEST_SEED)

    # reset不应推进仿真时间，因此先保存时间。
    time_before_reset = float(
        scene.data.time
    )

    # -------------------------
    # 2. Act：执行方块复位
    # -------------------------

    reset_cube(scene, pose)

    # 立即复制reset后的状态。
    # 必须使用copy，否则后续mj_step会继续修改原数组。
    qpos_after_reset = scene.data.qpos[
        qpos_address:qpos_address + 7
    ].copy()

    qvel_after_reset = scene.data.qvel[
        qvel_address:qvel_address + 6
    ].copy()

    body_position_after_reset = (
        scene.data.body("cube").xpos.copy()
    )

    time_after_reset = float(
        scene.data.time
    )

    print("State immediately after reset")
    print("sampled position:", pose.position)
    print("sampled yaw:", pose.yaw)
    print(
        "sampled yaw degrees:",
        np.degrees(pose.yaw),
    )
    print(
        "sampled quaternion:",
        pose.quaternion,
    )
    print("qpos:", qpos_after_reset)
    print("qvel:", qvel_after_reset)
    print(
        "body position:",
        body_position_after_reset,
    )
    print(
        "simulation time:",
        time_after_reset,
    )

    # -------------------------
    # 3. Assert：检查复位瞬间
    # -------------------------

    expected_qpos = np.concatenate(
        [
            pose.position,
            pose.quaternion,
        ]
    )

    # qpos必须与采样结果逐位一致。
    np.testing.assert_array_equal(
        qpos_after_reset,
        expected_qpos,
    )

    # 复位后六维速度必须严格清零。
    np.testing.assert_array_equal(
        qvel_after_reset,
        np.zeros(6),
    )

    # mj_forward后，body世界位置应与qpos位置一致。
    np.testing.assert_allclose(
        body_position_after_reset,
        pose.position,
        atol=1e-12,
        rtol=0.0,
    )

    # reset只更新状态，不能推进仿真时间。
    assert time_after_reset == time_before_reset

    # 方块生成高度应为配置的0.05 m。
    assert (
        pose.position[2]
        == DEFAULT_SPAWN_HEIGHT
    )

    # -------------------------
    # 4. Act：推进动力学落稳
    # -------------------------

    result = settle_cube(
        scene,
        duration=1.5,
    )

    # result现在是CubeSettleResult的实例。
    assert isinstance(
        result,
        CubeSettleResult,
    )

    xy_drift = float(
        np.linalg.norm(
            result.final_position[:2]
            - pose.position[:2]
        )
    )

    print("\nState after settling")
    print("settled:", result.settled)
    print("steps:", result.steps)
    print(
        "simulation duration:",
        result.simulation_duration,
    )
    print(
        "final position:",
        result.final_position,
    )
    print(
        "linear speed:",
        result.linear_speed,
    )
    print(
        "angular speed:",
        result.angular_speed,
    )
    print("xy drift:", xy_drift)

    # -------------------------
    # 5. Assert：检查落稳结果
    # -------------------------

    # 线速度和角速度都必须满足稳定阈值。
    assert result.settled

    # 最终高度应接近方块半高0.02 m。
    assert np.isclose(
        result.final_position[2],
        EXPECTED_SETTLED_HEIGHT,
        atol=HEIGHT_TOLERANCE,
        rtol=0.0,
    )

    # 竖直下落不应产生明显的水平漂移。
    assert xy_drift <= XY_DRIFT_TOLERANCE

    # 所有最终结果必须是有限数。
    assert np.all(
        np.isfinite(
            result.final_position
        )
    )
    assert np.isfinite(
        result.linear_speed
    )
    assert np.isfinite(
        result.angular_speed
    )

    print(
        "\nSingle cube reset and settling: PASSED"
    )

    check_batch_resets(scene)

    print(
        "\nDay 16 cube reset probe: PASSED"
    )


if __name__ == "__main__":
    main()
