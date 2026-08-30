"""Day 3 验收测试：reset 的可靠性、一致性与合法性。

运行方式（仓库根目录）：
    python -m pytest
"""

import numpy as np
import pytest

from panda_mujoco.simulation import CUBE_HOME_QPOS, PANDA_HOME_QPOS, PandaScene


def test_reset_10_times_identical():
    """验收项：连续 reset 10 次，全部状态逐位一致。"""
    scene = PandaScene()
    snapshots = []
    for _ in range(10):
        scene.reset_to_home()
        snapshots.append(
            np.concatenate([scene.data.qpos, scene.data.qvel, scene.data.ctrl])
        )
    for snap in snapshots[1:]:
        assert np.array_equal(snapshots[0], snap)  # 不用 allclose——要求逐位相等


def test_joint4_starts_legal():
    """验收项：joint4 不再以非法 0 位启动（其限位 [-3.0718,-0.0698] 不含 0）。"""
    scene = PandaScene()
    # pytest.approx：浮点数专用断言，容忍 1e-9 级表示误差
    assert scene.data.qpos[3] == pytest.approx(-1.57079)
    lo, hi = -3.0718, -0.0698
    assert lo <= scene.data.qpos[3] <= hi


def test_reset_clears_polluted_state():
    """验收项：无残留——把状态污染成离谱值后 reset，必须完全恢复。"""
    scene = PandaScene()
    scene.data.qpos[:] = 999.0
    scene.data.qvel[:] = 42.0
    scene.data.ctrl[:] = -7.0
    scene.reset_to_home()
    assert np.all(np.isfinite(scene.data.qpos))
    assert np.all(scene.data.qvel == 0.0)                      # 速度严格清零
    assert np.array_equal(scene.data.qpos[:9], PANDA_HOME_QPOS)
    assert np.array_equal(scene.data.qpos[9:16], CUBE_HOME_QPOS)


def test_home_ctrl_aligned_with_qpos():
    """设计约束：home 下 ctrl 前 7 格等于关节角，伺服无拉扯、姿态静止。"""
    scene = PandaScene()
    assert np.array_equal(scene.data.ctrl[:7], scene.data.qpos[:7])
    assert scene.data.ctrl[7] == 255.0  # 夹爪张开，与双指 0.04 一致


def test_all_single_joints_within_limits():
    """验收项：9 个单值关节全部在限位内（FREE 关节除外）。"""
    scene = PandaScene()
    m, d = scene.model, scene.data
    for jid in range(m.njnt):
        if not m.jnt_limited[jid]:
            continue
        lo, hi = m.jnt_range[jid]
        q = d.qpos[m.jnt_qposadr[jid]]
        assert lo - 1e-9 <= q <= hi + 1e-9, f"joint id={jid} 超限: {q}"
