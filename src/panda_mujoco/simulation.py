"""Panda 仿真基础模块（Day 3 交付）。

全项目唯一负责"加载模型 + 重置状态"的地方：模型路径、home 位姿、
reset 细节都收拢在本模块，之后的 demo / 测试 / Gym 环境一律通过
    from panda_mujoco.simulation import PandaScene
来使用，绝不自己写 from_xml_path 或 home 数值。

设计要点：
- reset 是"绝对写入"而非"增量调整"——不依赖调用前的状态，天然幂等，
  连续 reset 10 次结果逐位一致（Day 3 验收）；
- home 姿态取自 panda.xml 中被注释的官方 home keyframe，但把 ctrl
  与 qpos 对齐（原 keyframe 的 joint6/7 两者不一致，会导致开机甩动）。
"""

from pathlib import Path

import mujoco
import numpy as np

# 默认场景路径：从本文件反推项目根。
#   parents[0]=panda_mujoco/
#   parents[1]=src/
#   parents[2]=panda_mujoco_manipulation/
# Day 6 后，模型资产已经复制进仓库内的 assets/robots/panda，
# 因此代码不再依赖外部的 D:\Mujoco Project\model 目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = PROJECT_ROOT / "assets" / "robots" / "panda" / "scene_with_cube.xml"

# home 位姿：9 个数 = 7 个手臂关节角（弧度）+ 2 个手指行程（米）。
# 来源：panda.xml 被注释的 home keyframe 的 qpos 行。
# 关键值：joint4=-1.57079（肘部弯 90°，落在限位 [-3.0718,-0.0698] 中段，
# 替代非法的默认 0 位）；双指 0.04 = 张开待命（实测 0=闭合，0.04=张开）。
PANDA_HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 3.0, -1.7853, 0.04, 0.04]
)

# 方块初始状态：7 个数 = 位置 xyz + 四元数 wxyz（(1,0,0,0) 即无旋转）。
# 与 scene_with_cube.xml 中 <body name="cube" pos="0.45 0 0.05"> 一致。
CUBE_HOME_QPOS = np.array([0.45, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0])

# home 控制量：8 个数 = 7 个电机目标角度 + 1 个夹爪命令。
# 前 7 格直接复制 home 关节角（ctrl == qpos → 位置伺服无拉扯，姿态静止）；
# 夹爪 255 = 目标 0.04 m = 张开，与双指 home 值一致。
HOME_CTRL = np.concatenate([PANDA_HOME_QPOS[:7], [255.0]])


class PandaScene:
    """对 MjModel/MjData 的最小封装：加载一次，随时可靠地回到 home。"""

    def __init__(self, scene_path=None):
        # scene_path=None 时用默认场景；传入 Path/字符串可换场景（测试/实验用）
        self.scene_path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE
        # MjModel：编译后的静态结构（关节/限位/执行器，只读）
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        # MjData：运行时状态（qpos/qvel/ctrl/坐标，每步被重写）
        self.data = mujoco.MjData(self.model)
        # 创建后立刻落到 home，保证"刚构造出来的对象"就是已知安全状态
        self.reset_to_home()

    def reset_to_home(self):
        """把整个场景重置为 home 状态。幂等：调多少次结果都逐位相同。

        步骤（每步对应一条 Day 3 验收）：
        1. 机械臂 9 关节写入 PANDA_HOME_QPOS   → joint4 不再非法启动
        2. 方块 7 状态写入 CUBE_HOME_QPOS      → 方块回到标准位置
        3. qvel 清零、ctrl 写 home_ctrl        → 无上一步残留
        4. mj_forward 重算派生量               → site 坐标等读数立即可用
        5. 限位 + 有限性断言                   → 合法性由机器把关
        """
        d = self.data
        d.qpos[:9] = PANDA_HOME_QPOS      # 9 个单值关节（下标见审计表 qposadr）
        d.qpos[9:16] = CUBE_HOME_QPOS     # 方块自由关节（qposadr=9，占 7 格）
        d.qvel[:] = 0.0                   # 15 维速度全清零
        d.ctrl[:] = HOME_CTRL             # 8 格控制量归位
        mujoco.mj_forward(self.model, d)  # 只重算派生量，不推进时间
        self._assert_valid()

    def _assert_valid(self):
        """reset 后的机器验收：单值关节全部在限位内，全部状态有限。"""
        d, m = self.data, self.model
        for jid in range(m.njnt):
            if not m.jnt_limited[jid]:    # FREE 关节无限位，跳过
                continue
            lo, hi = m.jnt_range[jid]
            q = d.qpos[m.jnt_qposadr[jid]]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            # 容差 1e-9 抵消浮点表示误差
            assert lo - 1e-9 <= q <= hi + 1e-9, f"{name} qpos={q} 超出限位 [{lo}, {hi}]"
        assert np.all(np.isfinite(d.qpos)), "qpos 出现 NaN/Inf"
        assert np.all(np.isfinite(d.qvel)), "qvel 出现 NaN/Inf"
        assert np.all(np.isfinite(d.ctrl)), "ctrl 出现 NaN/Inf"

    # ---- 便捷读数（copy() 防止外部改动内部数组）----

    @property
    def ee_pos(self):
        """末端 site 的世界坐标 (x, y, z)。Day 2 审计确认场景唯一 site。"""
        return self.data.site("ee_center_site").xpos.copy()

    @property
    def cube_pos(self):
        """方块位置 (x, y, z)：自由关节 qpos 的前 3 格。"""
        return self.data.qpos[9:12].copy()
