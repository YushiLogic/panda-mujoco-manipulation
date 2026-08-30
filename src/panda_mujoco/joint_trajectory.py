"""关节空间轨迹规划器（Day 4 交付）。

MoveJ（关节空间移动）的核心思想：不关心末端走出的形状，只让
所有关节**同步、匀速**地从上一目标滑向下一目标——数学核心就是
一行线性插值：

    目标(t) = 起点 + (终点 - 起点) × 段内进度 s,  s ∈ [0, 1]

按"控制拍"推进（而不是物理步），与仿真步频解耦：
    物理频率 = 1/timestep = 500 Hz（每个 mj_step 走 2 ms）
    控制频率 = 物理频率 / CONTROL_EVERY（本 demo 为 50 Hz）
真机器人也是这个结构（内部 1 kHz，命令 50~500 Hz），所以这个
规划器将来可以直接搬到真机控制循环里。
"""

import numpy as np


class JointPath:
    """在若干关节目标之间做线性插值的路点发生器。

    用法（每个控制拍）：
        ctrl[:7] = plan.target()   # 取当前插值目标发给电机
        plan.advance()             # 推进一个控制拍

    约定：机械臂的当前关节角等于 goals[0]（例如刚 reset 到 home），
    规划器依次走 goals[0]→goals[1]→goals[2]→...
    """

    def __init__(self, goals, ticks_per_move=150):
        # goals：2 维以上列表，每个元素是 7 维关节角；统一转成 float 数组
        self.goals = [np.asarray(g, dtype=float) for g in goals]
        self.ticks_per_move = ticks_per_move  # 走完一段需要多少个控制拍
        self.seg = 0    # 当前段编号：seg=0 表示从 goals[0] 走向 goals[1]
        self.tick = 0   # 段内已经走了多少个控制拍

    def target(self):
        """返回当前控制拍应下发的插值目标（7 维关节角）。"""
        if self.finished():
            return self.goals[-1].copy()     # 走完后永远保持最后一个目标
        start = self.goals[self.seg]
        end = self.goals[self.seg + 1]
        s = self.tick / self.ticks_per_move  # 段内进度，0→1 线性增长
        return start + (end - start) * s     # ← MoveJ 的全部数学就这一行

    def advance(self):
        """推进一个控制拍；走满一段自动切入下一段。"""
        if self.finished():
            return
        self.tick += 1
        if self.tick >= self.ticks_per_move:
            self.tick = 0
            self.seg += 1                    # 切入下一段（调用方可借此检测"段完成"）

    def finished(self):
        """所有段都走完了吗？"""
        return self.seg >= len(self.goals) - 1

    def current_goal(self):
        """当前这段的目的地（画误差曲线、做到达判定时用）。"""
        return self.goals[min(self.seg + 1, len(self.goals) - 1)]
