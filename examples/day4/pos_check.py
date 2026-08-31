import os, sys
os.environ['PATH'] = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin') + ';' + os.environ['PATH']
import mujoco
import numpy as np
from panda_mujoco.simulation import PandaScene

POSE = [0.6, 0.0, 0.0, -1.57079, 0.0, 3.0, -1.7853]   # ← 改这里，验证你想加的路点

scene = PandaScene()
m = mujoco
s = scene
s.data.qpos[:7] = POSE
s.data.qvel[:] = 0
mujoco.mj_forward(s.model, s.data)

ok = True
for jid in range(7):                                    # 限位检查
    lo, hi = s.model.jnt_range[jid]
    if not lo <= POSE[jid] <= hi:
        print(f'✗ joint{jid+1} 超限: {POSE[jid]} ∉ [{lo}, {hi}]'); ok = False
n_self = sum(                                           # 自碰撞检查（排除地面和方块）
    1 for i in range(s.data.ncon)
    if 'cube' not in (mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, s.data.contact[i].geom1) or '') + \
       (mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, s.data.contact[i].geom2) or '')
    and 'floor' not in (mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, s.data.contact[i].geom1) or '') + \
        (mujoco.mj_id2name(s.model, mujoco.mjtObj.mjOBJ_GEOM, s.data.contact[i].geom2) or ''))
print('末端位置:', np.round(s.ee_pos, 3), ' 自碰撞接触数:', n_self)
print('✓ 安检通过' if ok and n_self == 0 else '✗ 此姿态不可用')