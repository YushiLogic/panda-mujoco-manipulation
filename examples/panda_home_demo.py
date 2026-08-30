"""Day 3 演示：加载场景 → reset 到 home → 在 viewer 中观察。

观察要点：
1. 机械臂应静止站在 home 姿态（肘部微弯、夹爪张开），无开机甩动；
2. 方块静静躺在 (0.45, 0, 0.02) 附近（从 0.05 落到地面）；
3. 控制台每 2 秒打印一次末端坐标——数值应当几乎不动（稳定性）。
"""

import time

import mujoco
import mujoco.viewer
from panda_mujoco.simulation import PandaScene


def main():
    scene = PandaScene()  # 构造时已自动 reset 到 home
    print("home 关节角 :", scene.data.qpos[:9])
    print("home 末端位姿:", scene.ee_pos)
    print("home 方块位置:", scene.cube_pos)

    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        step = 0
        while viewer.is_running():
            mujoco.mj_step(scene.model, scene.data)  # 推进物理（home 下应纹丝不动）
            viewer.sync()
            step += 1
            if step % 1000 == 0:  # 约 2 秒打印一次，观察漂移
                print(f"t={scene.data.time:5.1f}s  ee={scene.ee_pos}  cube={scene.cube_pos}")
            time.sleep(0.001)  # 轻微限速，避免打印刷屏过快


if __name__ == "__main__":
    main()
