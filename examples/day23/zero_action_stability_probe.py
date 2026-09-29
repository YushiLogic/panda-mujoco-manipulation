import numpy as np

from panda_mujoco.continuous_env import PandaContinuousEnv
from panda_mujoco.kinematics import get_ee_position

env = PandaContinuousEnv()
observation, info = env.reset(seed=42)

start_position = get_ee_position(env.task.scene).copy()
start_target = env.target_position.copy()
zero_action = np.zeros(7, dtype=np.float32)

for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(zero_action)

final_position = get_ee_position(env.task.scene)
drift_mm = np.linalg.norm(final_position - start_position) * 1000

print("steps:", info["episode_steps"])
print("target unchanged:", np.array_equal(env.target_position, start_target))
print("end-effector drift mm:", drift_mm)
print("observation finite:", np.all(np.isfinite(observation)))
print("action duration s:", 100 * info["control_period_s"])

env.close()