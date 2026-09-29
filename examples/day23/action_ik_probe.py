import mujoco

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.gripper import command_gripper
import numpy as np

from panda_mujoco.action_mapping import map_action_to_targets
from panda_mujoco.gripper import get_gripper_width
from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.motion import plan_pose_target
from panda_mujoco.simulation import PandaScene


scene = PandaScene()
start_position, start_rotation = get_ee_pose(scene)
start_width = get_gripper_width(scene)

# 只要求末端沿世界 +x 移动 2 mm。
action = np.array([1, 0, 0, 0, 0, 0, 0])
target_position, target_rotation, target_width = map_action_to_targets(
    action, start_position, start_rotation, start_width
)

# 保存控制场景的状态，用来检查规划有没有让机器人“瞬移”。
qpos_before = scene.data.qpos.copy()
ctrl_before = scene.data.ctrl.copy()
time_before = scene.data.time

plan = plan_pose_target(scene, target_position, target_rotation)

print("target displacement mm:", (target_position - start_position) * 1000)
print("IK success:", plan.success)
print("IK iterations:", plan.ik_iterations)
print("IK position error mm:", plan.position_error_norm * 1000)
print("joint target:", plan.joint_target)
print("qpos unchanged:", np.array_equal(scene.data.qpos, qpos_before))
print("ctrl unchanged:", np.array_equal(scene.data.ctrl, ctrl_before))
print("time unchanged:", scene.data.time == time_before)
if not plan.success:
    raise RuntimeError("IK could not find a joint target")

# 发送目标命令；这两行本身仍不会推进仿真。
command_arm_joint_positions(scene, plan.joint_target)
command_gripper(scene, target_width)

# 一个环境动作对应 10 个 MuJoCo 物理步。
for _ in range(10):
    apply_arm_bias_compensation(scene)
    mujoco.mj_step(scene.model, scene.data)

actual_position, _ = get_ee_pose(scene)

print("simulation time s:", scene.data.time)
print("actual displacement mm:", (actual_position - start_position) * 1000)
print("remaining error mm:", (target_position - actual_position) * 1000)

error_after_10_steps = np.linalg.norm(
    target_position - actual_position
) * 1000

# 不发送新目标；data.ctrl 会保持上一条命令。
for _ in range(100):
    apply_arm_bias_compensation(scene)
    mujoco.mj_step(scene.model, scene.data)

held_position, _ = get_ee_pose(scene)
error_after_hold = np.linalg.norm(
    target_position - held_position
) * 1000

print("error after first 10 steps mm:", error_after_10_steps)
print("error after 100 more steps mm:", error_after_hold)
print("total simulation time s:", scene.data.time)