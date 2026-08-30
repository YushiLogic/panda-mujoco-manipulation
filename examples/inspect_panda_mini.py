"""极简版模型审计（学习用，配套 examples/inspect_panda.py 完整版）。

只做四件事，从头到尾一条线，没有函数、没有参数解析：
    1. 编译模型
    2. 按名字查对象（查不到返回 -1）
    3. 打印 nq/nv/nu
    4. 检查 joint4 + 打印末端和方块坐标
先 100% 看懂这个，再看完整版就只差"工程包装"。
"""

import mujoco

# 极简版直接写死路径，聚焦逻辑本身（完整版用 Path(__file__) 反推，目的相同）。
# 字符串前的 r 表示"原始字符串"：反斜杠 \ 不当转义符用，专治 Windows 路径。
MODEL_PATH = r"D:\Mujoco Project\model\franka_emika_panda\scene_with_cube.xml"

# ---------- 1. 编译模型 ----------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)  # MjModel：静态结构（只读）
data = mujoco.MjData(model)  # MjData：运行时状态（qpos、坐标等，可变）

# ---------- 2. 查名字：名字 -> 整数 id，-1 = 不存在 ----------
# (对象类型, 名字) 列表：后续代码要按名字用到的关键对象
must_exist = [
    (mujoco.mjtObj.mjOBJ_JOINT, "joint1"),
    (mujoco.mjtObj.mjOBJ_JOINT, "joint4"),
    (mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
    (mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"),
    (mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"),
    (mujoco.mjtObj.mjOBJ_SITE, "ee_center_site"),
]
for obj_type, name in must_exist:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    print(f"{name:<16} -> id = {obj_id}")
    if obj_id == -1:  # 有任何一个查不到，审计就失败
        print("AUDIT FAILED")
        raise SystemExit(1)  # 让进程以失败状态退出（成功时默认退出码为 0）

# ---------- 3. 打印三个维度 ----------
# nq = qpos 数组长度；nv = 速度自由度；nu = 执行器个数
# 手工拆解：9 个普通关节各占 1 个 qpos + cube 自由关节占 7 个 qpos = 16
#          9 个普通关节各占 1 个 dof + cube 自由关节占 6 个 dof = 15
print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")

# ---------- 4a. joint4 隐患检查 ----------
joint4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint4")
print("joint4 限位      :", model.jnt_range[joint4_id])  # 按关节 id 查它的限位
# joint4 的数值存在 qpos0 的第几个下标？规范写法是查地址表（单值关节恰好等于 id）
adr = model.jnt_qposadr[joint4_id]
print("joint4 默认关节角:", model.qpos0[adr])
print("-> 默认值 0 不在 [-3.0718, -0.0698] 里，所以 Day3 必须显式设 home")

# ---------- 4b. 末端与方块的世界坐标 ----------
# mj_forward：由"当前关节角"重算一切派生量（含 site 世界坐标），时间不动
mujoco.mj_forward(model, data)
print("末端 site 坐标:", data.site("ee_center_site").xpos)  # 命名访问，免查 id
print("方块位置      :", data.joint("cube_joint").qpos[:3])  # 7 维 qpos 的前 3 个是 xyz

print("AUDIT PASSED")
