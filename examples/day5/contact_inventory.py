
"""Day 5 实验 3：把 MuJoCo 原始接触翻译成人能读懂的信息。

每个物理步，MuJoCo 都会重新计算 data.contact。每条记录直接给出两个
碰撞几何体的整数 ID（geom1、geom2），而不是“方块碰到地面”这样的
业务语义。本实验沿以下链路查询：

    contact 的 geom ID -> geom 名称 -> 所属 body ID -> body 名称

data.ncon 是当前物理步的接触点数量，不是接触物体数，也不是接触力。
"""

import mujoco

from panda_mujoco.simulation import PandaScene


def object_name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
) -> str:
    """根据对象类型和 ID 查名称；未命名对象显示为 ``<unnamed>``。

    同一个整数 ID 在 body 表和 geom 表中含义不同，因此调用时必须同时
    指定 object_type。模型允许某些碰撞 geom 不设置 name。
    """

    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else "<unnamed>"


def main() -> None:
    # 新建场景时 cube 位于 z=0.05，尚未落地，通常还没有 floor 接触。
    scene = PandaScene()
    model = scene.model
    data = scene.data

    # 推进 1 s，让 cube 在重力下落到地面。按仿真时间计算，不需要 sleep。
    num_steps = int(round(1.0 / model.opt.timestep))

    for _ in range(num_steps):
        mujoco.mj_step(model, data)

    # data.contact 只代表当前这一物理步，不会自动累积历史 contact。
    print(f"time={data.time:.3f} s")
    print(f"number of contacts={data.ncon}")

    for contact_index in range(data.ncon):
        # contact 数组是预分配空间，只有前 data.ncon 条在当前步有效。
        contact = data.contact[contact_index]

        # geom1/geom2 是 geom 表下标。两者没有固定业务顺序，不能假设
        # geom1 永远是 floor、geom2 永远是 cube。
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)

        # geom_bodyid 把碰撞形状映射回刚体：例如 cube_geom 属于 cube，
        # floor 属于 world。后续手指接触识别也依赖这层映射。
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])

        geom1_name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom1_id,
        )
        geom2_name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom2_id,
        )

        body1_name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body1_id,
        )
        body2_name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body2_id,
        )

        # dist < 0 表示软接触允许的微小压缩；dist 不是接触力。真正的
        # 接触力要结合求解结果，通过 mujoco.mj_contactForce 另外读取。
        print(
            f"contact[{contact_index}] | "
            f"geom1={geom1_name}({geom1_id}), "
            f"body1={body1_name}({body1_id}) | "
            f"geom2={geom2_name}({geom2_id}), "
            f"body2={body2_name}({body2_id}) | "
            f"distance={contact.dist:.3e} m"
        )


# 直接运行脚本时执行；作为模块导入时不自动启动仿真。
if __name__ == "__main__":
    main()
