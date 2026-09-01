"""接触检测助手（Day 5 交付）——把原始碰撞数据变成任务语义。

MuJoCo 每个物理步都会做碰撞检测，把"谁碰到了谁"写进 data.contact
数组。每个条目有 geom1/geom2（两个碰撞几何体的 ID）、位置、法线等；
它只描述当前物理步，不会自动保存历史接触。

本项目的一个关键事实（Day 5 侦察发现）：手指的 geom 全部没有名字
且每根手指由多个碰撞 geom 组成，所以不能只查一个 geom 名称。更稳定
的识别层是 body 名称（left_finger / right_finger / cube）：先收集每个
body 直接拥有的 geom ID，再检查 contact 的两个 geom 是否命中集合。

概念链路：
    body 名称 -> body ID -> 该 body 的所有 geom ID
    (contact.geom1, contact.geom2) -> 是否分别来自手指集合和目标集合

用法：
    sensor = ContactSensor(model)                # 预先算好手指/方块的 geom 集合
    left, right = sensor.finger_contacts(data)   # 本步左右指与方块的接触条数
    if sensor.is_grasped(data): ...              # 仅为 Day 5 双侧接触代理

注意：接触条目数不是接触力；当前 is_grasped 也不等于完整抓取成功。
真实抓取还要检查持续时间、物体是否离地、是否跟随夹爪以及是否掉落。
"""

import mujoco


def body_geoms(model, body_name):
    """返回由指定 body 直接拥有的所有 geom ID。

    model.geom_bodyid[g] 保存 geom g 所属的 body ID。这里扫描静态模型，
    筛出目标 body 的 geom；该操作只在 ContactSensor 构造时执行一次。
    """

    # mj_name2id 找不到名称时返回 -1，而不是抛出 Python 异常。显式检查
    # 可以避免悄悄得到空集合并把所有接触错误地判断为 False。
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        raise ValueError(f"模型里不存在名为 {body_name!r} 的 body")
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


class ContactSensor:
    """把当前 data.contact 分类为左指/右指与目标物体的接触。"""

    def __init__(self, model,
                 left_body="left_finger", right_body="right_finger",
                 target_body="cube"):
        # body 与 geom 的归属属于 MjModel 静态结构，仿真时不会变化。
        # 转成 set 后，每个物理步只需做快速集合成员判断。
        self.left = set(body_geoms(model, left_body))
        self.right = set(body_geoms(model, right_body))
        self.target = set(body_geoms(model, target_body))

    def _count(self, data, finger_set):
        """统计某根手指与目标物体之间的当前接触条目数。

        geom1/geom2 没有“手指在前、方块在后”的顺序保证，因此把双方
        组成无序 set，再分别检查是否命中手指集合和目标集合。
        """
        n = 0
        # data.contact 是预分配数组，只有前 data.ncon 条在当前步有效。
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            # 两个交集都非空，表示接触双方中一个来自指定手指，另一个
            # 来自目标物；满足后才计为“手指-目标”接触。
            if len(pair & finger_set) and len(pair & self.target):
                n += 1
        return n

    def finger_contacts(self, data):
        """返回本物理步的 (左指接触条数, 右指接触条数)。"""
        return self._count(data, self.left), self._count(data, self.right)

    def is_grasped(self, data):
        """返回当前是否为双侧接触（Day 5 的简化抓取代理）。

        当前实现只证明“一帧内左右手指都接触目标”，不能证明物体已经
        离地、能跟随夹爪运动或持续不掉落。第 3 周会扩展完整成功条件。
        """
        left, right = self.finger_contacts(data)
        return left > 0 and right > 0
