"""接触与抓取状态监测——把原始仿真数据变成任务语义。

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

Day 5 的基础用法：
    sensor = ContactSensor(model)                # 预先算好手指/方块的 geom 集合
    left, right = sensor.finger_contacts(data)   # 本步左右指与方块的接触条数
    if sensor.is_grasped(data): ...              # 仅为单帧双侧接触代理

Day 18 在此基础上加入 BilateralContactTracker、GraspStatus 和
GraspMonitor，进一步检查持续时间、夹爪宽度、物体是否离地以及速度。
注意：接触条目数不是接触力；单帧双侧接触也不等于完整抓取成功。
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from panda_mujoco.cube_reset import get_cube_joint_addresses
from panda_mujoco.gripper import get_gripper_width
from panda_mujoco.simulation import PandaScene


DEFAULT_CONTACT_DURATION = 0.1
DEFAULT_MIN_GRASP_WIDTH = 0.005
DEFAULT_MAX_GRASP_WIDTH = 0.075
DEFAULT_MIN_LIFT_HEIGHT = 0.05
DEFAULT_MAX_LINEAR_SPEED = 0.02
DEFAULT_MAX_ANGULAR_SPEED = 0.2


@dataclass(frozen=True)
class ContactSnapshot:
    """记录某一个物理步中的接触分类结果。

    这个类只描述当前帧，不累计历史，也不直接判断稳定抓取。
    """

    left_cube_contacts: int
    right_cube_contacts: int
    cube_floor_contacts: int
    total_contacts: int

    @property
    def bilateral_contact(self) -> bool:
        """左右两根手指是否都在当前帧接触方块。"""

        return (
            self.left_cube_contacts > 0
            and self.right_cube_contacts > 0
        )

    @property
    def cube_on_floor(self) -> bool:
        """方块当前是否与地面接触。"""

        return self.cube_floor_contacts > 0


@dataclass(frozen=True)
class GraspStatus:
    """汇总当前抓取相关的接触和物理状态。

    ``grasp_candidate`` 用于闭合夹爪后的预抬升检查；此时方块仍可位于
    地面。``grasp_success`` 则用于抬升后的最终验收，还要求方块离地且
    速度已经稳定。
    """

    contact_snapshot: ContactSnapshot

    consecutive_bilateral_steps: int
    required_bilateral_steps: int

    gripper_width: float

    cube_height: float
    lift_height: float

    cube_linear_speed: float
    cube_angular_speed: float

    width_valid: bool
    cube_lifted: bool
    cube_stable: bool

    @property
    def persistent_contact(self) -> bool:
        """双侧接触是否已经连续保持足够步数。"""

        return (
            self.consecutive_bilateral_steps
            >= self.required_bilateral_steps
        )

    @property
    def grasp_candidate(self) -> bool:
        """当前状态是否适合进入抬升阶段。"""

        return self.persistent_contact and self.width_valid

    @property
    def grasp_success(self) -> bool:
        """方块是否已被稳定夹持并成功抬离地面。"""

        return (
            self.grasp_candidate
            and self.cube_lifted
            and self.cube_stable
            and not self.contact_snapshot.cube_on_floor
        )

    @property
    def reason(self) -> str:
        """返回当前尚未成功的首要原因。"""

        if not self.contact_snapshot.bilateral_contact:
            return "no_bilateral_contact"

        if not self.persistent_contact:
            return "contact_not_persistent"

        if not self.width_valid:
            return "gripper_width_invalid"

        if self.contact_snapshot.cube_on_floor:
            return "cube_on_floor"

        if not self.cube_lifted:
            return "cube_not_lifted"

        if not self.cube_stable:
            return "cube_not_stable"

        return "none"


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


def get_geom_id(model, geom_name: str) -> int:
    """根据geom名称取得ID，找不到时明确报错。"""

    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    if geom_id == -1:
        raise ValueError(
            f"模型里不存在名为 {geom_name!r} 的 geom"
        )

    return int(geom_id)


class ContactSensor:
    """把当前 data.contact 分类为左指/右指与目标物体的接触。"""

    def __init__(
        self,
        model,
        left_body="left_finger",
        right_body="right_finger",
        target_body="cube",
        floor_geom="floor",
    ):
        # body 与 geom 的归属属于 MjModel 静态结构，仿真时不会变化。
        # 转成 set 后，每个物理步只需做快速集合成员判断。
        self.left = set(body_geoms(model, left_body))
        self.right = set(body_geoms(model, right_body))
        self.target = set(body_geoms(model, target_body))
        self.floor = {get_geom_id(model, floor_geom)}

    def _count_between(
        self,
        data,
        first_geoms: set[int],
        second_geoms: set[int],
    ) -> int:
        """统计两个geom集合之间的当前接触条目数。

        contact中的geom1和geom2没有固定顺序，因此必须同时检查：

        first -> second
        second -> first
        """

        count = 0

        # data.contact是预分配数组，只有前data.ncon项有效。
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            first_to_second = (
                geom1 in first_geoms
                and geom2 in second_geoms
            )

            second_to_first = (
                geom1 in second_geoms
                and geom2 in first_geoms
            )

            if first_to_second or second_to_first:
                count += 1

        return count

    def snapshot(self, data) -> ContactSnapshot:
        """读取当前物理步，返回分类后的接触快照。"""

        left_cube_contacts = self._count_between(
            data,
            self.left,
            self.target,
        )

        right_cube_contacts = self._count_between(
            data,
            self.right,
            self.target,
        )

        cube_floor_contacts = self._count_between(
            data,
            self.target,
            self.floor,
        )

        return ContactSnapshot(
            left_cube_contacts=left_cube_contacts,
            right_cube_contacts=right_cube_contacts,
            cube_floor_contacts=cube_floor_contacts,
            total_contacts=int(data.ncon),
        )

    def finger_contacts(self, data):
        """返回本物理步的左右手指-方块接触条数。"""

        snapshot = self.snapshot(data)

        return (
            snapshot.left_cube_contacts,
            snapshot.right_cube_contacts,
        )

    def is_grasped(self, data):
        """返回当前是否为双侧接触。

        这里只兼容 Day 5 代码，不代表稳定抓取成功。
        """

        return self.snapshot(data).bilateral_contact


class BilateralContactTracker:
    """统计双侧接触连续保持了多少个物理步。"""

    def __init__(self, required_steps: int):
        if required_steps <= 0:
            raise ValueError(
                "required_steps must be positive"
            )

        self.required_steps = required_steps
        self.consecutive_steps = 0

    @property
    def persistent_contact(self) -> bool:
        """双侧接触是否已经保持到规定步数。"""

        return (
            self.consecutive_steps
            >= self.required_steps
        )

    def update(
        self,
        snapshot: ContactSnapshot,
    ) -> bool:
        """输入新一帧快照，更新连续接触计数。"""

        if snapshot.bilateral_contact:
            self.consecutive_steps += 1
        else:
            self.consecutive_steps = 0

        return self.persistent_contact

    def reset(self) -> None:
        """清除上一回合留下的接触历史。"""

        self.consecutive_steps = 0


class GraspMonitor:
    """从仿真状态生成带时间记忆的 :class:`GraspStatus`。

    监测器把四类信息组合起来：当前接触、连续双侧接触步数、实际夹爪
    宽度，以及方块相对初始落稳位置的高度和速度。它只负责观察，不会
    向机械臂、夹爪或方块写入任何控制量和状态。
    """

    def __init__(
        self,
        scene: PandaScene,
        *,
        reference_cube_height: float | None = None,
        contact_duration: float = DEFAULT_CONTACT_DURATION,
        min_grasp_width: float = DEFAULT_MIN_GRASP_WIDTH,
        max_grasp_width: float = DEFAULT_MAX_GRASP_WIDTH,
        min_lift_height: float = DEFAULT_MIN_LIFT_HEIGHT,
        max_linear_speed: float = DEFAULT_MAX_LINEAR_SPEED,
        max_angular_speed: float = DEFAULT_MAX_ANGULAR_SPEED,
    ) -> None:
        timestep = float(scene.model.opt.timestep)

        positive_parameters = {
            "timestep": timestep,
            "contact_duration": contact_duration,
            "min_lift_height": min_lift_height,
            "max_linear_speed": max_linear_speed,
            "max_angular_speed": max_angular_speed,
        }

        for name, value in positive_parameters.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be positive and finite"
                )

        if (
            not np.isfinite(min_grasp_width)
            or not np.isfinite(max_grasp_width)
            or min_grasp_width < 0.0
            or max_grasp_width <= min_grasp_width
        ):
            raise ValueError(
                "grasp width range must be finite and increasing"
            )

        self.sensor = ContactSensor(scene.model)
        self.tracker = BilateralContactTracker(
            required_steps=math.ceil(
                contact_duration / timestep
            )
        )

        _, self._cube_qvel_address = (
            get_cube_joint_addresses(scene.model)
        )

        self.min_grasp_width = float(min_grasp_width)
        self.max_grasp_width = float(max_grasp_width)
        self.min_lift_height = float(min_lift_height)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)

        self.reference_cube_height = 0.0
        self._last_update_time: float | None = None
        self.reset(
            scene,
            reference_cube_height=reference_cube_height,
        )

    def reset(
        self,
        scene: PandaScene,
        *,
        reference_cube_height: float | None = None,
    ) -> None:
        """清除接触历史，并记录本回合的方块初始高度。"""

        self.tracker.reset()
        self._last_update_time = None

        if reference_cube_height is None:
            reference_cube_height = float(
                scene.data.body("cube").xpos[2]
            )

        if not np.isfinite(reference_cube_height):
            raise ValueError(
                "reference_cube_height must be finite"
            )

        self.reference_cube_height = float(
            reference_cube_height
        )

    def update(
        self,
        scene: PandaScene,
    ) -> GraspStatus:
        """读取当前物理步并更新接触历史，返回完整抓取状态。"""

        snapshot = self.sensor.snapshot(scene.data)
        current_time = float(scene.data.time)

        if not np.isfinite(current_time):
            raise RuntimeError(
                "simulation time is not finite"
            )

        if (
            self._last_update_time is not None
            and current_time < self._last_update_time
        ):
            raise RuntimeError(
                "simulation time moved backwards; "
                "call monitor.reset() after resetting the scene"
            )

        # 第一次读取计为一帧；同一仿真时刻的重复读取只刷新测量值，
        # 不重复累计接触历史。正常用法仍是在每个 mj_step 后调用一次。
        if (
            self._last_update_time is None
            or current_time > self._last_update_time
        ):
            self.tracker.update(snapshot)
            self._last_update_time = current_time

        gripper_width = float(
            get_gripper_width(scene)
        )
        cube_height = float(
            scene.data.body("cube").xpos[2]
        )
        lift_height = (
            cube_height - self.reference_cube_height
        )

        cube_velocity = scene.data.qvel[
            self._cube_qvel_address:
            self._cube_qvel_address + 6
        ]
        cube_linear_speed = float(
            np.linalg.norm(cube_velocity[:3])
        )
        cube_angular_speed = float(
            np.linalg.norm(cube_velocity[3:])
        )

        measurements = np.array(
            [
                gripper_width,
                cube_height,
                lift_height,
                cube_linear_speed,
                cube_angular_speed,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(measurements)):
            raise RuntimeError(
                "grasp measurements contain NaN or Inf"
            )

        width_valid = (
            self.min_grasp_width
            <= gripper_width
            <= self.max_grasp_width
        )
        cube_lifted = (
            lift_height >= self.min_lift_height
        )
        cube_stable = (
            cube_linear_speed <= self.max_linear_speed
            and cube_angular_speed <= self.max_angular_speed
        )

        return GraspStatus(
            contact_snapshot=snapshot,
            consecutive_bilateral_steps=(
                self.tracker.consecutive_steps
            ),
            required_bilateral_steps=(
                self.tracker.required_steps
            ),
            gripper_width=gripper_width,
            cube_height=cube_height,
            lift_height=lift_height,
            cube_linear_speed=cube_linear_speed,
            cube_angular_speed=cube_angular_speed,
            width_valid=width_valid,
            cube_lifted=cube_lifted,
            cube_stable=cube_stable,
        )
