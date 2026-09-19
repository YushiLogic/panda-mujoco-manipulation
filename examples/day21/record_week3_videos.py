"""Day 21：离屏录制第三周成功与受控失败演示。"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from panda_mujoco.panda_pick_task import PandaPickTask
from panda_mujoco.pick_controller import (
    PickResult,
    PickState,
    run_scripted_pick,
)
from panda_mujoco.simulation import PandaScene


VIDEO_DIRECTORY = Path("results/day21/videos")
SUCCESS_VIDEO = VIDEO_DIRECTORY / "week3_success_seed00.mp4"
FAILURE_VIDEO = (
    VIDEO_DIRECTORY
    / "week3_failure_contact_timeout.mp4"
)

FRAME_RATE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INITIAL_HOLD_SECONDS = 1.0
FINAL_HOLD_SECONDS = 2.0


class EpisodeVideoRecorder:
    """按固定帧率从MuJoCo物理步流中抽取并编码画面。"""

    def __init__(
        self,
        scene: PandaScene,
        output_path: Path,
        *,
        title: str,
        seed: int,
    ) -> None:
        self.scene = scene
        self.output_path = output_path
        self.title = title
        self.seed = seed

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.renderer = mujoco.Renderer(
            scene.model,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [0.40, 0.0, 0.30]
        self.camera.distance = 1.50
        self.camera.azimuth = 135.0
        self.camera.elevation = -25.0

        self.writer = imageio.get_writer(
            output_path,
            format="FFMPEG",
            mode="I",
            fps=FRAME_RATE,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            ffmpeg_log_level="error",
        )

        timestep = float(scene.model.opt.timestep)
        self.physics_steps_per_frame = max(
            1,
            round(1.0 / (FRAME_RATE * timestep)),
        )
        self.physics_step_count = 0
        self.frame_count = 0

    def _render_annotated_frame(self) -> np.ndarray:
        """渲染当前场景，并添加不会影响仿真的说明文字。"""

        self.renderer.update_scene(
            self.scene.data,
            camera=self.camera,
        )
        frame = self.renderer.render()

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (0, 0, FRAME_WIDTH, 54),
            fill=(0, 0, 0),
        )
        draw.text(
            (12, 8),
            self.title,
            fill=(255, 255, 255),
        )
        draw.text(
            (12, 30),
            (
                f"seed={self.seed}  "
                f"simulation time={self.scene.data.time:.3f} s"
            ),
            fill=(220, 220, 220),
        )
        return np.asarray(image)

    def append_current_frame(self) -> None:
        """立即写入一帧。"""

        self.writer.append_data(
            self._render_annotated_frame()
        )
        self.frame_count += 1

    def append_still(self, seconds: float) -> None:
        """将当前状态保持若干秒，便于观察开始或最终结果。"""

        frame = self._render_annotated_frame()
        repeat_count = round(seconds * FRAME_RATE)
        for _ in range(repeat_count):
            self.writer.append_data(frame)
            self.frame_count += 1

    def physics_step_callback(
        self,
        callback_scene: PandaScene,
    ) -> None:
        """由状态机在每个mj_step之后调用。"""

        if callback_scene is not self.scene:
            raise RuntimeError(
                "video callback received an unexpected scene"
            )

        self.physics_step_count += 1
        if (
            self.physics_step_count
            % self.physics_steps_per_frame
            == 0
        ):
            self.append_current_frame()

    def close(self) -> None:
        """关闭编码器和MuJoCo渲染器。"""

        self.writer.close()
        self.renderer.close()


def print_result(
    label: str,
    path: Path,
    result: PickResult,
    frame_count: int,
) -> None:
    """打印录像对应的状态机结果。"""

    print(f"\n{label}")
    print("video:", path.resolve())
    print("frames:", frame_count)
    print("success:", result.success)
    print("final state:", result.final_state.value)
    print(
        "failed stage:",
        None
        if result.failed_stage is None
        else result.failed_stage.value,
    )
    print("reason:", result.reason)
    print(
        "state trace:",
        " -> ".join(
            record.state.value
            for record in result.records
        ),
    )


def record_episode(
    *,
    output_path: Path,
    title: str,
    seed: int,
    close_timeout: float,
    expected_success: bool,
    expected_failed_stage: PickState | None,
) -> PickResult:
    """复位一个episode，录制状态机执行，并检查预期结局。"""

    task = PandaPickTask()
    task.reset(seed=seed)

    recorder = EpisodeVideoRecorder(
        task.scene,
        output_path,
        title=title,
        seed=seed,
    )

    try:
        recorder.append_still(INITIAL_HOLD_SECONDS)
        result = run_scripted_pick(
            task.scene,
            seed=seed,
            reset_scene=False,
            close_timeout=close_timeout,
            step_callback=(
                recorder.physics_step_callback
            ),
        )
        recorder.append_still(FINAL_HOLD_SECONDS)
    finally:
        recorder.close()

    if result.success is not expected_success:
        raise RuntimeError(
            "recorded episode did not reach the expected outcome"
        )
    if result.failed_stage is not expected_failed_stage:
        raise RuntimeError(
            "recorded episode failed at an unexpected stage"
        )

    print_result(
        title,
        output_path,
        result,
        recorder.frame_count,
    )
    return result


def main() -> None:
    """生成两段第三周交付视频。"""

    success_result = record_episode(
        output_path=SUCCESS_VIDEO,
        title="Week 3 scripted grasp: success",
        seed=0,
        close_timeout=1.0,
        expected_success=True,
        expected_failed_stage=None,
    )

    failure_result = record_episode(
        output_path=FAILURE_VIDEO,
        title=(
            "Injected failure: close timeout = 0.002 s"
        ),
        seed=0,
        close_timeout=0.002,
        expected_success=False,
        expected_failed_stage=PickState.VERIFY_CONTACT,
    )

    if success_result.final_state is not PickState.DONE:
        raise RuntimeError("success video did not reach DONE")
    if failure_result.reason != "no_bilateral_contact":
        raise RuntimeError(
            "failure video did not report contact failure"
        )

    print("\nWeek 3 demonstration videos: PASSED")


if __name__ == "__main__":
    main()
