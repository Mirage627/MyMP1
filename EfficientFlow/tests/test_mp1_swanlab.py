"""SwanLab payloads, GIF conversion, disabled mode, and run identity."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "mp1_swanlab_logging", Path(__file__).resolve().parents[1]/"common"/"swanlab_logging.py")
logging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(logging)


class FakeSDK:
    def __init__(self):
        self.initializations = []
        self.logs = []
        self.finishes = 0
        self.videos = []

    def init(self, **kwargs):
        self.initializations.append(kwargs)
        return types.SimpleNamespace(id=kwargs.get("id", "test-run-42"), log=self.log, finish=self.finish)

    def log(self, payload, step):
        self.logs.append((payload, step))

    def finish(self):
        self.finishes += 1

    def Video(self, path, caption):
        self.assert_gif(path)
        value = types.SimpleNamespace(path=path, caption=caption)
        self.videos.append(value)
        return value

    @staticmethod
    def assert_gif(path):
        with Image.open(path) as image:
            assert image.format == "GIF"


class SwanLabTests(unittest.TestCase):
    def gif(self, directory):
        path = Path(directory)/"original.gif"
        images = [Image.new("RGB", (20, 12), color) for color in ("red", "blue")]
        images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
        return path

    def test_scalars_and_video_both_reach_sdk_with_json_safe_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = FakeSDK()
            path = self.gif(directory)
            cfg = OmegaConf.create({"logging": {"mode": "offline", "project": "MP1", "name": "full"}})
            logger = logging.SwanLabLogger(cfg, directory, swanlab_module=sdk)
            logger.log({"train_loss": .5, "profile": "mp1"}, step=13, rollout={
                "test/mean_score": .8, "test/sim_video_1": {"type": "video", "path": str(path), "fps": 10}})
            self.assertEqual(sdk.initializations[0]["mode"], "offline")
            self.assertEqual(sdk.initializations[0]["experiment_name"], "full")
            self.assertEqual(sdk.initializations[0]["logdir"], str(Path(directory)/"swanlog"))
            payload, step = sdk.logs[0]
            self.assertEqual(step, 13)
            self.assertEqual(payload["train_loss"], .5)
            self.assertEqual(payload["test/mean_score"], .8)
            self.assertIs(payload["test/sim_video_1"], sdk.videos[0])
            manifest = json.loads((Path(directory)/"rollout_media.jsonl").read_text())
            self.assertEqual(manifest["videos"]["test/sim_video_1"]["path"], str(path.resolve()))
            logger.finish()
            logger.finish()
            self.assertEqual(sdk.finishes, 1)

    def test_disabled_needs_neither_sdk_nor_conversion_and_keeps_media_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(logging.importlib, "import_module", side_effect=AssertionError("SDK import")):
                logger = logging.SwanLabLogger(OmegaConf.create({}), directory)
                logger.log({"loss": .2}, step=0, rollout={"video": {
                    "type": "video", "path": "preserved.mp4", "fps": 10}})
                logger.finish()
            manifest = json.loads((Path(directory)/"rollout_media.jsonl").read_text())
            self.assertEqual(manifest["videos"]["video"]["path"], "preserved.mp4")

    def test_missing_recordings_fail_even_when_all_rollout_scores_are_zero(self):
        with self.assertRaisesRegex(RuntimeError, "Expected 6 rollout videos, got 0"):
            logging.validate_rollout_videos({"test/mean_score": 0.}, expected_count=6)
        with tempfile.TemporaryDirectory() as directory:
            values = {"test/mean_score": 0., "test/sim_video_1": {
                "type": "video", "path": str(self.gif(directory))}}
            self.assertEqual(logging.validate_rollout_videos(values, expected_count=1), 1)
            Path(values["test/sim_video_1"]["path"]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing or empty rollout video"):
                logging.validate_rollout_videos(values, expected_count=1)

    def test_resume_reuses_public_id_and_keeps_increasing_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = FakeSDK()
            cfg = OmegaConf.create({"logging": {"mode": "offline", "project": "MP1"}})
            first = logging.SwanLabLogger(cfg, directory, swanlab_module=sdk)
            first.log({"loss": 1.}, step=4)
            first.finish()
            resumed = logging.SwanLabLogger(cfg, directory, resume=True, swanlab_module=sdk)
            resumed.log({"loss": .5}, step=8)
            resumed.finish()
            self.assertEqual(sdk.initializations[1]["id"], "test-run-42")
            self.assertEqual(sdk.initializations[1]["resume"], "allow")
            self.assertEqual([step for _, step in sdk.logs], [4, 8])

    def test_media_failures_are_explicit_and_do_not_discard_scalar_results(self):
        with tempfile.TemporaryDirectory() as directory:
            values = {"score": .7, "video": {"type": "video", "path": str(Path(directory)/"missing.mp4")}}
            with self.assertRaisesRegex(RuntimeError, "Cannot prepare SwanLab"):
                logging.prepare_swanlab_metrics(values, directory, swanlab_module=FakeSDK())
            with self.assertWarns(RuntimeWarning):
                payload, records = logging.prepare_swanlab_metrics(
                    values, directory, swanlab_module=FakeSDK(), strict_media=False)
            self.assertEqual(payload["score"], .7)
            self.assertIn("FileNotFoundError", records["video"]["error"])

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV needed for actual MP4 codec check")
    def test_actual_mp4_becomes_gif_preserving_original_and_clip_duration(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory)/"rollout.mp4"
            with av.open(str(original), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=12)
                stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
                for index in range(6):
                    frame = av.VideoFrame.from_image(Image.new("RGB", (32, 24), (index*35, 20, 80)))
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            before = original.read_bytes()
            gif = logging.video_to_gif(original, Path(directory)/"preview.gif", max_frames=3, max_side=16)
            self.assertEqual(original.read_bytes(), before)
            with Image.open(gif) as preview:
                self.assertEqual(preview.format, "GIF")
                self.assertLessEqual(preview.n_frames, 3)
                self.assertLessEqual(max(preview.size), 16)
                duration = 0
                for index in range(preview.n_frames):
                    preview.seek(index)
                    duration += preview.info["duration"]
                self.assertAlmostEqual(duration, 500, delta=20)


if __name__ == "__main__":
    unittest.main()
