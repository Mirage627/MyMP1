"""Record a synthetic MP4 as a SwanLab GIF, without torch or a simulator."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("online", "offline"), default="offline")
    parser.add_argument("--output-dir", default="data/outputs/swanlab_video_check")
    args = parser.parse_args()
    import av
    from PIL import Image, ImageDraw
    import swanlab
    source_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("mp1_media", source_root/"common"/"swanlab_logging.py")
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mp4 = output/"synthetic_rollout.mp4"
    with av.open(str(mp4), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for index in range(20):
            image = Image.new("RGB", (96, 64), "#202020")
            ImageDraw.Draw(image).rectangle((index*4, 20, index*4+12, 35), fill="orange")
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    run = swanlab.init(project="MP1_ablation", experiment_name="video_pipeline_check",
                       mode=args.mode, logdir=str(output/"swanlog"))
    try:
        payload, records = media.prepare_swanlab_metrics(
            {"check/score": 1., "check/rollout_video": {"type": "video", "path": str(mp4), "fps": 10.}},
            output, swanlab_module=swanlab)
        with Image.open(records["check/rollout_video"]["preview"]) as preview:
            total_ms = 0
            for index in range(preview.n_frames):
                preview.seek(index)
                total_ms += preview.info["duration"]
            if abs(total_ms-2000) > 20:
                raise AssertionError(f"GIF playback duration changed: {total_ms}ms")
        run.log(payload, step=0)
        (output/"media_check.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("SWANLAB_VIDEO_CHECK_OK", flush=True)
        print(json.dumps({"mode": args.mode, "output": str(output), "media": records}, indent=2), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
