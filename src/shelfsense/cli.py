"""Local video tracking and a dependency-free contract demo."""

import argparse
import json
from itertools import islice
from math import isfinite
from pathlib import Path

from .pipeline import Frame, process
from .tracking import Track, YoloByteTracker
from .video import VideoReader
from .gaze import GazeAnalyzer
from .gaze_adapters import L2CSEstimator, OpenCVFaceDetector
from .calibration import Calibration
from .replay import map_gaze_file


class DemoTracker:
    def track(self, frame):
        return [Track(1, (10.0 + frame, 20.0, 60.0 + frame, 140.0), 1.0)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="ShelfSense tracking pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Synthetic contract demo, no inference")
    demo.add_argument("--output", type=Path, required=True)
    calibrate = commands.add_parser("calibrate", help="Fit and validate labeled shelf calibration")
    calibrate.add_argument("source", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("map-gaze", help="Map stored schema-2 gaze JSONL to shelf zones")
    replay.add_argument("source", type=Path)
    replay.add_argument("--calibration", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    track = commands.add_parser("track", help="Track people in a local CFR video")
    track.add_argument("source", type=Path)
    track.add_argument("--weights", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    track.add_argument("--device", default="cpu")
    track.add_argument("--confidence", type=float, default=0.1)
    track.add_argument("--fps", type=float)
    track.add_argument("--max-frames", type=int)
    track.add_argument("--gaze-weights", type=Path, help="Enable L2CS gaze with local weights")
    track.add_argument("--gaze-arch", default="ResNet50",
                       choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"])
    track.add_argument("--min-face-size", type=int, default=32)
    track.add_argument("--calibration", type=Path, help="Labeled shelf calibration JSON")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if args.command == "track":
        if not args.source.is_file() or not args.weights.is_file():
            parser.error("Source video and model weights must be existing local files")
        if not 0 < args.confidence <= 1:
            parser.error("--confidence must be in (0, 1]")
        if args.fps is not None and (not isfinite(args.fps) or args.fps <= 0):
            parser.error("--fps must be finite and positive")
        if args.max_frames is not None and args.max_frames <= 0:
            parser.error("--max-frames must be positive")
        if args.gaze_weights is not None and not args.gaze_weights.is_file():
            parser.error("Gaze weights must be an existing local file")
        if args.min_face_size <= 0:
            parser.error("--min-face-size must be positive")
        if args.calibration is not None and args.gaze_weights is None:
            parser.error("--calibration requires --gaze-weights")
    try:
        if args.command == "calibrate":
            calibration = Calibration.load(args.source)
            calibration.save(args.output)
            print(json.dumps(calibration.summary()))
            return 0
        elif args.command == "map-gaze":
            calibration = Calibration.load(args.calibration)
            with args.source.open() as source:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as output:
                    count = map_gaze_file(source, output, calibration)
        elif args.command == "demo":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                count = process((Frame(i, i / 30, i) for i in range(30)),
                                DemoTracker(), output, session_id="synthetic-demo")
        else:
            calibration = Calibration.load(args.calibration) if args.calibration is not None else None
            tracker = YoloByteTracker(str(args.weights), args.confidence, args.device)
            gaze = None
            if args.gaze_weights is not None:
                gaze = GazeAnalyzer(OpenCVFaceDetector(), L2CSEstimator(
                    str(args.gaze_weights), args.device, args.gaze_arch), args.min_face_size)
            with VideoReader(str(args.source), args.fps) as frames:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as output:
                    count = process(islice(frames, args.max_frames), tracker, output, gaze=gaze,
                                    calibration=calibration)
                    if count == 0:
                        raise ValueError("Video contains no decodable frames")
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\nFor dependency setup, see README.md and docs/gaze.md.\n")
    print(f"Wrote {count} frames to {args.output}")
    return 0
