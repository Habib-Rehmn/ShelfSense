"""Local video tracking and a dependency-free contract demo."""

import argparse
import json
import os
from itertools import islice
from math import isfinite
from pathlib import Path

from .pipeline import Frame, process
from .tracking import Track, YoloByteTracker
from .video import VideoReader
from .gaze import GazeAnalyzer
from .gazelle import SceneGazeAnalyzer
from .gazelle_adapters import GazelleEstimator, PoseHeadDetector
from .gaze_adapters import L2CSEstimator, OpenCVFaceDetector
from .calibration import Calibration
from .replay import map_gaze_file
from .attention import AttentionConfig, AttentionEngine
from .attention_replay import derive_attention


class DemoTracker:
    def track(self, frame):
        return [Track(1, (10.0 + frame, 20.0, 60.0 + frame, 140.0), 1.0)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="ShelfSense tracking pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    retail = commands.add_parser("evaluate-retail", help="Compare annotated and automatic heads on Retail Gaze")
    retail.add_argument("dataset", type=Path)
    retail.add_argument("--sessions", nargs="+", required=True)
    retail.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    retail.add_argument("--weights", type=Path, required=True)
    retail.add_argument("--head-weights", type=Path, required=True)
    retail.add_argument("--gaze-weights", type=Path, required=True)
    retail.add_argument("--device", default="cpu")
    retail.add_argument("--threads", type=int, default=2)
    retail.add_argument("--output-dir", type=Path, required=True)
    retail.add_argument("--original-frames", type=Path, help="Original Kaggle JPEG root, organized by subject/session")
    evaluate_parser = commands.add_parser("evaluate", help="Evaluate labeled held-out shelf observations")
    evaluate_parser.add_argument("source", type=Path)
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--calibration", type=Path, required=True)
    evaluate_parser.add_argument("--kind", choices=["raw", "smoothed"], default="raw")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    manifest = commands.add_parser("manifest", help="Fingerprint experiment inputs and installed package versions")
    manifest.add_argument("inputs", nargs="*", type=Path)
    manifest.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("smoke", help="Run the synthetic pipeline through database and evaluation")
    smoke.add_argument("--output-dir", type=Path, required=True)
    initialize = commands.add_parser("init-db", help="Create analytics tables")
    ingest = commands.add_parser("ingest", help="Validate and atomically import a completed attention export")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--calibration", type=Path, required=True)
    report = commands.add_parser("report", help="Print analytics for one processing run")
    report.add_argument("--run-id", required=True)
    report.add_argument("--session-id")
    for command in (initialize, ingest, report):
        command.add_argument("--database-url", default=os.environ.get(
            "SHELFSENSE_DATABASE_URL", "sqlite:///outputs/shelfsense.db"))
    demo = commands.add_parser("demo", help="Synthetic contract demo, no inference")
    demo.add_argument("--output", type=Path, required=True)
    calibrate = commands.add_parser("calibrate", help="Fit and validate labeled shelf calibration")
    calibrate.add_argument("source", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("map-gaze", help="Map stored schema-2 gaze JSONL to shelf zones")
    replay.add_argument("source", type=Path)
    replay.add_argument("--calibration", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    attention_parser = commands.add_parser("attention", help="Derive attention events from schema-3 JSONL")
    attention_parser.add_argument("source", type=Path)
    attention_parser.add_argument("--calibration", type=Path, required=True)
    attention_parser.add_argument("--output", type=Path, required=True)
    track = commands.add_parser("track", help="Track people in a local CFR video")
    track.add_argument("source", type=Path)
    track.add_argument("--weights", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    track.add_argument("--device", default="cpu")
    track.add_argument("--confidence", type=float, default=0.1)
    track.add_argument("--fps", type=float)
    track.add_argument("--max-frames", type=int)
    track.add_argument("--gaze-weights", type=Path, help="Enable gaze using local decoder weights")
    track.add_argument("--gaze-backend", choices=["gazelle", "l2cs"], default="gazelle")
    track.add_argument("--head-weights", type=Path, help="Local yolo26n-pose.pt for head localization")
    track.add_argument("--min-head-size", type=int, default=16)
    track.add_argument("--in-frame-threshold", type=float, default=.5)
    track.add_argument("--gaze-arch", default="ResNet50",
                       choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"])
    track.add_argument("--min-face-size", type=int, default=32)
    track.add_argument("--calibration", type=Path, help="Labeled shelf calibration JSON")
    track.add_argument("--attention", action="store_true", help="Enable smoothing and attention events")
    for command in (track, attention_parser):
        for name, field in AttentionConfig.__dataclass_fields__.items():
            command.add_argument("--" + name.replace("_", "-"), type=float, default=field.default)
    args = parser.parse_args(argv)
    if args.command == "evaluate-retail":
        from .retail_evaluation import run
        try:
            result = run(args)
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
            parser.exit(1, f"Retail evaluation failed: {exc}\n")
        print(json.dumps(result["metrics"], indent=2))
        return 0
    if args.command == "smoke":
        from .smoke import run_smoke
        try:
            result = run_smoke(args.output_dir)
        except (ValueError, OSError, RuntimeError) as exc:
            parser.exit(1, f"Smoke test failed: {exc}\n")
        print(f"Synthetic smoke passed: {result['frames']} frames; report at {args.output_dir / 'report.json'}")
        return 0
    if args.command in {"init-db", "ingest", "report"}:
        from .storage import Store
        from .ingestion import read_export
        try:
            store = Store(args.database_url)
            if args.command == "init-db":
                store.initialize()
                result = {"initialized": True}
            elif args.command == "ingest":
                with args.source.open() as source:
                    data = read_export(source, Calibration.load(args.calibration))
                result = store.ingest(data)
            else:
                result = store.report(args.run_id, args.session_id)
            print(json.dumps(result, allow_nan=False))
            return 0
        except (ValueError, KeyError, ImportError) as exc:
            parser.exit(1, f"Error: {exc}\n")
        except Exception:
            parser.exit(1, "Storage or input unavailable. Check the input path, database connection and init-db setup.\n")
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
        if args.gaze_weights is not None and args.gaze_backend == "gazelle":
            if args.head_weights is None or not args.head_weights.is_file():
                parser.error("Gaze-LLE requires --head-weights pointing to local yolo26n-pose.pt")
        if args.min_head_size <= 0:
            parser.error("--min-head-size must be positive")
        if not isfinite(args.in_frame_threshold) or not 0 <= args.in_frame_threshold <= 1:
            parser.error("--in-frame-threshold must be in [0, 1]")
        if args.min_face_size <= 0:
            parser.error("--min-face-size must be positive")
        if args.calibration is not None and args.gaze_weights is None:
            parser.error("--calibration requires --gaze-weights")
        if args.attention and args.calibration is None:
            parser.error("--attention requires --calibration")
    try:
        attention_config = None
        if args.command in {"track", "attention"}:
            attention_config = AttentionConfig(**{
                name: getattr(args, name) for name in AttentionConfig.__dataclass_fields__})
        if args.command in {"evaluate", "manifest"}:
            if args.command == "evaluate":
                from .evaluation import evaluate
                with args.source.open() as source:
                    result = evaluate(source, json.loads(args.labels.read_text()),
                                      Calibration.load(args.calibration), args.kind)
            else:
                from .provenance import environment_manifest
                result = environment_manifest(args.inputs)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                json.dump(result, output, indent=2, allow_nan=False)
            print(f"Wrote {args.command} report to {args.output}")
            return 0
        elif args.command == "calibrate":
            calibration = Calibration.load(args.source)
            calibration.save(args.output)
            print(json.dumps(calibration.summary()))
            return 0
        elif args.command in {"map-gaze", "attention"}:
            calibration = Calibration.load(args.calibration)
            with args.source.open() as source:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as output:
                    if args.command == "attention":
                        count = derive_attention(source, output, AttentionEngine(calibration, attention_config))
                    else:
                        count = map_gaze_file(source, output, calibration)
        elif args.command == "demo":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                count = process((Frame(i, i / 30, i) for i in range(30)),
                                DemoTracker(), output, session_id="synthetic-demo")
        else:
            calibration = Calibration.load(args.calibration) if args.calibration is not None else None
            if calibration is not None and calibration.image_targets != (args.gaze_backend == "gazelle"):
                raise ValueError("Gaze-LLE requires schema-2 shelf geometry; L2CS requires schema-1 labeled calibration")
            attention = AttentionEngine(calibration, attention_config) if args.attention else None
            tracker = YoloByteTracker(str(args.weights), args.confidence, args.device)
            gaze = None
            if args.gaze_weights is not None:
                if args.gaze_backend == "gazelle":
                    gaze = SceneGazeAnalyzer(PoseHeadDetector(str(args.head_weights), args.device),
                        GazelleEstimator(str(args.gaze_weights), args.device, args.in_frame_threshold),
                        args.min_head_size)
                else:
                    gaze = GazeAnalyzer(OpenCVFaceDetector(), L2CSEstimator(
                        str(args.gaze_weights), args.device, args.gaze_arch), args.min_face_size)
            with VideoReader(str(args.source), args.fps) as frames:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as output:
                    count = process(islice(frames, args.max_frames), tracker, output, gaze=gaze,
                                    calibration=calibration, attention=attention)
                    if count == 0:
                        raise ValueError("Video contains no decodable frames")
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\nFor dependency setup, see README.md and docs/gaze.md.\n")
    print(f"Wrote {count} frames to {args.output}")
    return 0
