"""Real-data gaze evaluation; annotated heads and automatic tracking compared."""

import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from .gazelle import SceneGazeAnalyzer
from .gazelle_adapters import GazelleEstimator, PoseHeadDetector
from .provenance import environment_manifest
from .retail_gaze import box_iou, load_sessions, match_track, prediction_record, summarize
from .tracking import YoloByteTracker


class OriginalFrames:
    """Read a complete every-fourth-frame sequence without lossy video encoding."""

    def __init__(self, directory):
        self.paths = sorted(directory.glob("*.jpg"), key=lambda p: int(p.stem))
        if not self.paths or [int(p.stem) for p in self.paths] != list(range(0, len(self.paths)*4, 4)):
            raise ValueError(f"Original JPEG sequence must start at 0 and increase by 4: {directory}")
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        import cv2
        if self.index == len(self.paths):
            return False, None
        path = self.paths[self.index]
        self.index += 1
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable original frame: {path}")
        return True, image

    def release(self):
        pass


def load_mask(path, size, actual):
    import cv2
    if path is None:
        return None, "unannotated"
    if not path.is_file():
        return None, "missing"
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.shape[:2] != (size[1], size[0]):
        raise ValueError(f"Invalid mask dimensions: {path}")
    if raw.ndim == 2:
        mask = raw > 0
    else:
        mask = raw[:, :, :3].max(axis=2) > 0
        if raw.shape[2] == 4:
            mask &= raw[:, :, 3] > 0
    x, y = actual
    if not (0 <= x < size[0] and 0 <= y < size[1] and mask[int(y), int(x)]):
        return mask, "label_outside_mask"
    return mask, "valid"


def run(args):
    import cv2
    import torch
    from .gazelle_adapters import GAZELLE_REVISION

    sessions = load_sessions(args.dataset, args.sessions, args.split)
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; choose a new path")
    for path in [args.weights, args.head_weights, args.gaze_weights,
                 *(s["video"] for s in sessions.values())]:
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
    if args.threads < 1:
        raise ValueError("Threads must be positive")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True)
    previews = args.output_dir / "previews"
    previews.mkdir()
    started = perf_counter()
    estimator = GazelleEstimator(str(args.gaze_weights), args.device)
    analyzer = SceneGazeAnalyzer(PoseHeadDetector(str(args.head_weights), args.device), estimator)
    model_load_s = perf_counter()-started
    records, clips, mask_paths = [], {}, set()
    with (args.output_dir / "predictions.jsonl").open("w") as export:
        for name, session in sessions.items():
            print(f"Evaluating {name}: {len(session['labels'])} {args.split} labels", flush=True)
            tracker = YoloByteTracker(str(args.weights), device=args.device)
            capture = (OriginalFrames(args.original_frames / name.replace("_", "/", 1))
                       if args.original_frames else cv2.VideoCapture(str(session["video"])))
            if not capture.isOpened():
                raise ValueError(f"Cannot open {session['video']}")
            fps = None if args.original_frames else capture.get(cv2.CAP_PROP_FPS)
            index, scored = 0, set()
            timings = {"tracking_s": 0., "annotated_head_gaze_s": 0., "automatic_head_gaze_s": 0.}
            track_ids = set()
            try:
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    height, width = image.shape[:2]
                    t = perf_counter()
                    tracks = tracker.track(image)
                    timings["tracking_s"] += perf_counter()-t
                    track_ids.update(t.track_id for t in tracks)
                    label = session["labels"].get(index)
                    if label is not None:
                        scored.add(index)
                        head = tuple(v*(width if i%2 == 0 else height) for i, v in enumerate(label["head"]))
                        actual = (label["point"][0]*width, label["point"][1]*height)
                        mask, mask_status = load_mask(label["mask"], (width, height), actual)
                        if label["mask"] and label["mask"].is_file():
                            mask_paths.add(label["mask"])
                        t = perf_counter()
                        point, confidence, reason = estimator.predict(image, [head])[0]
                        timings["annotated_head_gaze_s"] += perf_counter()-t
                        oracle = prediction_record(point, reason, actual, (width, height), mask, mask_status)
                        oracle.update(head_bbox=head, in_frame_score=confidence)
                        t = perf_counter()
                        observations = analyzer.observe(image, tracks)
                        timings["automatic_head_gaze_s"] += perf_counter()-t
                        track = match_track(tracks, head)
                        observation = next((o for o in observations if track and o.track_id == track.track_id), None)
                        auto = prediction_record(observation.target_image if observation else None,
                                                 observation.reason if observation else "no_unique_person_match",
                                                 actual, (width, height), mask, mask_status)
                        auto.update(track_id=track.track_id if track else None,
                                    head_bbox=observation.head_bbox if observation else None,
                                    head_iou=box_iou(head, observation.head_bbox) if observation and observation.head_bbox else None,
                                    in_frame_score=observation.in_frame_score if observation else None)
                        row = {"session": name, "frame_index": index, "split": label["split"],
                               "image_size": [width, height], "actual_point": actual,
                               "annotated_head": oracle, "automatic": auto}
                        records.append(row)
                        export.write(json.dumps(row, allow_nan=False)+"\n")
                        export.flush()
                        draw_preview(cv2, image, head, actual, mask, row, previews / f"{name}_{index:04d}.jpg")
                    index += 1
            finally:
                capture.release()
            if scored != set(session["labels"]):
                raise ValueError(f"Video ended before all annotations in {name} were decoded")
            clip_rows = [r for r in records if r["session"] == name]
            clips[name] = {"decoded_frames": index, "encoded_fps": fps,
                           "track_ids_seen": sorted(track_ids), "timings": timings,
                           **{mode: summarize([r[mode] for r in clip_rows]) for mode in ("annotated_head", "automatic")}}
            print(json.dumps({"session": name, "metrics": clips[name]}), flush=True)
    report = {"schema_version": 1, "dataset": "Voxel51/retail_gaze", "synthetic": False,
              "split": args.split, "sessions": args.sessions, "model": estimator.model_name,
              "image_source": "original_kaggle_jpegs" if args.original_frames else "huggingface_videos",
              "gazelle_revision": GAZELLE_REVISION, "device": args.device, "torch_threads": args.threads,
              "model_load_s": model_load_s, "elapsed_s": perf_counter()-started,
              "metrics": {mode: summarize([r[mode] for r in records]) for mode in ("annotated_head", "automatic")},
              "mask_statuses": dict(Counter(r["annotated_head"]["mask_status"] for r in records)),
              "by_session": clips,
              "limitations": ["Small diagnostic sample, not a full benchmark or participant-disjoint evaluation.",
                              "No training or threshold tuning on these clips.",
                              "10 FPS is reconstructed playback timing; no dwell-time accuracy is claimed.",
                              "Region hit means predicted pixel inside the annotated target mask, not product classification.",
                              "Masks with ground truth outside their foreground are excluded from region scores.",
                              "Automatic gaze runs on labeled frames; tracking processes every decoded frame.",
                              "Person matching uses labeled head center; no identity ground truth or tracking accuracy is claimed."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    inputs = [args.dataset / "samples.json", args.dataset / "frames.json", args.weights,
              args.head_weights, args.gaze_weights, *(s["video"] for s in sessions.values()), *sorted(mask_paths)]
    if args.original_frames:
        inputs.extend(p for name in sessions for p in sorted(
            (args.original_frames / name.replace("_", "/", 1)).glob("*.jpg")))
    manifest = environment_manifest(inputs)
    for entry, path in zip(manifest["inputs"], inputs, strict=True):
        entry["path"] = str(path)
    manifest["evaluation_sources"] = environment_manifest([Path(__file__), Path(__file__).with_name("retail_gaze.py")])["inputs"]
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return report


def draw_preview(cv2, image, head, actual, mask, row, path):
    canvas = image.copy()
    if mask is not None:
        tint = canvas.copy()
        tint[mask] = (0, 180, 0)
        canvas = cv2.addWeighted(canvas, .8, tint, .2, 0)
    integer = lambda p: tuple(round(v) for v in p)
    cv2.rectangle(canvas, integer(head[:2]), integer(head[2:]), (0, 255, 0), 2)
    cv2.circle(canvas, integer(actual), 6, (0, 255, 0), -1)
    for mode, color in [("annotated_head", (255, 150, 0)), ("automatic", (0, 0, 255))]:
        prediction = row[mode]
        if prediction["point"]:
            cv2.drawMarker(canvas, integer(prediction["point"]), color, cv2.MARKER_CROSS, 16, 2)
        if prediction.get("head_bbox"):
            box = prediction["head_bbox"]
            cv2.rectangle(canvas, integer(box[:2]), integer(box[2:]), color, 1)
    lines = [f"{row['session']} frame {row['frame_index']} | green: label",
             "blue: annotated-head gaze | red: automatic gaze",
             "automatic: " + (row["automatic"]["reason"] or "estimated")]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (8, 20+i*22), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3)
        cv2.putText(canvas, line, (8, 20+i*22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write preview {path}")
