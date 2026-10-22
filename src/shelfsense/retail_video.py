"""Render Retail Gaze labels and saved predictions as annotated videos."""

import argparse
import json
from math import isfinite
from pathlib import Path

from .retail_evaluation import OriginalFrames, load_mask
from .retail_gaze import load_sessions


def draw_target(cv2, image, box, point, color):
    integer = lambda p: tuple(round(v) for v in p)
    if box is not None:
        cv2.rectangle(image, integer(box[:2]), integer(box[2:]), color, 2)
    if point is not None:
        cv2.circle(image, integer(point), 6, color, -1, cv2.LINE_AA)
        if box is not None:
            # A visual guide, not a measured eye position or 3-D gaze ray.
            origin = ((box[0]+box[2])/2, box[1]+.35*(box[3]-box[1]))
            cv2.arrowedLine(image, integer(origin), integer(point), color, 2,
                           cv2.LINE_AA, tipLength=.035)


def panel(cv2, image, title, subtitle, color):
    import numpy as np
    header = np.zeros((68, image.shape[1], 3), dtype=image.dtype)
    header[:] = (28, 24, 20)
    cv2.putText(header, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    cv2.putText(header, subtitle, (12, 53), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def render(args):
    import cv2
    import numpy as np

    if not isfinite(args.fps) or args.fps <= 0:
        raise ValueError("Playback FPS must be finite and positive")
    sessions = load_sessions(args.dataset, args.sessions, "all")
    predictions = {}
    for line in args.predictions.read_text().splitlines():
        row = json.loads(line)
        key = row["session"], row["frame_index"]
        if key in predictions:
            raise ValueError("Duplicate prediction frame")
        predictions[key] = row
    if args.output_dir.exists():
        raise ValueError("Choose a new output directory")
    args.output_dir.mkdir(parents=True)
    summary = {"playback_fps": args.fps, "timing": "Illustrative playback, not original recording time",
               "arrow_origin": "Heuristic head-box origin, not measured eye coordinates", "sessions": {}}
    green, blue, red = (80, 240, 80), (255, 180, 50), (80, 100, 255)
    for name, session in sessions.items():
        source = OriginalFrames(args.original_frames / name.replace("_", "/", 1))
        writers, index, scored = [], 0, 0
        try:
            while True:
                ok, image = source.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                if not writers:
                    for suffix, size in [("labels", (width, height+68)),
                                         ("comparison", (width*3, height+68))]:
                        writer = cv2.VideoWriter(str(args.output_dir/f"{name}_{suffix}.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
                        if not writer.isOpened():
                            raise RuntimeError("OpenCV could not create the video writer")
                        writers.append(writer)
                label = session["labels"].get(index)
                labeled = image.copy()
                note = "No annotation for this frame"
                if label:
                    head = tuple(v*(width if i%2 == 0 else height) for i,v in enumerate(label["head"]))
                    point = (label["point"][0]*width, label["point"][1]*height)
                    mask, status = load_mask(label["mask"], (width, height), point)
                    if mask is not None:
                        tint = labeled.copy()
                        tint[mask] = green
                        labeled = cv2.addWeighted(labeled, .65, tint, .35, 0)
                        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(labeled, contours, -1, green, 1)
                    draw_target(cv2, labeled, head, point, green)
                    note = f"{label['split']} label | region: {status}"
                tag = f"{name} | frame {index} | "
                labeled = panel(cv2, labeled, "DATASET LABELS", tag+note, green)
                panels = [labeled]
                row = predictions.get((name, index))
                if row:
                    if row["image_size"] != [width, height]:
                        raise ValueError("Prediction resolution differs from video")
                    scored += 1
                for mode, title, color in [("annotated_head", "MODEL + ANNOTATED HEAD", blue),
                                            ("automatic", "MODEL + AUTOMATIC HEAD", red)]:
                    predicted = image.copy()
                    message = "Not evaluated on this frame"
                    if row:
                        value = row[mode]
                        draw_target(cv2, predicted, value.get("head_bbox"), value["point"], color)
                        message = ("Gaze estimated" if value["point"] is not None
                                   else "Unknown: "+str(value["reason"]))
                    panels.append(panel(cv2, predicted, title, tag+message, color))
                writers[0].write(labeled)
                writers[1].write(np.concatenate(panels, axis=1))
                index += 1
        finally:
            source.release()
            for writer in writers:
                writer.release()
        if not index or max(session["labels"]) >= index:
            raise ValueError("Original sequence does not cover annotations")
        summary["sessions"][name] = {"frames": index, "prediction_frames": scored,
                                     "duration_s": index/args.fps}
        print(f"Rendered {name}: {index} frames, {scored} prediction frames", flush=True)
    (args.output_dir/"video_manifest.json").write_text(json.dumps(summary, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--original-frames", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.)
    args = parser.parse_args()
    try:
        render(args)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Video export failed: {exc}\n")


if __name__ == "__main__":
    main()
