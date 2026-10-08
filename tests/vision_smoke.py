"""Opt-in real YOLO26n/OpenCV smoke; synthetic pixels are not an accuracy test."""

import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from shelfsense.cli import main
from shelfsense.gaze_adapters import OpenCVFaceDetector
from shelfsense.provenance import environment_manifest


def run(weights):
    weights = Path(weights)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video, output = root/"blank.avi", root/"tracks.jsonl"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (640, 480))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the synthetic video")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            for _ in range(10):
                writer.write(frame)
        finally:
            writer.release()
        if OpenCVFaceDetector().detect(frame):
            raise AssertionError("Frontal-face baseline reported a face in a blank frame")
        main(["track", str(video), "--weights", str(weights), "--output", str(output), "--device", "cpu"])
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if len(rows) != 10 or [r["frame_index"] for r in rows] != list(range(10)):
            raise AssertionError("Video tracking did not preserve all ten frames")
        if abs(rows[-1]["timestamp_s"]-.9) > 1e-8 or len({r["session_id"] for r in rows}) != 1:
            raise AssertionError("Video timestamps or session identity changed")
        result = {"synthetic": True, "model": "YOLO26n", "frames": len(rows),
                  "track_observations": sum(len(r["tracks"]) for r in rows),
                  "accuracy_measured": False, "manifest": environment_manifest([weights])}
        Path("/tmp/vision-smoke-report.json").write_text(json.dumps(result, indent=2))
        print("YOLO26n CPU and OpenCV smoke passed: 10 synthetic frames; no accuracy claim.")


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
