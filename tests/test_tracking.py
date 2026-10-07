import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from shelfsense.cli import main
from shelfsense.pipeline import Frame, process
from shelfsense.tracking import Track, YoloByteTracker
from shelfsense.video import VideoReader


class PipelineTests(unittest.TestCase):
    def test_tracks_and_empty_frames_keep_media_time(self):
        tracker = MagicMock()
        tracker.track.side_effect = [[Track(7, (1, 2, 5, 8), .8)], [],
                                     [Track(7, (2, 2, 6, 8), .9)]]
        output = io.StringIO()
        self.assertEqual(process([Frame(i, i / 10, None) for i in range(3)],
                                 tracker, output, "session-a"), 3)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(rows[1]["tracks"], [])
        self.assertEqual(rows[2]["timestamp_s"], .2)
        self.assertEqual(rows[2]["tracks"][0]["track_id"], 7)
        self.assertEqual(rows[0]["session_id"], "session-a")

    def test_invalid_timestamps_and_indices(self):
        for frame in [Frame(0, .1, None), Frame(1, 0, None),
                      Frame(1, float("nan"), None), Frame(1, -.1, None)]:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                process([Frame(0, 0, None), frame],
                        SimpleNamespace(track=lambda _: []), io.StringIO())

    def test_duplicate_tracks_rejected(self):
        track = Track(1, (0, 0, 10, 10), 1)
        with self.assertRaises(ValueError):
            process([Frame(0, 0, None)], SimpleNamespace(track=lambda _: [track, track]),
                    io.StringIO())

    def test_invalid_boxes_and_confidence(self):
        for bbox, confidence in [((0, 0, 0, 10), 1), ((0, 0, 10, 10), float("nan")),
                                 ((0, 0, float("inf"), 10), 1)]:
            with self.assertRaises(ValueError):
                Track(1, bbox, confidence)

    def test_sessions_are_distinct(self):
        ids = []
        for _ in range(2):
            output = io.StringIO()
            process([Frame(0, 0, None)], SimpleNamespace(track=lambda _: []), output)
            ids.append(json.loads(output.getvalue())["session_id"])
        self.assertNotEqual(*ids)


class AdapterTests(unittest.TestCase):
    def test_ultralytics_contract(self):
        model = MagicMock()
        model.names = {0: "person"}
        boxes = MagicMock()
        boxes.id.cpu().tolist.return_value = [4]
        boxes.xyxy.cpu().tolist.return_value = [[1, 2, 10, 20]]
        boxes.conf.cpu().tolist.return_value = [.9]
        model.track.return_value = [SimpleNamespace(boxes=boxes)]
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda _: model)}):
            tracker = YoloByteTracker("local.pt")
            self.assertEqual(tracker.track("image")[0].track_id, 4)
            model.track.assert_called_once_with(
                "image", persist=True, tracker="bytetrack.yaml", classes=[0],
                conf=.1, device="cpu", verbose=False)
            boxes.id = None
            self.assertEqual(tracker.track("image"), [])

    def test_video_releases_on_processing_error(self):
        capture = MagicMock()
        capture.get.return_value = 25
        capture.read.return_value = (True, "image")
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FPS=5)
        with patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertRaises(RuntimeError):
                with VideoReader("clip.mp4") as video:
                    frames = iter(video)
                    self.assertEqual(next(frames).timestamp_s, 0)
                    self.assertEqual(next(frames).timestamp_s, .04)
                    raise RuntimeError("inference failed")
        capture.release.assert_called_once()

    def test_invalid_fps_releases_capture(self):
        capture = MagicMock()
        capture.get.return_value = 0
        with patch.dict("sys.modules", {"cv2": SimpleNamespace(
                VideoCapture=lambda _: capture, CAP_PROP_FPS=5)}):
            with self.assertRaises(ValueError):
                with VideoReader("clip.mp4"):
                    pass
        capture.release.assert_called_once()


class CliTests(unittest.TestCase):
    def test_demo_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo.jsonl"
            self.assertEqual(main(["demo", "--output", str(output)]), 0)
            content = output.read_text()
            self.assertEqual(len(content.splitlines()), 30)
            with self.assertRaises(SystemExit) as exc:
                main(["demo", "--output", str(output)])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(output.read_text(), content)

    def test_missing_source_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing.jsonl"
            with self.assertRaises(SystemExit):
                main(["track", str(Path(directory) / "absent.mp4"), "--weights",
                      "absent.pt", "--output", str(output)])
            self.assertFalse(output.exists())
