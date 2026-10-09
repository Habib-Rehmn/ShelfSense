import io
import json
from contextlib import redirect_stderr, redirect_stdout
from math import pi, sqrt
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from shelfsense.gaze import Angles, GazeAnalyzer, clip_box
from shelfsense.cli import main
from shelfsense.gaze_adapters import L2CSEstimator, OpenCVFaceDetector
from shelfsense.pipeline import Frame, process
from shelfsense.tracking import Track


class Image:
    shape = (200, 300, 3)

    def __getitem__(self, slices):
        return slices


class GazeTests(unittest.TestCase):
    def setUp(self):
        self.detector = MagicMock()
        self.estimator = MagicMock()
        self.estimator.estimate.return_value = Angles(0, 0)
        self.analyzer = GazeAnalyzer(self.detector, self.estimator)
        self.track = Track(1, (0, 0, 100, 200), .9)

    def test_clipping_handles_partial_and_invalid_boxes(self):
        self.assertEqual(clip_box((-1.5, -2, 50.2, 500), 300, 200), (0, 0, 51, 200))
        for box in [(310, 0, 350, 10), (50, 0, 10, 10),
                    (0, 0, float("nan"), 10)]:
            self.assertIsNone(clip_box(box, 300, 200))

    def test_direction_convention_and_normalization(self):
        self.assertEqual(Angles(0, 0).direction, (0, 0, -1))
        self.assertAlmostEqual(Angles(pi / 2, 0).direction[0], -1)
        self.assertAlmostEqual(Angles(0, pi / 2).direction[1], -1)
        direction = Angles(.7, -.4).direction
        self.assertAlmostEqual(sqrt(sum(v * v for v in direction)), 1)
        for value in [float("nan"), float("inf"), 180]:
            with self.assertRaises(ValueError):
                Angles(value, 0)

    def test_face_coordinates_and_crop_are_in_full_frame(self):
        self.detector.detect.return_value = [(-5, 10, 40, 60)]
        result = self.analyzer.observe(Image(), [self.track])[0]
        self.assertEqual(result.face_bbox, (0, 10, 40, 60))
        self.assertEqual(result.status, "estimated")
        self.estimator.estimate.assert_called_once_with((slice(10, 60), slice(0, 40)))

    def test_missing_small_and_lower_body_faces(self):
        for faces, reason in [([], "no_face"), ([(10, 10, 20, 20)], "face_too_small"),
                              ([(10, 150, 60, 190)], "no_face")]:
            self.detector.detect.return_value = faces
            result = self.analyzer.observe(Image(), [self.track])[0]
            self.assertEqual(result.reason, reason)
            self.assertIsNone(result.direction_camera)
        self.estimator.estimate.assert_not_called()

    def test_one_face_cannot_be_assigned_to_two_shoppers(self):
        self.detector.detect.return_value = [(10, 10, 60, 60)]
        second = Track(2, (0, 0, 110, 200), .9)
        results = self.analyzer.observe(Image(), [self.track, second])
        self.assertEqual([r.reason for r in results], ["ambiguous_face"] * 2)
        self.estimator.estimate.assert_not_called()

    def test_two_faces_in_one_person_are_ambiguous(self):
        self.detector.detect.return_value = [(10, 10, 50, 50), (50, 10, 90, 50)]
        self.assertEqual(self.analyzer.observe(Image(), [self.track])[0].reason, "ambiguous_face")

    def test_invalid_prediction_is_unknown_but_runtime_failure_propagates(self):
        self.detector.detect.return_value = [(10, 10, 60, 60)]
        self.estimator.estimate.side_effect = lambda _: Angles(float("nan"), 0)
        self.assertEqual(self.analyzer.observe(Image(), [self.track])[0].reason, "invalid_angles")
        self.estimator.estimate.side_effect = RuntimeError("device failed")
        with self.assertRaises(RuntimeError):
            self.analyzer.observe(Image(), [self.track])

    def test_empty_tracks_skip_face_inference(self):
        self.assertEqual(self.analyzer.observe(Image(), []), [])
        self.detector.detect.assert_not_called()

    def test_pipeline_preserves_track_order_and_unknown(self):
        self.detector.detect.return_value = [(10, 10, 60, 60)]
        second = Track(2, (150, 0, 250, 200), .9)
        tracker = SimpleNamespace(track=lambda _: [self.track, second])
        output = io.StringIO()
        process([Frame(0, 0, Image())], tracker, output, "test", gaze=self.analyzer)
        row = json.loads(output.getvalue())
        self.assertEqual(row["schema_version"], 2)
        self.assertEqual([g["track_id"] for g in row["gaze"]], [1, 2])
        self.assertEqual(row["gaze"][1]["reason"], "no_face")
        self.assertEqual(row["gaze"][0]["direction_camera"], [0, 0, -1])


class AdapterTests(unittest.TestCase):
    def test_opencv_detector_converts_xywh_to_xyxy(self):
        cv2 = MagicMock()
        cv2.data.haarcascades = "/opencv/data/"
        cv2.CascadeClassifier.return_value.empty.return_value = False
        cv2.CascadeClassifier.return_value.detectMultiScale.return_value = [(10, 20, 40, 50)]
        with patch.dict("sys.modules", {"cv2": cv2}):
            detector = OpenCVFaceDetector()
            self.assertEqual(detector.detect("bgr"), [(10, 20, 50, 70)])
        cv2.cvtColor.assert_called_once_with("bgr", cv2.COLOR_BGR2GRAY)

    def test_l2cs_initialization_uses_local_weights_and_explicit_device(self):
        torch = MagicMock()
        model = MagicMock()
        utils = SimpleNamespace(getArch=MagicMock(return_value=model), prep_input_numpy=MagicMock())
        with patch.dict("sys.modules", {"torch": torch, "cv2": MagicMock(), "l2cs.utils": utils}):
            L2CSEstimator("local.pkl", "0")
        torch.device.assert_called_once_with("cuda:0")
        torch.load.assert_called_once_with("local.pkl", map_location=torch.device.return_value,
                                           weights_only=True)
        utils.getArch.assert_called_once_with("ResNet50", 90)
        model.to.return_value.eval.assert_called_once()

    def test_l2cs_preprocessing_and_angle_output_order(self):
        # Exercise adapter conversion without importing torch or downloading weights.
        adapter = L2CSEstimator.__new__(L2CSEstimator)
        adapter.cv2 = MagicMock()
        adapter.torch = MagicMock()
        adapter.prepare = MagicMock()
        adapter.device = "cpu"
        adapter.indices = "bins"
        adapter.model = MagicMock(return_value=("horizontal", "vertical"))
        horizontal = MagicMock()
        vertical = MagicMock()
        horizontal.__mul__.return_value.__sub__.return_value.item.return_value = 90
        vertical.__mul__.return_value.__sub__.return_value.item.return_value = -45
        adapter.torch.sum.side_effect = [horizontal, vertical]
        angles = adapter.estimate("bgr crop")
        self.assertAlmostEqual(angles.yaw_rad, pi / 2)
        self.assertAlmostEqual(angles.pitch_rad, -pi / 4)
        adapter.cv2.cvtColor.assert_called_once_with("bgr crop", adapter.cv2.COLOR_BGR2RGB)
        adapter.cv2.resize.assert_called_once_with(adapter.cv2.cvtColor.return_value, (224, 224))
        adapter.prepare.assert_called_once_with(adapter.cv2.resize.return_value, "cpu")


class CliGazeTests(unittest.TestCase):
    def test_gaze_cli_wires_real_analyzer_and_serializes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, weights, gaze_weights = [root / name for name in ("clip.mp4", "yolo.pt", "gaze.pkl")]
            for path in (source, weights, gaze_weights):
                path.touch()
            output = root / "results.jsonl"
            tracker = SimpleNamespace(track=lambda _: [Track(7, (0, 0, 100, 200), .9)])
            detector = SimpleNamespace(detect=lambda _: [(10, 10, 60, 60)])
            estimator = SimpleNamespace(estimate=lambda _: Angles(.1, -.2))
            video = MagicMock()
            video.__enter__.return_value = [Frame(0, 0, Image())]
            with patch("shelfsense.cli.YoloByteTracker", return_value=tracker), \
                 patch("shelfsense.cli.OpenCVFaceDetector", return_value=detector), \
                 patch("shelfsense.cli.L2CSEstimator", return_value=estimator) as l2cs, \
                 patch("shelfsense.cli.VideoReader", return_value=video), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["track", str(source), "--weights", str(weights),
                                       "--gaze-weights", str(gaze_weights), "--gaze-backend", "l2cs", "--output", str(output)]), 0)
            l2cs.assert_called_once_with(str(gaze_weights), "cpu", "ResNet50")
            row = json.loads(output.read_text())
            self.assertEqual(row["gaze"][0]["yaw_rad"], .1)
            self.assertEqual(row["gaze"][0]["track_id"], 7)

    def test_missing_gaze_weights_fail_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, weights = root / "clip.mp4", root / "yolo.pt"
            source.touch()
            weights.touch()
            output = root / "results.jsonl"
            with patch("shelfsense.cli.YoloByteTracker") as tracker, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["track", str(source), "--weights", str(weights), "--gaze-weights",
                      str(root / "absent.pkl"), "--output", str(output)])
            tracker.assert_not_called()
            self.assertFalse(output.exists())
