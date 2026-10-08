from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from math import pi
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from shelfsense.calibration import Calibration
from shelfsense.cli import main
from shelfsense.gaze import GazeObservation
from shelfsense.geometry import ShelfGeometry, validate_zones, zone_at
from shelfsense.pipeline import Frame, process
from shelfsense.replay import map_gaze_file
from shelfsense.tracking import Track

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT / "configs/shelf.synthetic.json").read_text())


def observation(yaw=0, pitch=0, **kwargs):
    return GazeObservation(1, "estimated", face_bbox=kwargs.get("face_bbox", (300, 180, 340, 220)),
                           yaw_rad=yaw, pitch_rad=pitch)


class GeometryTests(unittest.TestCase):
    def test_perspective_corners_and_diagonal_intersection(self):
        # This quadrilateral has diagonal intersection (250, 212.5), mapping to center.
        geometry = ShelfGeometry([640, 480], [[100, 100], [400, 100], [500, 400], [0, 400]])
        for point, expected in zip(geometry.corners, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            actual = geometry.to_shelf(point)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)
        for a in geometry.to_shelf((250, 212.5)):
            self.assertAlmostEqual(a, .5)

    def test_bad_geometry_is_rejected(self):
        for corners in [ [[0, 0]] * 4, [[0, 0], [100, 100], [100, 0], [0, 100]],
                         [[0, 0], [640, 0], [700, 480], [0, 480]],
                         [[0, 0], [10, 0], [20, 0], [0, 10]],
                         [[0, 0], [100, 0], [50, 20], [100, 100]] ]:
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                ShelfGeometry([640, 480], corners)
        for size in [[0, 10], [20.5, 10], [True, 10]]:
            with self.assertRaises(ValueError):
                ShelfGeometry(size, fixture()["corners"])

    def test_zone_edges_are_unique_and_outer_border_included(self):
        zones = validate_zones(fixture()["zones"])
        for point, expected in [((0, 0), "zone_a"), ((1/3, .5), "zone_b"),
                                ((2/3, .5), "zone_c"), ((1, 1), "zone_c"),
                                ((-.001, .5), None), ((1.001, .5), None)]:
            self.assertEqual(zone_at(point, zones), expected)

    def test_zone_overlap_duplicate_ids_and_invalid_values(self):
        invalid = [[], [{"id": "a", "bounds": [0, 0, 2, 1]}],
                   [{"id": "a", "bounds": [0, 0, .7, 1]}, {"id": "b", "bounds": [.6, 0, 1, 1]}],
                   [{"id": "a", "bounds": [0, 0, .5, 1]}, {"id": "a", "bounds": [.5, 0, 1, 1]}]]
        for zones in invalid:
            with self.assertRaises(ValueError):
                validate_zones(zones)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(fixture())

    def test_fit_maps_three_zones_and_persists_identically(self):
        for yaw, expected in [(.4, "zone_a"), (0, "zone_b"), (-.4, "zone_c")]:
            result = self.calibration.map_observation(observation(yaw), (640, 480))
            self.assertEqual(result.zone_id, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            self.calibration.save(path)
            loaded = Calibration.load(path)
            self.assertEqual(loaded.calibration_id, self.calibration.calibration_id)
            self.assertEqual(loaded.map_observation(observation(), (640, 480)),
                             self.calibration.map_observation(observation(), (640, 480)))
            with self.assertRaises(FileExistsError):
                self.calibration.save(path)

    def test_unknown_gaze_and_out_of_range_are_not_forced_into_zones(self):
        cases = [(GazeObservation(1, "unknown", "no_face"), "gaze_unavailable"),
                 (observation(.7), "outside_calibration"),
                 (observation(face_bbox=(310, 180, 350, 220)), "outside_calibration"),
                 (observation(pi), "invalid_gaze"), (observation(float("nan")), "invalid_gaze")]
        for item, reason in cases:
            result = self.calibration.map_observation(item, (640, 480))
            self.assertEqual(result.status, "unknown")
            self.assertEqual(result.reason, reason)
            self.assertIsNone(result.zone_id)

    def test_resolution_mismatch_rejected_even_with_no_tracks(self):
        with self.assertRaises(ValueError):
            self.calibration.map_frame([], (1920, 1080))

    def test_degenerate_and_malformed_samples_rejected(self):
        for change in [lambda d: d.update(samples=d["samples"][:2]),
                       lambda d: d.update(schema_version=42),
                       lambda d: d["samples"][0].update(target_image=[0, 0]),
                       lambda d: d["samples"][0].update(yaw_rad=float("inf")),
                       lambda d: d["samples"][0].update(face_bbox=[-1, 0, 10, 10]),
                       lambda d: [s.update(pitch_rad=s["yaw_rad"]) for s in d["samples"]],
                       lambda d: [s.update(pitch_rad=0) for s in d["samples"]]]:
            data = fixture()
            change(data)
            with self.assertRaises(ValueError):
                Calibration(data)

    def test_varying_head_position_is_learned_with_independent_samples(self):
        data = fixture()
        data["samples"] = []
        for cx, cy, yaw, pitch in [(x, y, a, b) for x in [250, 350] for y in [150, 230]
                                  for a in [-.2, .2] for b in [-.2, .2]]:
            u, v = .5 + .3*(cx/640-.5) - .5*yaw, .5 + .2*(cy/480-.4) - .5*pitch
            data["samples"].append(dict(face_bbox=[cx-20, cy-20, cx+20, cy+20],
                                        yaw_rad=yaw, pitch_rad=pitch,
                                        target_image=[80+480*u, 80+320*v]))
        model = Calibration(data)
        result = model.map_observation(observation(.1, -.1, face_bbox=(280, 170, 320, 210)), (640, 480))
        self.assertEqual(result.status, "estimated")
        self.assertAlmostEqual(result.shelf_point[0], .5+.3*(300/640-.5)-.05)
        self.assertAlmostEqual(result.shelf_point[1], .5+.2*(190/480-.4)+.05)

    def test_off_shelf_prediction_and_zone_gaps(self):
        data = fixture()
        # Independent samples on a diamond leave corner combinations unseen.
        data["samples"] = []
        for yaw, pitch in [(-.4, 0), (.4, 0), (0, -.3), (0, .3), (0, 0), (.1, .1)]:
            u = .5 + yaw + pitch
            data["samples"].append(dict(face_bbox=[300, 180, 340, 220], yaw_rad=yaw,
                                        pitch_rad=pitch, target_image=[80+480*u, 240]))
        model = Calibration(data)
        result = model.map_observation(observation(.4, .3), (640, 480))
        self.assertEqual(result.reason, "off_shelf")
        data = fixture()
        data["zones"] = [{"id": "left", "bounds": [0, 0, .2, 1]}]
        result = Calibration(data).map_observation(observation(), (640, 480))
        self.assertEqual(result.reason, "unmapped_zone")

    def test_input_mutation_does_not_change_saved_calibration(self):
        data = fixture()
        model = Calibration(data)
        data["shelf_id"] = "changed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            model.save(path)
            self.assertEqual(Calibration.load(path).calibration_id, model.calibration_id)


class IntegrationTests(unittest.TestCase):
    def test_pipeline_schema_three(self):
        image = SimpleNamespace(shape=(480, 640, 3))
        tracker = SimpleNamespace(track=lambda _: [Track(1, (260, 140, 380, 470), .9)])
        gaze = SimpleNamespace(observe=lambda *args: [observation()])
        output = io.StringIO()
        process([Frame(0, 0, image)], tracker, output, gaze=gaze, calibration=Calibration(fixture()))
        row = json.loads(output.getvalue())
        self.assertEqual(row["schema_version"], 3)
        self.assertEqual(row["shelf_targets"][0]["zone_id"], "zone_b")
        self.assertTrue(row["calibration"]["fit_rmse_is_training_error"])

    def test_calibration_requires_gaze(self):
        with self.assertRaises(ValueError):
            process([], None, io.StringIO(), calibration=Calibration(fixture()))

    def test_cli_fit_and_replay(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            calibration, output = Path(directory)/"calibration.json", Path(directory)/"targets.jsonl"
            self.assertEqual(main(["calibrate", str(ROOT/"configs/shelf.synthetic.json"),
                                   "--output", str(calibration)]), 0)
            self.assertEqual(main(["map-gaze", str(ROOT/"examples/gaze.synthetic.jsonl"),
                                   "--calibration", str(calibration), "--output", str(output)]), 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["shelf_targets"][0]["zone_id"] for r in rows],
                             ["zone_a", "zone_b", "zone_c", None, None])

    def test_replay_rejects_wrong_ids_resolution_and_nonmonotonic_time(self):
        original = json.loads((ROOT/"examples/gaze.synthetic.jsonl").read_text().splitlines()[0])
        for mutation in [lambda r: r.update(image_size=[320, 240]),
                         lambda r: r["gaze"][0].update(track_id=8),
                         lambda r: r.update(timestamp_s=float("nan")),
                         lambda r: r.update(schema_version=1),
                         lambda r: r.pop("image_size")]:
            row = deepcopy(original)
            mutation(row)
            with self.assertRaisesRegex(ValueError, "line 1"):
                map_gaze_file(io.StringIO(json.dumps(row)), io.StringIO(), Calibration(fixture()))
        with self.assertRaisesRegex(ValueError, "line 2"):
            map_gaze_file(io.StringIO((json.dumps(original)+"\n")*2), io.StringIO(), Calibration(fixture()))
