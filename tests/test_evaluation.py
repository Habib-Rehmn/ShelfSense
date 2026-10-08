from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from shelfsense.calibration import Calibration
from shelfsense.cli import main
from shelfsense.evaluation import angular_error, evaluate
from shelfsense.provenance import environment_manifest, file_digest
from shelfsense.smoke import run_smoke

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.model = Calibration.load(ROOT/"configs/shelf.synthetic.json")
        self.labels = {"schema_version": 1, "dataset_id": "test", "split": "test", "synthetic": True,
                       "calibration_session_ids": ["calibration-session"], "samples": [
            {"session_id": "heldout", "frame_index": i, "track_id": 1, "zone_id": zone,
             "conditions": {"distance_m": 1 if i < 2 else 2}}
            for i, zone in enumerate(["zone_a", "zone_b", "zone_c", "zone_a"])]}
        self.rows = []
        for i, (zone, point) in enumerate([("zone_a", [.1, .5]), ("zone_c", [.9, .5]), (None, None)]):
            self.rows.append({"schema_version": 3, "session_id": "heldout", "frame_index": i,
                              "timestamp_s": i/10, "image_size": [640, 480], "calibration": self.model.summary(),
                              "tracks": [{"track_id": 1}], "gaze": [], "shelf_targets": [
                {"track_id": 1, "status": "estimated" if zone else "unknown", "zone_id": zone, "shelf_point": point}]})

    def score(self, kind="raw"):
        return evaluate(io.StringIO("\n".join(json.dumps(row) for row in self.rows)),
                        self.labels, self.model, kind)

    def test_unknown_and_missing_predictions_reduce_all_label_accuracy(self):
        report = self.score()
        metrics = report["metrics"]
        self.assertEqual(metrics["coverage"], .5)
        self.assertEqual(metrics["accuracy_all_labels"], .25)
        self.assertEqual(metrics["accuracy_accepted"], .5)
        self.assertEqual(metrics["missing_predictions"], 1)
        self.assertEqual(metrics["confusion"]["counts"], [[1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertAlmostEqual(metrics["macro_f1"], (2/3)/3)
        self.assertEqual(report["by_condition"]["distance_m"]["2"]["coverage"], 0)
        self.assertIsNone(report["by_condition"]["distance_m"]["2"]["accuracy_accepted"])

    def test_optional_point_and_angular_metrics_have_separate_denominators(self):
        self.labels["samples"][0].update(shelf_point=[.2, .5], direction_camera=[0, 0, -1])
        self.rows[0]["gaze"] = [{"track_id": 1, "direction_camera": [1, 0, 0]}]
        metrics = self.score()["metrics"]
        self.assertAlmostEqual(metrics["mean_shelf_point_error"], .1)
        self.assertEqual(metrics["shelf_point_samples"], 1)
        self.assertAlmostEqual(metrics["mean_angular_error_deg"], 90)
        self.assertEqual(metrics["angular_samples"], 1)

    def test_angular_normalization_and_invalid_vectors(self):
        self.assertAlmostEqual(angular_error([0, 0, -2], [0, 0, -1]), 0)
        self.assertAlmostEqual(angular_error([0, 0, -2], [0, 0, 1]), 180)
        for vector in [[0, 0, 0], [0, 0, float("nan")], [1, 2]]:
            with self.assertRaises(ValueError):
                angular_error(vector, [0, 0, -1])

    def test_leakage_duplicate_keys_and_invalid_truth_rejected(self):
        original = deepcopy(self.labels)
        for change in [lambda d: d.update(calibration_session_ids=["heldout"]),
                       lambda d: d.update(split="train"), lambda d: d.update(samples=[]),
                       lambda d: d["samples"].append(d["samples"][0]),
                       lambda d: d["samples"][0].update(zone_id="made-up"),
                       lambda d: d["samples"][0].update(shelf_point=[.9, .5]),
                       lambda d: d["samples"][0].update(conditions={"distance_m": float("nan")})]:
            self.labels = deepcopy(original)
            change(self.labels)
            with self.assertRaises(ValueError):
                self.score()

    def test_prediction_validation_and_complete_smoothed_sessions(self):
        original = deepcopy(self.rows)
        for change in [lambda rows: rows[0]["calibration"].update(calibration_id="wrong"),
                       lambda rows: rows[0].update(image_size=[1, 1]),
                       lambda rows: rows[0]["shelf_targets"][0].update(zone_id="wrong"),
                       lambda rows: rows[1].update(timestamp_s=0),
                       lambda rows: rows[1]["tracks"][0].update(track_id=2)]:
            self.rows = deepcopy(original)
            change(self.rows)
            with self.assertRaises(ValueError):
                self.score()
        self.rows = deepcopy(original)
        with self.assertRaises(ValueError):
            self.score("smoothed")
        metadata = {"run_id": "test-run", "calibration_id": self.model.calibration_id}
        for row in self.rows:
            row.update(schema_version=4, record_type="frame", attention_metadata=metadata,
                       attention=row["shelf_targets"])
        with self.assertRaisesRegex(ValueError, "missing session endings"):
            self.score("smoothed")
        self.rows.append({"schema_version": 4, "record_type": "session_end", "session_id": "heldout",
                          "timestamp_s": .2, "attention_metadata": metadata})
        self.assertEqual(self.score("smoothed")["metrics"]["coverage"], .5)

    def test_unlabeled_predictions_not_scored_and_condition_missing_not_grouped(self):
        self.labels["samples"] = [self.labels["samples"][0]]
        self.assertEqual(self.score()["unlabeled_predictions"], 2)
        self.assertEqual(self.score()["metrics"]["accuracy_all_labels"], 1)

    def test_evaluate_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, labels, output = [Path(directory)/p for p in ("predictions.jsonl", "labels.json", "report.json")]
            source.write_text("\n".join(json.dumps(row) for row in self.rows))
            labels.write_text(json.dumps(self.labels))
            args = ["evaluate", str(source), "--labels", str(labels), "--calibration",
                    str(ROOT/"configs/shelf.synthetic.json"), "--output", str(output)]
            self.assertEqual(main(args), 0)
            self.assertEqual(json.loads(output.read_text())["metrics"]["coverage"], .5)
            with self.assertRaises(SystemExit):
                main(args)


class ReproducibilityTests(unittest.TestCase):
    def test_manifest_hashes_inputs_and_omits_environment_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"sample.txt"
            path.write_text("sample")
            manifest = environment_manifest([path])
            self.assertEqual(manifest["inputs"][0]["sha256"], file_digest(path))
            self.assertNotIn("environment", manifest)
            self.assertIn("python", manifest)
            path.write_text("changed")
            self.assertNotEqual(manifest["inputs"][0]["sha256"], file_digest(path))

    def test_end_to_end_smoke_and_existing_directory_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"smoke"
            result = run_smoke(output, ROOT)
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["frames"], 23)
            self.assertAlmostEqual(result["analytics"]["total_attention_s"], 1.8)
            self.assertAlmostEqual(result["evaluation"]["raw"]["metrics"]["coverage"], 20/21)
            self.assertEqual(result["evaluation"]["raw"]["metrics"]["accuracy_accepted"], 1)
            self.assertTrue((output/"manifest.json").is_file())
            self.assertTrue((output/"analytics.db").is_file())
            with self.assertRaises(FileExistsError):
                run_smoke(output, ROOT)
