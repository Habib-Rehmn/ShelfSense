from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from shelfsense.attention import AttentionConfig, AttentionEngine
from shelfsense.attention_replay import derive_attention
from shelfsense.calibration import Calibration
from shelfsense.cli import main
from shelfsense.gaze import GazeObservation
from shelfsense.geometry import zone_at
from shelfsense.pipeline import Frame, process
from shelfsense.tracking import Track

ROOT = Path(__file__).resolve().parents[1]


def calibration():
    return Calibration.load(ROOT / "configs/shelf.synthetic.json")


def frame(t, point=(.5, .5), session="s1", track_id=1, missing=False):
    model = calibration()
    target = {"track_id": track_id, "status": "estimated" if point else "unknown",
              "reason": None if point else "no_face", "shelf_point": point,
              "zone_id": zone_at(point, model.zones) if point else None}
    return {"schema_version": 3, "session_id": session, "frame_index": round(t*1000),
            "timestamp_s": t, "calibration": model.summary(), "image_size": [640, 480],
            "tracks": [] if missing else [{"track_id": track_id}],
            "shelf_targets": [] if missing else [target]}


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.engine = AttentionEngine(calibration())

    def confirm(self):
        rows = [self.engine.process(frame(i/10)) for i in range(6)]
        return rows

    def test_threshold_and_session_end_use_observed_duration(self):
        rows = self.confirm()
        self.assertFalse(any(e["type"] == "ATTENTION_STARTED" for r in rows[:5] for e in r["events"]))
        start = rows[-1]["events"][0]
        self.assertEqual(start["type"], "ATTENTION_STARTED")
        self.assertEqual(start["timestamp_s"], 0)
        self.assertEqual(start["emitted_at_s"], .5)
        end, left = self.engine.finish()[0]["events"]
        self.assertEqual(end["episode_id"], start["episode_id"])
        self.assertEqual(end["duration_s"], .5)
        self.assertEqual(end["end_time_s"], .5)
        self.assertEqual(left["type"], "SHOPPER_LEFT")
        self.assertEqual(self.engine.finish(), [])
        with self.assertRaises(ValueError):
            self.engine.process(frame(.6))

    def test_short_candidate_emits_no_attention_events(self):
        self.engine.process(frame(0))
        self.engine.process(frame(.1))
        events = self.engine.finish()[0]["events"]
        self.assertEqual([e["type"] for e in events], ["SHOPPER_LEFT"])

    def test_unknown_closes_at_previous_valid_time_and_resets_smoothing(self):
        self.confirm()
        result = self.engine.process(frame(.6, None))
        self.assertEqual(result["events"][0]["duration_s"], .5)
        self.assertEqual(result["events"][0]["reason"], "gaze_unavailable")
        self.assertEqual(result["attention"][0]["status"], "unknown")
        resumed = self.engine.process(frame(.7, (.9, .5)))
        self.assertEqual(resumed["attention"][0]["shelf_point"], (.9, .5))
        self.assertEqual(resumed["attention"][0]["candidate_duration_s"], 0)

    def test_missing_track_closes_attention_before_shopper_timeout(self):
        self.confirm()
        result = self.engine.process(frame(.6, missing=True))
        self.assertEqual([e["type"] for e in result["events"]], ["ATTENTION_ENDED"])
        self.assertEqual(result["events"][0]["end_time_s"], .5)
        result = self.engine.process(frame(1.5, missing=True))
        self.assertEqual([e["type"] for e in result["events"]], ["SHOPPER_LEFT"])
        self.assertEqual(result["events"][0]["timestamp_s"], .5)
        self.assertEqual(self.engine.sessions["s1"].tracks, {})

    def test_observation_gap_is_not_counted_or_smoothed_across(self):
        self.confirm()
        result = self.engine.process(frame(.9, (.1, .5)))
        self.assertEqual(result["events"][0]["reason"], "observation_gap")
        self.assertEqual(result["events"][0]["duration_s"], .5)
        self.assertEqual(result["attention"][0]["candidate_duration_s"], 0)
        self.assertEqual(result["attention"][0]["shelf_point"], (.1, .5))

    def test_same_id_after_timeout_emits_leave_then_enter(self):
        self.confirm()
        result = self.engine.process(frame(2))
        self.assertEqual([e["type"] for e in result["events"]],
                         ["ATTENTION_ENDED", "SHOPPER_LEFT", "SHOPPER_ENTERED"])
        self.assertIsNone(result["attention"][0]["episode_id"])

    def test_short_track_absence_restarts_candidate_without_new_entry(self):
        self.confirm()
        self.engine.process(frame(.6, missing=True))
        result = self.engine.process(frame(.7))
        self.assertEqual(result["events"], [])
        self.assertEqual(result["attention"][0]["candidate_duration_s"], 0)

    def test_exact_gap_limit_preserves_episode(self):
        self.engine.process(frame(0))
        self.engine.process(frame(.25))
        result = self.engine.process(frame(.5))
        self.assertEqual([e["type"] for e in result["events"]], ["ATTENTION_STARTED"])

    def test_zone_change_ends_old_episode_and_starts_new_candidate(self):
        self.engine = AttentionEngine(calibration(), AttentionConfig(smoothing_tau_s=.001))
        self.confirm()
        result = self.engine.process(frame(.6, (.1, .5)))
        self.assertEqual(result["events"][0]["reason"], "zone_changed")
        self.assertEqual(result["events"][0]["zone_id"], "zone_b")
        self.assertEqual(result["attention"][0]["zone_id"], "zone_a")
        self.assertEqual(result["attention"][0]["candidate_duration_s"], 0)

    def test_large_motion_within_same_zone_resets_episode(self):
        self.engine = AttentionEngine(calibration(), AttentionConfig(smoothing_tau_s=.001, max_radius=.03))
        self.confirm()
        result = self.engine.process(frame(.6, (.55, .5)))
        self.assertEqual(result["events"][0]["reason"], "point_moved")
        self.assertEqual(result["attention"][0]["zone_id"], "zone_b")

    def test_time_based_ema_matches_across_sampling_rates(self):
        points = []
        for step in [.1, .2]:
            engine = AttentionEngine(calibration(), AttentionConfig(max_radius=1))
            engine.process(frame(0, (.45, .5)))
            for i in range(1, round(1/step)+1):
                result = engine.process(frame(i*step, (.55, .5)))
            points.append(result["attention"][0]["shelf_point"][0])
        self.assertAlmostEqual(*points)

    def test_same_track_id_in_two_sessions_is_independent(self):
        self.confirm()
        result = self.engine.process(frame(0, session="s2"))
        self.assertEqual(result["attention"][0]["candidate_duration_s"], 0)
        summaries = self.engine.finish()
        self.assertEqual(len(summaries), 2)
        ended = [e for s in summaries for e in s["events"] if e["type"] == "ATTENTION_ENDED"]
        self.assertEqual([e["session_id"] for e in ended], ["s1"])

    def test_track_ids_have_independent_candidates(self):
        self.confirm()
        row = frame(.6)
        other = frame(.6, (.1, .5), track_id=2)
        row["tracks"] += other["tracks"]
        row["shelf_targets"] += other["shelf_targets"]
        result = self.engine.process(row)
        self.assertIsNotNone(result["attention"][0]["episode_id"])
        self.assertIsNone(result["attention"][1]["episode_id"])

    def test_invalid_input_is_rejected_without_advancing_state(self):
        self.engine.process(frame(0))
        for change in [lambda r: r.update(timestamp_s=float("nan")),
                       lambda r: r["calibration"].update(calibration_id="wrong"),
                       lambda r: r.update(image_size=[320, 240]),
                       lambda r: r["shelf_targets"][0].update(zone_id="wrong"),
                       lambda r: r["shelf_targets"][0].update(shelf_point=[-1, .5]),
                       lambda r: r["tracks"][0].update(track_id=2),
                       lambda r: r.update(timestamp_s=0)]:
            row = frame(.1)
            change(row)
            with self.assertRaises(ValueError):
                self.engine.process(row)
        result = self.engine.process(frame(.1))
        self.assertEqual(result["attention"][0]["candidate_duration_s"], .1)

    def test_unmapped_smoothed_point_ends_episode(self):
        data = json.loads((ROOT/"configs/shelf.synthetic.json").read_text())
        data["zones"] = [{"id": "left", "bounds": [0, 0, .4, 1]},
                         {"id": "right", "bounds": [.6, 0, 1, 1]}]
        model = Calibration(data)
        engine = AttentionEngine(model, AttentionConfig(smoothing_tau_s=.15))
        def custom(t, x):
            row = frame(t, (x, .5))
            row["calibration"] = model.summary()
            row["shelf_targets"][0]["zone_id"] = zone_at((x, .5), model.zones)
            return row
        for i in range(6):
            engine.process(custom(i/10, .3))
        result = engine.process(custom(.6, .7))
        self.assertEqual(result["attention"][0]["reason"], "unmapped_zone")
        self.assertEqual(result["events"][0]["reason"], "unmapped_zone")

    def test_config_rejects_invalid_thresholds(self):
        for name in AttentionConfig.__dataclass_fields__:
            for value in [0, -1, float("nan"), float("inf"), True]:
                with self.assertRaises(ValueError):
                    AttentionConfig(**{name: value})
        with self.assertRaises(ValueError):
            AttentionConfig(max_gap_s=2, track_timeout_s=1)


class IntegrationTests(unittest.TestCase):
    def test_tracking_cli_enables_temporal_engine(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, weights, gaze_weights = [root/name for name in ["video.mp4", "yolo.pt", "gaze.pkl"]]
            for path in (source, weights, gaze_weights):
                path.touch()
            output = root/"attention.jsonl"
            tracker = SimpleNamespace(track=lambda _: [Track(1, (260, 140, 380, 470), 1)])
            gaze = SimpleNamespace(observe=lambda *args: [GazeObservation(
                1, "estimated", face_bbox=(300, 180, 340, 220), yaw_rad=0, pitch_rad=0)])
            video = MagicMock()
            video.__enter__.return_value = [Frame(i, i/10, SimpleNamespace(shape=(480, 640, 3))) for i in range(6)]
            with patch("shelfsense.cli.YoloByteTracker", return_value=tracker), \
                 patch("shelfsense.cli.OpenCVFaceDetector"), patch("shelfsense.cli.L2CSEstimator"), \
                 patch("shelfsense.cli.GazeAnalyzer", return_value=gaze), \
                 patch("shelfsense.cli.VideoReader", return_value=video):
                self.assertEqual(main(["track", str(source), "--weights", str(weights),
                                       "--gaze-weights", str(gaze_weights), "--gaze-backend", "l2cs", "--calibration",
                                       str(ROOT/"configs/shelf.synthetic.json"), "--attention",
                                       "--output", str(output)]), 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[-1]["record_type"], "session_end")
            self.assertEqual(rows[-1]["events"][0]["duration_s"], .5)

    def test_replay_writes_summary_and_unique_event_ids(self):
        rows = [frame(i/10) for i in range(6)]
        output = io.StringIO()
        count = derive_attention(io.StringIO("\n".join(json.dumps(r) for r in rows)),
                                 output, AttentionEngine(calibration()))
        result = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(count, 6)
        self.assertEqual(len(result), 7)
        self.assertEqual(result[-1]["record_type"], "session_end")
        events = [e for r in result for e in r["events"]]
        self.assertEqual(len({e["event_id"] for e in events}), len(events))

    def test_replay_error_has_line_number_and_no_completion_marker(self):
        output = io.StringIO()
        with self.assertRaisesRegex(ValueError, "line 2"):
            derive_attention(io.StringIO(json.dumps(frame(0))+"\n{}"), output, AttentionEngine(calibration()))
        self.assertNotIn('"record_type": "session_end"', output.getvalue())
        with self.assertRaises(ValueError):
            derive_attention(io.StringIO(""), io.StringIO(), AttentionEngine(calibration()))

    def test_pipeline_finalizes_confirmed_episode(self):
        model = calibration()
        tracker = SimpleNamespace(track=lambda _: [Track(1, (260, 140, 380, 470), 1)])
        gaze = SimpleNamespace(observe=lambda *args: [GazeObservation(
            1, "estimated", face_bbox=(300, 180, 340, 220), yaw_rad=0, pitch_rad=0)])
        frames = [Frame(i, i/10, SimpleNamespace(shape=(480, 640, 3))) for i in range(6)]
        output = io.StringIO()
        self.assertEqual(process(frames, tracker, output, gaze=gaze, calibration=model,
                                 attention=AttentionEngine(model)), 6)
        result = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(result[-1]["events"][0]["duration_s"], .5)
        self.assertEqual(result[-1]["events"][0]["type"], "ATTENTION_ENDED")

    def test_cli_replay_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, output = Path(directory)/"targets.jsonl", Path(directory)/"attention.jsonl"
            source.write_text("\n".join(json.dumps(frame(i/10)) for i in range(6)))
            args = ["attention", str(source), "--calibration", str(ROOT/"configs/shelf.synthetic.json"),
                    "--output", str(output), "--min-duration-s", ".4"]
            self.assertEqual(main(args), 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(any(e["type"] == "ATTENTION_STARTED" for e in rows[4]["events"]))
            before = output.read_text()
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(output.read_text(), before)
