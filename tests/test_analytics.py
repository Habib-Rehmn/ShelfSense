from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from shelfsense.attention import AttentionEngine
from shelfsense.attention_replay import derive_attention
from shelfsense.calibration import Calibration
from shelfsense.cli import main
from shelfsense.ingestion import read_export
from shelfsense.replay import map_gaze_file
from shelfsense.storage import Store

ROOT = Path(__file__).resolve().parents[1]


def export_text():
    model = Calibration.load(ROOT/"configs/shelf.synthetic.json")
    targets, attention = io.StringIO(), io.StringIO()
    with (ROOT/"examples/attention-gaze.synthetic.jsonl").open() as source:
        map_gaze_file(source, targets, model)
    derive_attention(io.StringIO(targets.getvalue()), attention, AttentionEngine(model))
    return attention.getvalue(), model


class ImportTests(unittest.TestCase):
    def test_complete_run_metrics_and_no_raw_frames_persisted(self):
        text, model = export_text()
        data = read_export(io.StringIO(text), model)
        session = data["sessions"][0]
        self.assertEqual(len(data["episodes"]), 2)
        self.assertEqual(session["frame_count"], 23)
        self.assertEqual(session["track_observations"], 21)
        self.assertEqual(session["valid_observations"], 20)
        self.assertEqual((session["tracked"], session["viewed"], session["confirmed"]), (1, 1, 1))
        self.assertAlmostEqual(sum(sum(row) for row in session["heatmap"]), 1.8)
        self.assertNotIn("face_bbox", json.dumps(data))
        self.assertNotIn("shelf_targets", json.dumps(data))

    def test_missing_completion_and_tampered_events_are_rejected(self):
        text, model = export_text()
        rows = [json.loads(line) for line in text.splitlines()]
        for values in [rows[:-1], rows+[rows[-1]], rows+[rows[0]]]:
            with self.assertRaises(ValueError):
                read_export(io.StringIO("\n".join(json.dumps(row) for row in values)), model)
        changed = deepcopy(rows)
        for row in changed:
            for event in row["events"]:
                if event["type"] == "ATTENTION_ENDED":
                    event["duration_s"] = 100
        with self.assertRaisesRegex(ValueError, "recomputed"):
            read_export(io.StringIO("\n".join(json.dumps(row) for row in changed)), model)

    def test_calibration_and_metadata_must_match(self):
        text, model = export_text()
        rows = [json.loads(line) for line in text.splitlines()]
        rows[1]["attention_metadata"]["run_id"] = "different"
        with self.assertRaises(ValueError):
            read_export(io.StringIO("\n".join(json.dumps(row) for row in rows)), model)
        with self.assertRaises(ValueError):
            read_export(io.StringIO(""), model)

    def test_whitespace_does_not_change_digest(self):
        text, model = export_text()
        reformatted = "\n".join(json.dumps(json.loads(line), sort_keys=True) for line in text.splitlines())
        self.assertEqual(read_export(io.StringIO(text), model)["digest"],
                         read_export(io.StringIO(reformatted), model)["digest"])


class StoreFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.url = "sqlite:///"+str(Path(self.temp.name)/"analytics.db")
        self.store = Store(self.url)
        self.store.initialize()
        text, model = export_text()
        self.data = read_export(io.StringIO(text), model)


class StoreTests(StoreFixture, unittest.TestCase):
    def test_idempotent_import_rankings_heatmap_and_filtering(self):
        self.assertTrue(self.store.ingest(self.data)["imported"])
        self.assertFalse(self.store.ingest(self.data)["imported"])
        run_id = self.data["run_id"]
        report = self.store.report(run_id)
        self.assertEqual(len(self.store.runs()), 1)
        self.assertAlmostEqual(report["total_attention_s"], 1.8)
        self.assertAlmostEqual(report["coverage"], 20/21)
        self.assertEqual([p["zone_id"] for p in report["products"]], ["zone_c", "zone_a", "zone_b"])
        self.assertAlmostEqual(sum(p["attention_share"] for p in report["products"]), 1)
        self.assertEqual(report["funnel"], {"tracked": 1, "viewed": 1, "confirmed": 1})
        selected = self.store.report(run_id, self.data["sessions"][0]["session_id"])
        self.assertEqual(selected["products"], report["products"])
        self.assertEqual(len(self.store.events(run_id, limit=2)), 2)
        self.assertNotEqual(self.store.events(run_id, limit=1), self.store.events(run_id, limit=1, offset=1))

    def test_conflict_and_transaction_rollback(self):
        broken = deepcopy(self.data)
        broken["events"].append(broken["events"][0])
        with self.assertRaises(Exception):
            self.store.ingest(broken)
        self.assertEqual(self.store.runs(), [])
        self.store.ingest(self.data)
        changed = deepcopy(self.data)
        changed["digest"] = "conflicting-content"
        with self.assertRaises(ValueError):
            self.store.ingest(changed)
        self.assertAlmostEqual(self.store.report(self.data["run_id"])["total_attention_s"], 1.8)

    def test_runs_remain_separate_and_filters_are_parameterized(self):
        self.store.ingest(self.data)
        text, model = export_text()
        second = read_export(io.StringIO(text), model)
        self.store.ingest(second)
        self.assertEqual(len(self.store.runs()), 2)
        self.assertAlmostEqual(self.store.report(second["run_id"])["total_attention_s"], 1.8)
        with self.assertRaises(KeyError):
            self.store.report("' OR 1=1 --")
        with self.assertRaises(KeyError):
            self.store.report(second["run_id"], "missing")
        with self.assertRaises(ValueError):
            self.store.events(second["run_id"], limit=501)

    def test_cli_import_and_report(self):
        text, model = export_text()
        source = Path(self.temp.name)/"attention.jsonl"
        source.write_text(text)
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["ingest", str(source), "--calibration", str(ROOT/"configs/shelf.synthetic.json"),
                                   "--database-url", self.url]), 0)
        run_id = json.loads(captured.getvalue())["run_id"]
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["report", "--run-id", run_id, "--database-url", self.url]), 0)
        self.assertAlmostEqual(json.loads(captured.getvalue())["total_attention_s"], 1.8)

    def test_two_sessions_are_scoped_and_empty_runs_have_zero_metrics(self):
        model = Calibration.load(ROOT/"configs/shelf.synthetic.json")
        targets = io.StringIO()
        with (ROOT/"examples/attention-gaze.synthetic.jsonl").open() as source:
            map_gaze_file(source, targets, model)
        records = [json.loads(line) for line in targets.getvalue().splitlines()]
        other = deepcopy(records)
        for record in other:
            record["session_id"] = "second-session"
        output = io.StringIO()
        derive_attention(io.StringIO("\n".join(json.dumps(r) for r in records+other)), output, AttentionEngine(model))
        data = read_export(io.StringIO(output.getvalue()), model)
        self.store.ingest(data)
        self.assertEqual(self.store.report(data["run_id"])["funnel"]["tracked"], 2)
        self.assertAlmostEqual(self.store.report(data["run_id"])["total_attention_s"], 3.6)
        self.assertAlmostEqual(self.store.report(data["run_id"], "second-session")["total_attention_s"], 1.8)
        for record in records:
            record.update(tracks=[], gaze=[], shelf_targets=[])
        output = io.StringIO()
        derive_attention(io.StringIO("\n".join(json.dumps(r) for r in records)), output, AttentionEngine(model))
        data = read_export(io.StringIO(output.getvalue()), model)
        self.store.ingest(data)
        report = self.store.report(data["run_id"])
        self.assertEqual(report["total_attention_s"], 0)
        self.assertEqual(report["coverage"], 0)
        self.assertEqual(report["funnel"], {"tracked": 0, "viewed": 0, "confirmed": 0})
        self.assertTrue(all(p["avg_attention_s"] == 0 for p in report["products"]))


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "Install analytics,test extras to test HTTP endpoints")
class ApiTests(StoreFixture, unittest.TestCase):
    def test_http_contract_and_errors(self):
        from fastapi.testclient import TestClient
        from shelfsense.api import create_app
        self.store.ingest(self.data)
        with TestClient(create_app(self.url)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(len(client.get("/runs").json()), 1)
            for path in ["/sessions", "/analytics", "/analytics/products", "/analytics/heatmap", "/analytics/funnel", "/events"]:
                self.assertEqual(client.get(path, params={"run_id": self.data["run_id"]}).status_code, 200)
                self.assertEqual(client.get(path).status_code, 422)
                self.assertEqual(client.get(path, params={"run_id": "missing"}).status_code, 404)
            self.assertEqual(client.get("/events", params={"run_id": self.data["run_id"], "limit": 501}).status_code, 422)
            self.assertEqual(client.post("/runs").status_code, 405)

    def test_uninitialized_storage_returns_sanitized_error(self):
        from fastapi.testclient import TestClient
        from shelfsense.api import create_app
        with TestClient(create_app("sqlite:///"+str(Path(self.temp.name)/"uninitialized.db"))) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(self.temp.name, response.text)


@unittest.skipUnless(os.environ.get("SHELFSENSE_TEST_POSTGRES_URL") and importlib.util.find_spec("psycopg"),
                     "Set SHELFSENSE_TEST_POSTGRES_URL and install psycopg for PostgreSQL integration")
class PostgresTests(unittest.TestCase):
    def test_real_postgres_transactions_and_queries(self):
        store = Store(os.environ["SHELFSENSE_TEST_POSTGRES_URL"])
        store.initialize()
        text, model = export_text()
        data = read_export(io.StringIO(text), model)
        try:
            self.assertTrue(store.ingest(data)["imported"])
            self.assertFalse(store.ingest(data)["imported"])
            self.assertAlmostEqual(store.report(data["run_id"])["total_attention_s"], 1.8)
            self.assertEqual(len(store.sessions(data["run_id"])), 1)
            self.assertTrue(store.events(data["run_id"]))
            conflicting = deepcopy(data)
            conflicting["digest"] = "different-content"
            with self.assertRaises(ValueError):
                store.ingest(conflicting)
        finally:
            with store.connection() as connection:
                for table in ("ss_episodes", "ss_events", "ss_sessions", "ss_runs"):
                    store.execute(connection, f"DELETE FROM {table} WHERE run_id=?", (data["run_id"],))


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Install analytics extra for dashboard AppTest")
class DashboardTests(StoreFixture, unittest.TestCase):
    def test_dashboard_renders_imported_run(self):
        from streamlit.testing.v1 import AppTest
        self.store.ingest(self.data)

        def response(url, timeout):
            parsed = urlparse(url)
            params = {key: value[0] for key, value in parse_qs(parsed.query).items()}
            if parsed.path == "/runs":
                data = self.store.runs()
            elif parsed.path == "/sessions":
                data = self.store.sessions(params["run_id"])
            elif parsed.path == "/analytics":
                data = self.store.report(params["run_id"], params.get("session_id"))
            else:
                data = self.store.events(params["run_id"], params.get("session_id"))
            return io.StringIO(json.dumps(data))

        with patch("urllib.request.urlopen", side_effect=response):
            app = AppTest.from_file(str(ROOT/"dashboard/app.py")).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.metric[0].value, "1.80 s")
        self.assertEqual(app.metric[1].value, "2")

    def test_dashboard_empty_state_and_connection_failure(self):
        from streamlit.testing.v1 import AppTest
        with patch("urllib.request.urlopen", return_value=io.StringIO("[]")):
            app = AppTest.from_file(str(ROOT/"dashboard/app.py")).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.info), 1)
        with patch("urllib.request.urlopen", side_effect=OSError("unavailable")):
            app = AppTest.from_file(str(ROOT/"dashboard/app.py")).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
