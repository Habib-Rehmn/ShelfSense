"""Transactional PostgreSQL storage with SQLite for local development."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .ingestion import canonical, HEATMAP_COLUMNS, HEATMAP_ROWS

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS ss_runs (
        run_id TEXT PRIMARY KEY, digest TEXT NOT NULL, metadata TEXT NOT NULL,
        imported_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS ss_sessions (
        run_id TEXT NOT NULL REFERENCES ss_runs(run_id), session_id TEXT NOT NULL,
        summary TEXT NOT NULL, PRIMARY KEY (run_id, session_id))""",
    """CREATE TABLE IF NOT EXISTS ss_events (
        run_id TEXT NOT NULL REFERENCES ss_runs(run_id), event_id TEXT NOT NULL,
        session_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
        PRIMARY KEY (run_id, event_id),
        FOREIGN KEY (run_id, session_id) REFERENCES ss_sessions(run_id, session_id))""",
    """CREATE TABLE IF NOT EXISTS ss_episodes (
        run_id TEXT NOT NULL, episode_id TEXT NOT NULL, session_id TEXT NOT NULL,
        track_id BIGINT NOT NULL, zone_id TEXT NOT NULL, start_time_s DOUBLE PRECISION NOT NULL,
        end_time_s DOUBLE PRECISION NOT NULL, duration_s DOUBLE PRECISION NOT NULL CHECK(duration_s >= 0),
        PRIMARY KEY (run_id, episode_id),
        FOREIGN KEY (run_id, session_id) REFERENCES ss_sessions(run_id, session_id))""",
    "CREATE INDEX IF NOT EXISTS ss_event_session ON ss_events(run_id, session_id, sequence)",
    "CREATE INDEX IF NOT EXISTS ss_episode_zone ON ss_episodes(run_id, zone_id)",
]


class Store:
    def __init__(self, url):
        if not isinstance(url, str) or not url.startswith(("sqlite:///", "postgresql://", "postgres://")):
            raise ValueError("Database URL must use sqlite:/// or postgresql://")
        self.url = url
        self.sqlite = url.startswith("sqlite:///")
        if self.sqlite and url[len("sqlite:///"):] in {"", ":memory:"}:
            raise ValueError("SQLite requires a persistent file path")

    @contextmanager
    def connection(self):
        if self.sqlite:
            path = Path(self.url[len("sqlite:///"):])
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
        else:
            import psycopg
            from psycopg.rows import dict_row
            connection = psycopg.connect(self.url, row_factory=dict_row, connect_timeout=10)
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(self, connection, sql, params=()):
        return connection.execute(sql if self.sqlite else sql.replace("?", "%s"), params)

    def initialize(self):
        with self.connection() as connection:
            for sql in SCHEMA:
                self.execute(connection, sql)

    def ingest(self, data):
        with self.connection() as connection:
            inserted = self.execute(connection,
                "INSERT INTO ss_runs VALUES (?, ?, ?, ?) ON CONFLICT (run_id) DO NOTHING",
                (data["run_id"], data["digest"], canonical(data["metadata"]), datetime.now(timezone.utc).isoformat()))
            if inserted.rowcount == 0:
                existing = self.execute(connection, "SELECT digest FROM ss_runs WHERE run_id=?", (data["run_id"],)).fetchone()
                if existing["digest"] != data["digest"]:
                    raise ValueError("Run ID already exists with different content")
                return {"run_id": data["run_id"], "imported": False}
            for session in data["sessions"]:
                self.execute(connection, "INSERT INTO ss_sessions VALUES (?, ?, ?)",
                             (data["run_id"], session["session_id"], canonical(session)))
            for sequence, event in enumerate(data["events"]):
                self.execute(connection, "INSERT INTO ss_events VALUES (?, ?, ?, ?, ?)",
                             (data["run_id"], event["event_id"], event["session_id"], sequence, canonical(event)))
            for episode in data["episodes"]:
                self.execute(connection, "INSERT INTO ss_episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (data["run_id"], episode["episode_id"], episode["session_id"], episode["track_id"],
                     episode["zone_id"], episode["start_time_s"], episode["end_time_s"], episode["duration_s"]))
        return {"run_id": data["run_id"], "imported": True}

    def runs(self):
        with self.connection() as connection:
            return [{"run_id": row["run_id"], "imported_at": row["imported_at"],
                     **json.loads(row["metadata"])} for row in self.execute(connection,
                     "SELECT * FROM ss_runs ORDER BY imported_at DESC, run_id").fetchall()]

    def sessions(self, run_id):
        with self.connection() as connection:
            if not self.execute(connection, "SELECT run_id FROM ss_runs WHERE run_id=?", (run_id,)).fetchone():
                raise KeyError("Run not found")
            return [json.loads(row["summary"]) for row in self.execute(connection,
                    "SELECT summary FROM ss_sessions WHERE run_id=? ORDER BY session_id", (run_id,)).fetchall()]

    def report(self, run_id, session_id=None):
        sessions = self.sessions(run_id)
        if session_id is not None:
            sessions = [s for s in sessions if s["session_id"] == session_id]
            if not sessions:
                raise KeyError("Session not found")
        products = {}
        heatmap = [[0.] * HEATMAP_COLUMNS for _ in range(HEATMAP_ROWS)]
        for session in sessions:
            for zone_id, values in session["zones"].items():
                product = products.setdefault(zone_id, {"zone_id": zone_id, "episode_count": 0,
                    "unique_tracks": 0, "total_attention_s": 0.})
                for key in ("episode_count", "unique_tracks", "total_attention_s"):
                    product[key] += values[key]
            for y in range(HEATMAP_ROWS):
                for x in range(HEATMAP_COLUMNS):
                    heatmap[y][x] += session["heatmap"][y][x]
        total = sum(product["total_attention_s"] for product in products.values())
        for product in products.values():
            product["avg_attention_s"] = product["total_attention_s"]/product["episode_count"] if product["episode_count"] else 0.
            product["attention_share"] = product["total_attention_s"]/total if total else 0.
        observations = sum(s["track_observations"] for s in sessions)
        valid = sum(s["valid_observations"] for s in sessions)
        return {"run_id": run_id, "session_id": session_id, "session_count": len(sessions),
                "frame_count": sum(s["frame_count"] for s in sessions),
                "track_observations": observations, "valid_observations": valid,
                "coverage": valid/observations if observations else 0.,
                "total_attention_s": total,
                "products": sorted(products.values(), key=lambda p: (-p["total_attention_s"], p["zone_id"])),
                "funnel": {key: sum(s[key] for s in sessions) for key in ("tracked", "viewed", "confirmed")},
                "heatmap": {"columns": HEATMAP_COLUMNS, "rows": HEATMAP_ROWS,
                            "unit": "observed_gaze_seconds", "values": heatmap}}

    def events(self, run_id, session_id=None, limit=100, offset=0):
        if type(limit) is not int or not 1 <= limit <= 500 or type(offset) is not int or offset < 0:
            raise ValueError("Invalid event pagination")
        sessions = self.sessions(run_id)
        if session_id is not None and not any(s["session_id"] == session_id for s in sessions):
            raise KeyError("Session not found")
        sql, params = "SELECT payload FROM ss_events WHERE run_id=?", [run_id]
        if session_id is not None:
            sql += " AND session_id=?"
            params.append(session_id)
        with self.connection() as connection:
            rows = self.execute(connection, sql+" ORDER BY sequence LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
            return [json.loads(row["payload"]) for row in rows]
