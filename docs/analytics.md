# Module 5: persistence, analytics API and dashboard

The analytics layer imports completed schema-4 attention exports. It supports
PostgreSQL with Psycopg 3 and a dependency-free SQLite development backend.
FastAPI serves read-only metrics; Streamlit presents the selected processing run.
No camera frames, face crops or person bounding boxes are stored in this database.

## Local walkthrough

First generate the synthetic attention file using [Module 4](attention.md).
Initialize SQLite, import that file, and inspect the returned `run_id`:

```bash
PYTHONPATH=src python3 -m shelfsense init-db
PYTHONPATH=src python3 -m shelfsense ingest outputs/module4-attention.jsonl \
  --calibration configs/shelf.synthetic.json
PYTHONPATH=src python3 -m shelfsense report --run-id YOUR_RUN_ID
```

The default database is `outputs/shelfsense.db`. These commands use only the
Python standard library. The synthetic report contains two completed attention
episodes, 1.8 seconds total attention, 20/21 mapped track observations, and one
track in each funnel stage. The heatmap also totals 1.8 observed-gaze seconds
for this particular example; that equality is not guaranteed for other data.

For the web interface, install optional dependencies and start two terminals:

```bash
source .venv/bin/activate
pip install -e '.[analytics,test]'
uvicorn shelfsense.api:create_app --factory --host 127.0.0.1 --port 8000
```

```bash
source .venv/bin/activate
streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open the dashboard at `http://127.0.0.1:8501`. API documentation is available at
`http://127.0.0.1:8000/docs`. The dashboard uses `SHELFSENSE_API_URL`, defaulting
to `http://127.0.0.1:8000`. The API reads `SHELFSENSE_DATABASE_URL`, with the same
SQLite default as the CLI. Set variables in each terminal or export them before
launching. `.env` files are not loaded automatically; `.env.example` documents
the defaults. Run commands from the repository root for relative SQLite paths.

## PostgreSQL

Create a database and role using your local PostgreSQL setup, install the
analytics extra, and set a connection URL in the shell:

```bash
export SHELFSENSE_DATABASE_URL='postgresql://USER:PASSWORD@localhost:5432/shelfsense'
shelfsense init-db
shelfsense ingest outputs/module4-attention.jsonl --calibration configs/shelf.synthetic.json
uvicorn shelfsense.api:create_app --factory --host 127.0.0.1 --port 8000
```

Use your own role/credentials in place of the placeholders. Keep the URL out of
Git. The database owner must permit table creation for `init-db`; normal API
queries only read the tables. The current API has no authentication and the
documented launch binds locally. Containerized deployment is Module 6.

SQL uses bound parameters and explicit transactions, following the
[Psycopg connection API](https://www.psycopg.org/psycopg3/docs/basic/usage.html).
Initial schema tables are:

| Table | Stored content |
| --- | --- |
| `ss_runs` | Run ID, canonical content digest, calibration/zone/config metadata and import time |
| `ss_sessions` | Per-session counters, zone metrics and aggregate heatmap |
| `ss_events` | Original verified events with emission sequence |
| `ss_episodes` | Completed episodes, session/track/zone IDs, start/end times and duration |

Tables and indexes are created idempotently. This is the initial schema; there
is no automatic upgrade mechanism for future incompatible schema changes yet.
JSON payloads use TEXT for portability. Calibration training samples are not
stored; only the calibration ID, zone definitions and processing settings are.

## Import behavior

`ingest` requires the matching calibration JSON. It replays the exported shelf
observations through the temporal engine, verifying smoothed results, events,
settings and completion records. It rejects altered event durations, missing
session endings, unexpected records after endings, and mixed run configurations.
This validates consistency with the current temporal implementation, not the
correctness of the underlying gaze estimates or footage.

Validation happens before database writes. The run, sessions, events and episodes
are then inserted in one transaction. Importing identical content under the same
run ID returns `imported: false`; whitespace/key-order changes do not affect the
digest. Different content using that ID is rejected. Failed inserts roll back the
whole run. Reprocessing footage creates a different run ID, so all metric queries
require a run selection and never automatically sum alternative runs of a clip.

Frames are streamed during validation. Aggregates and derived events/episodes
are retained until the transaction; memory scales with event and track counts,
not video frame count. This is an offline ingestion path, not live streaming.

## Metric definitions

- **Episode count:** completed `ATTENTION_ENDED` events in a zone. Short
  unconfirmed looks do not count. This is not a count of purchases or products.
- **Total/average attention:** sum/mean of completed episode durations. Attention
  share divides a zone's total by the selected run/session's total across zones.
- **Unique tracks:** distinct `(session_id, track_id)` pairs with a completed
  episode in the zone. Reused IDs can merge people, and broken tracks can count
  one person multiple times. These are not verified unique viewers.
- **Coverage:** mapped smoothed track observations divided by all track
  observations. This is observation-weighted, not time-weighted. Empty frames
  contribute no track observations. A zero denominator produces zero coverage.
- **Funnel:** tracked IDs → IDs with at least one valid shelf observation → IDs
  with a completed attention episode. It does not measure store entry, physical
  shelf visits, facing direction, reaching or conversion.
- **Heatmap:** 12×8 normalized shelf bins containing observed-gaze seconds,
  including candidates that never become confirmed episodes. For consecutive
  valid observations of the same track within the gap limit, elapsed time is
  assigned to the earlier smoothed point. Unknowns, absent tracks and long gaps
  add no time. The final point receives no guessed extra frame duration.

Zone gaps and off-shelf predictions remain unknown. Zero-episode zones appear in
rankings with zero totals, averages and shares. Session filtering applies to all
metrics and event queries. Coordinates use x left-to-right and y top-to-bottom.

## API

| GET endpoint | Required query | Optional query |
| --- | --- | --- |
| `/health` | None | None |
| `/runs` | None | None |
| `/sessions` | `run_id` | None |
| `/analytics` | `run_id` | `session_id` |
| `/analytics/products` | `run_id` | `session_id` |
| `/analytics/heatmap` | `run_id` | `session_id` |
| `/analytics/funnel` | `run_id` | `session_id` |
| `/events` | `run_id` | `session_id`, `limit` (1–500), `offset` |

`products` currently means manually configured shelf zones, not recognized SKUs.
Event pagination follows emission order; effective timestamps can go backwards
when delayed events close earlier observations. Unknown runs/sessions return 404,
invalid queries return 422, and unavailable/uninitialized storage returns a
sanitized 503. There are no HTTP mutation endpoints; import uses the local CLI.

## Validation status

Core tests exercise the synthetic pipeline through actual SQLite transactions,
metrics, duplicate imports, conflicting runs, rollback, filtering, empty sessions
and rejection of tampered/incomplete exports. Optional tests use FastAPI
TestClient and Streamlit AppTest, and a real PostgreSQL integration test runs
when `SHELFSENSE_TEST_POSTGRES_URL` is configured. The analytics CI job installs
the extras and supplies PostgreSQL 16 so those tests can run without skips.

Host PyPI DNS access is blocked, but the permitted Docker builder can install
dependencies. Module 6 adds a hashed analytics lock and build stages for API,
Streamlit and PostgreSQL integration tests. See [deployment validation](deployment.md)
for the current scope. Check the complete PostgreSQL 16 Compose CI results after
pushing. Real inference and accuracy still need weights and consented footage.
