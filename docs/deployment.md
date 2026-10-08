# Local deployment and release checks

Docker Compose runs PostgreSQL 16, table initialization, FastAPI, and Streamlit.
The analytics image runs as a non-root user and excludes footage, model weights,
credentials and the local project outline. It does not include GPU/vision models.

## Start the services

Create a local `.env` with `SHELFSENSE_POSTGRES_PASSWORD` set to your chosen
password (see `.env.example`). Compose reads this file automatically. Keep it
ignored by Git. Then, from the repository root:

```bash
mkdir -p outputs data
docker compose config --quiet
docker compose up --build --detach --wait api dashboard
```

The dependency chain waits for database health, successful schema initialization,
and API health before starting the dashboard. Host ports bind to localhost:

- Dashboard: `http://127.0.0.1:8501`
- API and documentation: `http://127.0.0.1:8000/docs`
- PostgreSQL stays on the Compose network, with no published host port.

The database URL omits its password; Psycopg/libpq receives it through
[`PGPASSWORD`](https://www.postgresql.org/docs/current/libpq-envars.html). The
dashboard only gets the API URL. This is a local research deployment, with no
authentication or public hosting configuration.

## Import a run

Generate an export on the host first:

```bash
PYTHONPATH=src python3 -m shelfsense smoke --output-dir outputs/deployment-smoke
docker compose run --rm cli python -m shelfsense ingest \
  /imports/deployment-smoke/attention.jsonl --calibration configs/shelf.synthetic.json
```

The CLI service mounts host `outputs/` read-only at `/imports`, `configs/` at
`/app/configs`, and `data/` at `/data`. Use `/data/calibration.json` for a real
calibration kept outside Git. Container-generated output can go to `/app/outputs`,
a named artifact volume. Refresh the dashboard and select the imported run.
Reimporting that run is safe and returns `imported: false`.

View logs and stop services while retaining the database:

```bash
docker compose logs api dashboard
docker compose down
```

The `postgres-data` volume persists across restarts. Changing the password in
`.env` does not change an existing database role's password; update the role
through PostgreSQL if you rotate it. This module creates the initial schema and
does not migrate future incompatible versions automatically.

## Reproducibility boundaries

The Python base image is pinned by digest. A resolved, hashed analytics/test lock
is committed and has been installed successfully in the Docker build environment.
See [dependency locking](../requirements/README.md) for update and verification
commands. When `requirements/analytics.lock` exists, the Dockerfile installs its
hashed versions; keep it present for release builds. Without it the build falls
back to declared ranges. The PostgreSQL 16 image tag can still move, and the
Python package build toolchain uses isolated build dependencies.
The image always saves its installed package snapshot at
`/app/installed-packages.txt`.

For a reproducible release, retain the validated lock, pin the PostgreSQL image
digest and record the built application image digest, and retain dataset/model hashes
and experiment manifests. The vision/gaze environment requires a separate lock
and an exact L2CS source revision. Do not treat the current floating Git `main`
dependency as a pinned model environment.

## CI and local verification

The workflow runs three independent checks:

1. Core tests and the synthetic smoke command on Python 3.11 and 3.12, uploading
   the synthetic report and input/environment manifest.
2. Dependency-backed API/dashboard tests plus a real PostgreSQL service test.
3. Compose build/startup, synthetic import into PostgreSQL, API metric assertions
   and dashboard health. Its volumes are cleaned only on the isolated CI runner.

Locally, Compose configuration validation and the permitted Docker build succeed.
The build's `test` stage runs API TestClient and Streamlit AppTest checks as well
as the full core suite and synthetic smoke. The separate `postgres-test` stage
starts a temporary PostgreSQL 15 instance and runs the database integration test.
Host Python package downloads and direct Docker Compose runtime access remain
restricted. The complete multi-service PostgreSQL 16 deployment must still pass
the configured CI job or a run on your machine; a successful build is not proof
that Compose orchestration has run.
