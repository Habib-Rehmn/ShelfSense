# Dependency locking

`analytics.lock` contains resolved versions and package hashes for the analytics
and test extras, generated with uv 0.9.4 on Linux/Python 3.12. Installation and the
API/dashboard tests are verified in the Docker build environment. The core
dependency-free package also supports Python 3.11; this lock targets Python 3.12.

To regenerate and validate on a network-enabled machine:

```bash
uv pip compile pyproject.toml --extra analytics --extra test --generate-hashes \
  --output-file requirements/analytics.lock
uv pip sync requirements/analytics.lock
pip install --no-deps -e .
python -m unittest discover -s tests -v
```

Use a dedicated Python 3.12 virtual environment for this command. Set
`SHELFSENSE_TEST_POSTGRES_URL` to a test database to run the PostgreSQL check.
The Dockerfile automatically uses this lock if present; otherwise it resolves
the declared analytics ranges and saves the installed versions inside the image.
Keep the lock present for release builds. Commit updates only after the
dependency-backed tests pass. You can also regenerate via the Docker build:

```bash
docker build --target dependency-lock --output type=local,dest=requirements .
docker build --target test -t shelfsense:test .
docker build --target postgres-test -t shelfsense:postgres-test .
```

The PostgreSQL test stage uses Debian's PostgreSQL 15; Compose and CI additionally
target PostgreSQL 16. Test-only database binaries are excluded from the runtime image.

The vision/gaze environment needs its own validated lock, model file hashes,
and an exact L2CS Git revision. It is not part of the analytics image.
