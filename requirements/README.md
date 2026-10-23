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

## CPU vision environment

`vision-cpu.lock` records the complete tested Linux x86_64 / Python 3.12 CPU
runtime, including the CPU-only PyTorch builds. Every wheel has an exact HTTPS
URL and SHA-256 hash, and installation enforces `--require-hashes`. The dependency-free package
continues to support Python 3.11; this lock is platform-specific.

Use `Dockerfile.vision` to install it, prepare verified models and pinned Hub
sources, and run offline inference. See [release setup](../docs/releasing.md).
The existing `Dockerfile.vision-test` remains a development smoke recipe with
version ranges; it is not the reproducible release image.

When changing the CPU environment, regenerate the lock in an isolated Python
3.12 Linux environment, retain installation provenance and rerun the full CPU
build. Update model source revisions/hashes only after reviewing upstream changes.
Do not reuse this CPU lock for CUDA or the legacy L2CS extra.

To regenerate from a reviewed installation report (not from untrusted metadata):

```bash
python scripts/lock_vision_report.py outputs/vision-release/vision-install.json \
  requirements/vision-cpu.lock
docker build -f Dockerfile.vision --target runtime -t shelfsense:vision .
```

For a version update, first resolve and test the intended versions in a separate
Linux/Python 3.12 environment with `pip install --report`, then use that report.
The lock generator allows only PyPI and PyTorch artifact hosts. Review the diff
and validate the resulting image before committing a new lock.
