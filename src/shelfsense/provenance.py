"""Portable experiment fingerprints without recording environment credentials."""

from hashlib import sha256
from importlib import metadata
import platform
from pathlib import Path
import subprocess


def file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest(paths=()):
    def git(*args):
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=False)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    status = git("status", "--porcelain")
    return {"schema_version": 1, "python": platform.python_version(),
            "system": platform.system(), "machine": platform.machine(),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]},
            "inputs": [{"name": Path(p).name, "sha256": file_digest(p)} for p in paths]}
