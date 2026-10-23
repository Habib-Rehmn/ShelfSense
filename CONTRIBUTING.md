# Contributing to ShelfSense

ShelfSense is an experimental research baseline. Contributions that improve
reproducibility, evaluation quality and documented failure handling are welcome.
Please discuss major model or architecture changes in an issue before implementing them.

## Development setup

Use Python 3.11 or 3.12. From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
python -m shelfsense smoke --output-dir outputs/contributor-smoke
```

The core checks require no model downloads. Optional analytics, OpenCV and
PostgreSQL checks skip when their dependencies/services are absent; report skips
alongside passing tests. Use a new output directory when repeating the smoke.
For CPU model inference, follow [the reproducible vision setup](docs/releasing.md).
For analytics, follow [dependency locking](requirements/README.md).

## Pull requests

- Keep a change focused, explain the problem and resulting behavior, and list
  the relevant validation. Add a regression test for behavioral fixes.
- Preserve explicit unknown observations, session-scoped track IDs, frame
  alignment and complete-export checks. Never silently discard failed predictions.
- Keep optional ML imports lazy so the dependency-free core remains usable.
- For model changes, report dataset/split, coverage, metrics, model hashes and
  runtime conditions. Use a validation split for tuning; do not tune on test results.
- Update documentation when commands, defaults, output contracts or limitations change.
- Do not commit customer footage, datasets, weights, credentials, generated reports
  or local project outlines. Documentation media need source and license attribution.

Check `git diff --check` before submitting. CI runs the core suite on Python 3.11
and 3.12, analytics integration and Docker Compose checks. The manual CPU vision
workflow additionally runs pinned pretrained models without a private local image.

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md), not a
public bug report. Contributors must have permission to submit their code and media.
