# Preparing a research release

## Fresh-clone CPU build

Requires Docker with BuildKit, Linux x86_64 execution and sufficient disk space
for PyTorch, cached downloads and image layers (allow several GB). The initial
build needs internet access. It does not need local weights, datasets, previous
Docker images or the ignored `outputs/prepare-vision.Dockerfile`.

```bash
git clone https://github.com/Habib-Rehmn/ShelfSense.git
cd ShelfSense
docker build -f Dockerfile.vision --target runtime -t shelfsense:vision .
docker build -f Dockerfile.vision --target report \
  --output type=local,dest=outputs/vision-release .
```

The second command reuses the build cache to export validation/provenance reports.
The build runs unit tests, real YOLO26n inference and Gaze-LLE inference with
synthetic pixels and supplied head boxes under `RUN --network=none`. This confirms
execution, not real-world accuracy. The existing Retail Gaze evaluation/video
Docker recipes can then use the newly built `shelfsense:vision` tag.

Run a local video with no container network and read-only input mounts:

```bash
mkdir -p outputs
docker run --rm --network none \
  -v "$PWD/data:/data:ro" -v "$PWD/outputs:/outputs" \
  shelfsense:vision track /data/shelf.mp4 \
  --weights /models/yolo26n.pt --head-weights /models/yolo26n-pose.pt \
  --gaze-weights /models/gazelle_dinov2_vitb14_inout.pt \
  --device cpu --max-frames 20 --output /outputs/gaze.jsonl
```

Outputs must be new. Docker runs as root in this research vision image, so exported
files may need ownership adjustment on Linux. It is not a hosted inference service.

### What is pinned

- Python base image digest and Python 3.12 / Linux x86_64 CPU target.
- Runtime package versions, exact wheel URLs and SHA-256 hashes in
  `requirements/vision-cpu.lock` (including setuptools).
- Gaze-LLE and DINOv2 source revisions and archive SHA-256 values.
- YOLO26n, YOLO26n-pose, Gaze-LLE decoder and DINOv2 backbone checkpoint SHA-256 values.

`scripts/prepare_vision.py` downloads only the URLs in `configs/vision-assets.json`,
verifies hashes and preserves upstream license files. It changes the single nested
DINOv2 Hub call in the private Gaze-LLE cache from a moving branch to the pinned
revision. The prepared manifest records this patch. It does not modify global
Torch functions or vendor model code into ShelfSense's package.

This pinned source preparation is specific to the Docker recipe. Ordinary
`pip install -e '.[vision,gaze]'` uses the declared version ranges and upstream
Hub behavior; use the prepared image for the release-tested environment.
System packages still come from Debian repositories, and deterministic predictions
across hardware are not guaranteed. The legacy L2CS extra is outside this CPU lock.

## Release checks

1. Resolve the project license and confirm third-party notices and demo attribution.
2. From a clean checkout, run the documented core smoke and CPU build. Review all
   skipped tests. Confirm the remote core/analytics/container CI jobs pass.
3. Run Gitleaks over all available Git history and the working tree. Rotate any
   exposed credentials before removing them; history edits alone do not revoke secrets.
4. Confirm `.env`, weights, datasets, `project.md` and generated reports remain
   untracked. Only the selected attributed documentation media are exceptions.
5. Enable GitHub private vulnerability reporting and appropriate branch protection
   in repository settings. Check the private reporting link in `SECURITY.md`.
6. Publish a research release with limitations, metrics and provenance. Do not
   describe the annotation demo as model accuracy or claim validated store deployment.

The agent does not change repository visibility or publish a release automatically.
The owner controls these GitHub settings and the final publication.

## Local readiness audit (2026-10-22)

Gitleaks 8.30.1 was downloaded from the upstream release and checked against its
published SHA-256. A scan using `git . --log-opts=--all --redact=100` examined 13
locally available commits through `d254441` and found no leaks. A working-tree
scan also found no leaks. This is scanner evidence, not a guarantee that all
secrets or remote-only branches have been inspected. Redacted reports are kept
under ignored `outputs/` rather than committed.

The GitHub secret-scan workflow repeats the full-history scan on pushes and pull
requests. CPU inference validation is a manual workflow because it downloads
large weights; it needs no Retail Gaze dataset or private infrastructure.

The fresh-base CPU build passed locally: 100 tests discovered, 95 passed and five
optional analytics/PostgreSQL checks skipped, followed by pretrained YOLO26n and
Gaze-LLE inference with networking disabled. The host core suite passed 94 tests
with six skips, and the synthetic end-to-end smoke passed. These checks do not
replace the existing poor Retail Gaze accuracy results or imply production readiness.

The project license selection is still awaiting the repository owner's decision;
adding readiness documentation alone does not grant an open-source license.
