# ShelfSense

Single-camera shelf attention research project. **Six implementation modules:** person
tracking, pose-landmark head localization, Gaze-LLE image-target estimation,
shelf geometry mapping, temporal smoothing, attention events, database ingestion,
an analytics API, a dashboard, held-out evaluation tools and Docker deployment.
This is a research baseline; real-video accuracy remains unmeasured.
The detailed project outline is kept locally.

## Run the foundation

Run the complete synthetic workflow, including database ingestion and evaluation:

```bash
PYTHONPATH=src python3 -m shelfsense smoke --output-dir outputs/smoke-001
```

It writes analytics, raw/smoothed evaluation, replay timings and an environment
manifest. The output directory must be new. No neural models run in this check.

Python 3.11 or newer. No dependencies or downloads are needed for the demo:

```bash
PYTHONPATH=src python3 -m shelfsense demo --output outputs/demo.jsonl
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The demo writes 30 synthetic frames for one moving bounding box. It does not run
a detector or demonstrate tracking accuracy. Output files are never overwritten.

## Track a video

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[vision]'
shelfsense track data/shelf.mp4 --weights models/yolo26n.pt \
  --output outputs/tracks.jsonl --device cpu
```

Our baseline is **YOLO26n** (`yolo26n.pt`) for person detection, with ByteTrack
for tracking. YOLO26 requires Ultralytics 8.4.0 or newer; see the
[official model documentation](https://docs.ultralytics.com/models/yolo26/).
Supply the local checkpoint and a constant-frame-rate video.
Weights and footage are not bundled. The optional vision installation includes
PyTorch through Ultralytics and can be large. Adapter configuration follows the
[official tracking documentation](https://docs.ultralytics.com/modes/track/):
person class only, persistent tracking, explicit `bytetrack.yaml`.
Dependency ranges are declared; an exact tested vision lock is deferred until
the inference environment is validated.

Use `--device 0` for a supported GPU, `--max-frames 100` for a short run, or
`--fps 30` to override missing/incorrect FPS metadata. Timestamps are frame index
divided by FPS, starting at zero. Variable-frame-rate input is not supported.
The default detector threshold of 0.1 preserves low-confidence candidates for
ByteTrack association. Confidence is detector confidence, not ID certainty.

## Enable gaze estimation

Gaze-LLE ViT-B in/out is the default. Install the adapter and supply local weights:

```bash
pip install -e '.[vision,gaze]'
shelfsense track data/shelf.mp4 --weights models/yolo26n.pt \
  --gaze-weights models/gazelle_dinov2_vitb14_inout.pt \
  --head-weights models/yolo26n-pose.pt \
  --output outputs/gaze.jsonl --max-frames 100
```

See [Gaze-LLE setup](docs/gazelle.md) for downloads, first-run DINOv2 caching,
CPU setup and limitations. Missing and ambiguous heads remain unknown.
L2CS is now a legacy option requiring `--gaze-backend l2cs`.

## Calibrate shelf zones

For Gaze-LLE, copy `configs/shelf.image-target.example.json` to `data/shelf.json`
and enter your actual frame size, shelf corners and zones. No angle samples are
needed. Follow the [Gaze-LLE run instructions](docs/gazelle.md#shelf-setup-and-run).

The original angle-based synthetic calibration demo remains available:

```bash
PYTHONPATH=src python3 -m shelfsense calibrate configs/shelf.synthetic.json \
  --output outputs/calibration.json
PYTHONPATH=src python3 -m shelfsense map-gaze examples/gaze.synthetic.jsonl \
  --calibration outputs/calibration.json --output outputs/targets.jsonl
```

For legacy L2CS video, collect labeled gaze/target samples for your fixed camera and
pass the resulting `--calibration` file to `track` alongside `--gaze-weights`.
This is a supervised mapping baseline; shelf corners alone cannot locate gaze.
See [calibration setup and limitations](docs/calibration.md).

## Derive attention events

Add `--attention` to a calibrated tracking run, or replay stored shelf targets:

```bash
PYTHONPATH=src python3 -m shelfsense attention outputs/targets.jsonl \
  --calibration outputs/calibration.json --output outputs/attention.jsonl
```

This smooths coordinates using media time and confirms stable attention episodes
after a configurable duration. Unknown gaze and observation gaps break episodes.
The short Module 3 demo does not contain long enough looks to confirm attention;
use the longer [Module 4 synthetic example](docs/attention.md) to see completed
episodes. See that guide for thresholds, event fields and exact duration rules.

## Store and explore analytics

After generating the longer Module 4 example:

```bash
PYTHONPATH=src python3 -m shelfsense init-db
PYTHONPATH=src python3 -m shelfsense ingest outputs/module4-attention.jsonl \
  --calibration configs/shelf.synthetic.json
PYTHONPATH=src python3 -m shelfsense report --run-id YOUR_RUN_ID
```

Imports are atomic and idempotent. SQLite works without additional dependencies;
PostgreSQL, FastAPI and Streamlit are available through the analytics extra:

```bash
pip install -e '.[analytics,test]'
uvicorn shelfsense.api:create_app --factory --host 127.0.0.1 --port 8000
# In a second terminal:
streamlit run dashboard/app.py --server.address 127.0.0.1
```

The dashboard shows zone rankings, observed-gaze heatmaps, coverage and an
attention funnel for one selected processing run. See [analytics setup and
metric definitions](docs/analytics.md) for PostgreSQL configuration, endpoints
and validation limits.

## Evaluate and deploy

```bash
PYTHONPATH=src python3 -m shelfsense evaluate outputs/attention.jsonl \
  --labels data/heldout-labels.json --calibration data/calibration.json \
  --kind smoothed --output outputs/evaluation.json
```

Evaluation reports coverage, all-label accuracy, macro F1, confusion matrices,
and point/angular errors when suitable labels exist. Missing predictions count
as abstentions. See [evaluation and experiment records](docs/evaluation.md).

Docker Compose starts PostgreSQL, initialization, the API and dashboard. Set
`SHELFSENSE_POSTGRES_PASSWORD` in a local `.env`, then run:

```bash
docker compose up --build --detach --wait api dashboard
```

See [deployment instructions](docs/deployment.md) for import commands, storage,
runtime validation and reproducibility limits.

## Output contract

Each JSONL line contains `schema_version`, `session_id`, `frame_index`,
`timestamp_s`, and `tracks`. Each track has `track_id`, `bbox` (pixel coordinates
`[x1, y1, x2, y2]`) and `confidence`. Empty frames are preserved with `tracks: []`.
Pair session ID with track ID when storing shopper observations.
Gaze-enabled runs add `image_size` and `gaze` in schema 2. Calibrated runs use
schema 3, adding calibration metadata and `shelf_targets` with explicit unknowns.
Attention-enabled output uses schema 4 and includes smoothed observations and
events. It also appends session-end records; consumers must check `record_type`.
Output is streamed with bounded pipeline memory. On an error, output may contain
a partial run and must not be treated as complete. OpenCV can report decoding
failure like end-of-file; successful completion does not prove clip integrity.

## Verification and limits

Unit tests cover pipeline contracts, CLI behavior, adapter arguments and video
resource cleanup using fakes. The synthetic CLI is also runnable end to end.
Pretrained YOLO26n CPU inference and OpenCV video I/O pass the optional Docker
smoke on ten blank synthetic frames. This checks the actual tracking CLI, frame
counts, timestamps and session identity. Person detection, ByteTrack identity
continuity and real-video performance still need a consented clip. No accuracy/FPS
claims yet.
Gaze adapter tests validate preprocessing and angle conversion using fakes;
pretrained-model inference is not yet validated. Calibration/geometry and replay
tests run without ML dependencies, including the synthetic mapping example.
Real shelf-zone accuracy remains unmeasured. Temporal tests cover gaps, zone
changes, session endings and event durations; the episode thresholds remain
unvalidated on real footage. SQLite ingestion and analytics are integration-tested.
FastAPI, Streamlit and PostgreSQL tests require optional dependencies/services.
Docker test stages install the hashed analytics lock and run integration checks;
CI also validates PostgreSQL 16 and full Compose startup. Host-only tests skip
unavailable extras. See [validation details](docs/deployment.md).
Real-data evaluation and annotated store video remain. See the
[Gaze-LLE validation procedure](docs/gazelle.md#validation).

## Commit workflow

One reviewed module per commit, approximately six commits total. After checks,
commit the module and ask the owner to push before beginning the next one.
Do not commit footage, model weights, credentials or generated outputs.
