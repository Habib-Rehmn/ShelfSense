# ShelfSense

Single-camera shelf attention research project. **Modules 1–4 of 6:** person
tracking, frontal-face localization, an L2CS-Net gaze adapter, and supervised
shelf-zone calibration, temporal smoothing and attention events. Database,
API analytics and dashboard are planned.
The detailed project outline is kept locally.

## Run the foundation

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

Install the optional adapter and supply local L2CS weights:

```bash
pip install -e '.[vision,gaze]'
shelfsense track data/shelf.mp4 --weights models/yolo26n.pt \
  --gaze-weights models/L2CSNet_gaze360.pkl \
  --output outputs/gaze.jsonl --max-frames 100
```

See [gaze setup and coordinate conventions](docs/gaze.md) for model sources,
limitations and the output schema. The OpenCV face detector is a frontal-face
baseline; missing and ambiguous observations remain unknown.

## Calibrate shelf zones

Try the complete calibration and mapping path with synthetic inputs:

```bash
PYTHONPATH=src python3 -m shelfsense calibrate configs/shelf.synthetic.json \
  --output outputs/calibration.json
PYTHONPATH=src python3 -m shelfsense map-gaze examples/gaze.synthetic.jsonl \
  --calibration outputs/calibration.json --output outputs/targets.jsonl
```

For real video, collect labeled gaze/target samples for your fixed camera and
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
Real YOLO/ByteTrack inference needs a separate smoke test with installed vision
dependencies, model weights and a consented clip. No accuracy/FPS claims yet.
Gaze adapter tests validate preprocessing and angle conversion using fakes;
pretrained-model inference is not yet validated. Calibration/geometry and replay
tests run without ML dependencies, including the synthetic mapping example.
Real shelf-zone accuracy remains unmeasured. Temporal tests cover gaps, zone
changes, session endings and event durations; the episode thresholds remain
unvalidated on real footage. Annotated video, database, API analytics and
dashboard are not implemented.

## Commit workflow

One reviewed module per commit, approximately six commits total. After checks,
commit the module and ask the owner to push before beginning the next one.
Do not commit footage, model weights, credentials or generated outputs.
