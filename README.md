# ShelfSense

Single-camera shelf attention research project. **Module 1 of 6:** person
tracking. Gaze inference and analytics are planned, not yet implemented.
See [project.md](project.md) for the deliverables and commit checkpoints.

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
shelfsense track data/shelf.mp4 --weights models/yolo11n.pt \
  --output outputs/tracks.jsonl --device cpu
```

Supply a local COCO person-detection model and a constant-frame-rate video.
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

## Output contract

Each JSONL line contains `schema_version`, `session_id`, `frame_index`,
`timestamp_s`, and `tracks`. Each track has `track_id`, `bbox` (pixel coordinates
`[x1, y1, x2, y2]`) and `confidence`. Empty frames are preserved with `tracks: []`.
Pair session ID with track ID when storing shopper observations.
Output is streamed with bounded pipeline memory. On an error, output may contain
a partial run and must not be treated as complete. OpenCV can report decoding
failure like end-of-file; successful completion does not prove clip integrity.

## Verification and limits

Unit tests cover pipeline contracts, CLI behavior, adapter arguments and video
resource cleanup using fakes. The synthetic CLI is also runnable end to end.
Real YOLO/ByteTrack inference needs a separate smoke test with installed vision
dependencies, model weights and a consented clip. No accuracy/FPS claims yet.
The current module exports tracking observations; annotated video, gaze,
fixations, database and dashboard are not implemented.

## Commit workflow

One reviewed module per commit, approximately six commits total. After checks,
commit the module and ask the owner to push before beginning the next one.
Do not commit footage, model weights, credentials or generated outputs.
