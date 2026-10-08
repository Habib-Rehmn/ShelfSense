# Module 6: evaluation and experiment records

Evaluation compares predictions to independently supplied labels. It does not
create ground truth from model output. The bundled labels are deliberately
synthetic and test the software contract, not model accuracy.

## One-command synthetic smoke test

From the repository root, with Python 3.11+:

```bash
PYTHONPATH=src python3 -m shelfsense smoke --output-dir outputs/smoke-001
```

The output directory must be new. The command runs calibration mapping, temporal
events, validated SQLite ingestion, duplicate-import checks, analytics, and both
raw and smoothed evaluation. It writes:

| File | Content |
| --- | --- |
| `targets.jsonl` | Schema-3 mapped shelf observations |
| `attention.jsonl` | Schema-4 observations/events and session completion |
| `analytics.db` | Local SQLite analytics database |
| `report.json` | Synthetic analytics, evaluation and replay timings |
| `manifest.json` | Input hashes, Python/platform, installed package versions and Git state |

Expected contract results: 23 frames, two attention episodes totaling 1.8 seconds,
and 20 accepted predictions among 21 labeled track observations. The rejected
observation is included in the accuracy denominator. Zone B has no support in
this synthetic clip, which affects the explicitly defined macro F1 below.

Replay FPS measures only mapping and temporal replay from synthetic gaze, not
YOLO, L2CS, video decoding, database throughput or dashboard performance. Timings
vary with hardware and are not real-time inference claims.

## Label format

Use `examples/labels.synthetic.json` as a format example. A label document has:

```json
{
  "schema_version": 1,
  "dataset_id": "shelf-camera-01-heldout",
  "split": "test",
  "synthetic": false,
  "calibration_session_ids": ["calibration-session-01"],
  "samples": [
    {
      "session_id": "heldout-session-01",
      "frame_index": 12,
      "track_id": 1,
      "zone_id": "zone_a",
      "shelf_point": [0.2, 0.5],
      "conditions": {"camera_id": "camera-01", "distance_m": 2, "lighting": "bright"}
    }
  ]
}
```

`shelf_point`, `direction_camera` and `conditions` are optional. Directions must
be nonzero finite three-vectors in the documented camera coordinate convention.
Do not infer angular ground truth from a 2D target pixel alone. Conditions support
`camera_id`, `distance_m`, `camera_angle_deg` and `lighting`, each producing a
separate metric breakdown. Missing condition values are excluded from that
breakdown. Values are grouped by their string representation.

Keys are `(session_id, frame_index, track_id)` and must be unique. Annotations
must refer to the exported tracker IDs; this evaluator does not independently
match labeled people to detections or measure identity switches. Annotate target
points independently of the model. Only on-shelf labeled targets are supported
by the current zone metric; off-shelf rejection needs a separate labeled task.

`split` must be `validation` or `test`. Declare the sessions used for calibration;
the evaluator rejects overlap with labeled evaluation sessions. This check
trusts the declared IDs: the calibration file does not itself record participant
or session provenance. Keep participants and sessions separate before collecting
or fitting data, and maintain your dataset manifest outside Git. Different IDs
alone do not prove that no frames or participants leaked between splits.

## Run an evaluation

```bash
PYTHONPATH=src python3 -m shelfsense evaluate outputs/attention.jsonl \
  --labels data/heldout-labels.json --calibration data/calibration.json \
  --kind raw --output outputs/raw-evaluation.json
PYTHONPATH=src python3 -m shelfsense evaluate outputs/attention.jsonl \
  --labels data/heldout-labels.json --calibration data/calibration.json \
  --kind smoothed --output outputs/smoothed-evaluation.json
```

Raw evaluation reads `shelf_targets` from schema 3 or 4. Smoothed evaluation
reads `attention` from schema 4, including unconfirmed candidates. Schema-4 files
must include session endings. One file must use one schema, calibration and
processing run. Neither output overwrites an existing file.

## Exact metric definitions

- **Coverage:** accepted zone predictions / all labeled samples. Unknown outputs
  and missing tracks/frames are abstentions. Missing predictions are also reported
  separately; extra unlabeled predictions are counted but not scored.
- **All-label accuracy:** correctly predicted zones / all labeled samples.
  An abstention is incorrect for this on-shelf target task.
- **Accepted accuracy:** correctly predicted zones / accepted predictions, or
  null if none are accepted. Always read this alongside coverage.
- **Precision/recall/F1:** per configured zone, with zero for an undefined
  denominator. Unknown predictions count as false negatives for the true zone.
- **Macro F1:** unweighted mean across every configured zone, including zones
  with zero label support. Per-zone support is included so absent classes are
  visible. Compare methods using identical labels and zone definitions.
- **Confusion matrix:** rows are true zones; columns are predicted zones plus a
  final null/abstention column. This cannot collide with a real zone name.
- **Shelf-point error:** mean Euclidean error in normalized shelf coordinates,
  only when a point label and an accepted prediction both exist. Its sample count
  is separate. It is not a metric distance in centimeters.
- **Angular error:** mean angular distance in degrees where both direction labels
  and model vectors exist. Vectors are normalized before comparison. This uses
  raw gaze vectors for both raw/smoothed zone evaluations because the EMA smooths
  shelf coordinates, not gaze angles. Angular sample count is reported separately.

An unavailable point/angular metric is null, never a fabricated zero. The tool
does not compute statistical confidence intervals, event matching/duration error,
identity-switch metrics or off-shelf detection metrics yet. Synthetic duration
tests check algorithm bookkeeping, not real attention accuracy.

## Experiment procedure

1. Reserve independent participants/sessions for validation and test before
   calibration. Validate prompted looks where feasible rather than assuming
   participants complied with instructions.
2. Calibrate and tune thresholds on training/validation data only. Freeze camera
   setup, weights, calibration and temporal settings before evaluating test data.
3. Run the same labeled clips with raw and smoothed predictions. Inspect coverage,
   class support and error breakdowns by distance, angle and lighting.
4. Record input fingerprints and runtime versions:

```bash
PYTHONPATH=src python3 -m shelfsense manifest data/heldout-labels.json \
  data/calibration.json models/yolo26n.pt models/L2CSNet_gaze360.pkl \
  --output outputs/experiment-manifest.json
```

The manifest stores basenames and SHA-256 hashes, not file content or environment
credentials. It marks a dirty Git checkout. Package versions are an environment
record, not a resolved dependency lock or GPU-hardware inventory; record device
model, drivers and inference command alongside it. Keep actual model timing
separate from replay timing, including warmup and whether decoding is included.
No real-world scores are claimed until these experiments run on labeled footage.
