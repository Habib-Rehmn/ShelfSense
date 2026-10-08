# Module 3: shelf calibration and zone mapping

This module uses a **supervised affine baseline**, not a reconstructed 3D gaze
ray. A homography turns labeled shelf pixels into normalized shelf coordinates.
A separate fitted model predicts those coordinates from face-center position
and gaze angles. It needs labeled calibration samples from the actual camera
setup. Four shelf corners alone cannot provide this mapping.

## Run the synthetic example

No OpenCV, model weights or network access is needed:

```bash
PYTHONPATH=src python3 -m shelfsense calibrate configs/shelf.synthetic.json \
  --output outputs/calibration.json
PYTHONPATH=src python3 -m shelfsense map-gaze examples/gaze.synthetic.jsonl \
  --calibration outputs/calibration.json --output outputs/targets.jsonl
```

The five records produce `zone_a`, `zone_b`, `zone_c`, unknown due to range, and
unknown due to missing gaze. All inputs are synthetic and demonstrate plumbing
only. The very small training residual is not an accuracy measurement.
Output files are exclusive: use a new path for subsequent runs.

## Collect your own calibration

1. Fix the camera and video resolution. Record `[width, height]` in `image_size`.
2. Label shelf corners in pixel coordinates, in order: top-left, top-right,
   bottom-right, bottom-left. Use a convex shelf quadrilateral.
3. Define rectangular zones with unique IDs and normalized
   `[x_min, y_min, x_max, y_max]` bounds. The example splits the shelf into thirds.
4. On consented footage, have participants look at known target points across
   the shelf. Match each labeled moment to its estimated `face_bbox`, `yaw_rad`
   and `pitch_rad`. Store the corresponding target pixel as `target_image`.
   Use coordinates from the original frame, with no unrecorded crop or resize.
5. Put these samples in a JSON document with the same fields as
   `configs/shelf.synthetic.json`. Use your own `shelf_id`. Store real participant
   samples under ignored `data/`, not in the public synthetic example.
6. Run `calibrate` to validate, fit and save the calibration. Keep independent
   participants/sessions for evaluation; do not include those labels in this file.

At least six samples are required, with variation in both gaze angles. Six is a
software minimum, not enough evidence for accuracy. Collect many independent
observations across the head positions, angles and shelf points of interest.
Collinear/repeated features can make fitting impossible; the command rejects
rank-deficient data rather than silently picking arbitrary coefficients.

## Use with YOLO26n and L2CS

```bash
shelfsense track data/shelf.mp4 --weights models/yolo26n.pt \
  --gaze-weights models/L2CSNet_gaze360.pkl \
  --calibration outputs/calibration.json --output outputs/shelf.jsonl
```

`--calibration` requires `--gaze-weights`. A frame-resolution mismatch is an
error, even on frames with no detected people. Camera movement or a different
crop also invalidates calibration, but cannot be detected from resolution alone.

Alternatively, `map-gaze` reprocesses stored schema-2 gaze JSONL without running
the models again. Every input frame must include `image_size`, session, frame
index, timestamp, tracks and matching gaze observations. Older schema-2 exports
without resolution must be regenerated or explicitly annotated with the known
original resolution; the replay command does not guess it. Timestamps and frame
indices must increase within each session. Malformed input reports a line number.
Like tracking output, a failed replay can leave partial output.

## Mapping and coverage

The fitted features are normalized face-center x/y and yaw/pitch in radians.
Constant face-center dimensions are dropped from the fit, but remain constrained
at prediction time. Other features are scaled to their observed training range;
QR least squares fits one affine output for each shelf coordinate.

An observation must stay within every training feature's minimum/maximum range
(apart from floating-point tolerance). A calibration made with a fixed head
center therefore accepts only that center. To handle moving shoppers, collect
independent calibration samples spanning their positions. The range check is
only a rectangular coverage gate: it does not guarantee that every combination
inside the range was observed or that a prediction is accurate. Face size/depth,
appearance and camera viewpoint can still change performance.

Only front-facing angles strictly between −π/2 and π/2 are accepted by this
baseline. This is a model-domain restriction, not a check of a 3D shelf ray.
No angular confidence or physical eye origin is inferred.

Zones cannot overlap. Shared boundaries belong to the zone on the right/bottom
because bounds are half-open; outer edges at 1 are included. Gaps are allowed
and produce unknown targets. Predictions outside the shelf are never clipped
into a product zone, except for numerical error within 1e-8 of an outer edge.

## Schema 3 output

Tracking and gaze fields are preserved. Added fields:

| Field | Meaning |
| --- | --- |
| `image_size` | Original `[width, height]` |
| `calibration` | Shelf ID, method, SHA-256 calibration ID, resolution and training fit RMSE |
| `shelf_targets` | One result per gaze observation, in the same order |

Each target has `track_id`, `status`, `reason`, `shelf_point`, and `zone_id`.
Estimated targets have a point and zone. Unknown reasons are `gaze_unavailable`,
`invalid_gaze`, `outside_calibration`, `off_shelf`, or `unmapped_zone`.
An off-shelf or unmapped prediction retains its point for diagnostics, with a
null zone. It must not count as product attention in later analytics.

Saved calibration contains the original labeled samples and settings; loading
refits deterministically. The ID hashes those inputs. The reported fit RMSE is
the root mean squared Euclidean error on **training samples**, in normalized
shelf coordinates. It is explicitly labeled as training error and must not be
reported as held-out performance. Real accuracy/coverage evaluation remains
outstanding. A metric 3D ray-plane mapping is deferred.
