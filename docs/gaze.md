# Module 2: faces and gaze

The tracking CLI now accepts `--gaze-weights`, `--gaze-arch` and
`--min-face-size` (default 32 pixels). Without gaze weights, schema 1 tracking
output remains unchanged. With weights, each frame uses schema 2 and adds a
`gaze` list with exactly one observation per track, in track order.

## Setup

Use Python 3.11+ and install `pip install -e '.[vision,gaze]'`. This extra installs
the official [L2CS-Net repository](https://github.com/Ahmednull/L2CS-Net), including
its transitive dependencies. The source dependency currently follows `main`;
this is not a reproducibility lock. Record the installed source revision and
dependency versions when running experiments. The upstream package imports its
RetinaFace dependency even though ShelfSense uses a separate OpenCV detector.

Download a Gaze360 checkpoint using the pretrained-model link in the upstream
README, store it under `models/`, and select the matching architecture (default
ResNet50). We load the local state dictionary with `weights_only=True`.
No model checkpoint or footage is bundled with ShelfSense.

```bash
shelfsense track data/shelf.mp4 --weights models/yolo26n.pt \
  --gaze-weights models/L2CSNet_gaze360.pkl --gaze-arch ResNet50 \
  --min-face-size 32 --output outputs/gaze.jsonl --device cpu
```

`--device 0` maps to `cuda:0` in the gaze adapter. Model/preprocessing imports
are lazy, so core tests and the existing demo do not require ML packages.

## Localization and association

OpenCV's bundled frontal-face Haar cascade finds face boxes in the full frame.
This is a replaceable baseline, not a robust profile/head detector. It exposes
no calibrated confidence. The analyzer clips boxes to frame bounds, then matches
each face center to the upper 60% of a tracked person's box. That gate is a
heuristic, not anatomical head localization. It requires one face per person
and one person per face; overlapping candidates are marked ambiguous.

Only matched faces at least the configured size in both dimensions are passed
to L2CS. The crop uses full-frame pixel coordinates. Upstream preprocessing
converts BGR to RGB, resizes to 224×224, then applies L2CS's supplied input
transform. Inference runs with gradients disabled.

## Angles and axes

The upstream [pipeline](https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/pipeline.py)
calls the first model output `pitch`. Its
[renderer](https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/vis.py) uses that
first output for horizontal displacement. ShelfSense maps the first head to
`yaw_rad` and the second to `pitch_rad` to agree with that rendering convention.
The upstream [model definition](https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/model.py)
also returns its yaw head first and pitch head second.
Both outputs use the expectation of 90 softmax bins, scaled by 4 degrees and
offset by −180 degrees, then converted to radians.

Camera axes are x right, y down, z away from camera. A frontal direction is
`(0, 0, -1)`; positive yaw points image-left and positive pitch image-up:

```text
direction_camera = (-cos(pitch) sin(yaw), -sin(pitch), -cos(pitch) cos(yaw))
```

This is a declared adapter convention, pending empirical sign checks with the
chosen checkpoint. It is not a calibrated shelf ray: no 3D eye origin or shelf
plane has been recovered. Face-box center is not a metric eye position.

## Observation fields

| Field | Meaning |
| --- | --- |
| `track_id` | Shopper track within this session |
| `status` | `estimated` or `unknown` |
| `reason` | Null when estimated; otherwise `no_face`, `ambiguous_face`, `face_too_small`, or `invalid_angles` |
| `face_bbox` | Clipped `[x1,y1,x2,y2]` pixels when a unique face exists; otherwise null |
| `yaw_rad`, `pitch_rad` | Canonical radians, or null |
| `direction_camera` | Unit direction in the convention above, or null |

These are eye-gaze model estimates, not head-pose estimates. No fabricated gaze
confidence is emitted. Face size and association checks reject some failures,
but cannot detect every blurred, poorly lit or occluded face. Hardware errors,
incompatible model weights and dependency errors abort the run rather than
silently being reported as missing faces.

## Validation status

Core tests cover association ambiguity, missing/small faces, clipping, crop
coordinates, invalid angles, axes, serialization and adapter calls. These tests
use fakes for neural inference. A real checkpoint/video smoke test and empirical
accuracy evaluation remain outstanding; no gaze accuracy is claimed.
Installing OpenCV in the development sandbox was attempted but blocked by PyPI
DNS/network access, so the real cascade smoke test also remains outstanding.
