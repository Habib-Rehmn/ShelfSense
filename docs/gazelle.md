# Gaze-LLE for store-camera video

The default gaze backend is **Gaze-LLE DINOv2 ViT-B with the in/out head**
(`gazelle_dinov2_vitb14_inout`). YOLO26n + ByteTrack still track people.
Gaze-LLE predicts a target in the scene image, not yaw/pitch or a physical 3D ray.
The shopper and target shelf must be visible in the same camera view.

## Install and download

Use a dedicated environment. On a CPU-only machine, install CPU PyTorch first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[vision,gaze]'
mkdir -p models data outputs
curl -L --fail -o models/yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
curl -L --fail -o models/yolo26n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt
curl -L --fail -o models/gazelle_dinov2_vitb14_inout.pt https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt
```

The gaze extra installs the runtime dependencies. On first model creation,
PyTorch Hub loads the [official Gaze-LLE code](https://github.com/fkryan/gazelle)
at revision `2d78f9f3bd2a5db360354954ec1a4c526ee8ab55`. This avoids an upstream
package-discovery issue with installing its Git source as a wheel. The upstream
code also downloads DINOv2 backbone code and weights through PyTorch Hub. The local
Gaze-LLE checkpoint contains only decoder weights. An offline run needs that
Hub cache populated first. The DINOv2 Hub source and runtime package ranges remain unpinned; this is not a complete
reproducibility lock; retain the tested environment and model hashes.

## Head localization and unknown observations

A second, small YOLO26n-pose model provides COCO facial landmarks. At least two
of nose/eyes/ears must have confidence >= 0.5. A padded square around their spread
estimates a head box. This is a heuristic localization baseline, not a dedicated
head detector. Rear views, low resolution and occlusion can yield no head.
We do not substitute the top of every person box or force a prediction.

Head centers must match exactly one person's upper 60% region, and that person
must have exactly one candidate head. Ambiguity and small heads become unknown.
Accepted heads share one Gaze-LLE scene forward pass per frame. The default
`--min-head-size` is 16 pixels; it is a gate, not a guarantee of usable detail.

The global heatmap maximum defines the target using heatmap cell centers in
original-frame pixel coordinates. An in-frame score below
`--in-frame-threshold` (default 0.5), invalid output or a flat heatmap becomes
unknown. This threshold has not been calibrated on store footage. Heatmap peak
values are not reported as calibrated accuracy/confidence. A target outside the
shelf remains unknown; the model's in-frame score is not an on-shelf score.

## Shelf setup and run

Copy `configs/shelf.image-target.example.json` to `data/shelf.json`. Replace its
example image size, shelf ID and four corners with your video's actual geometry.
Corners are pixel coordinates in TL, TR, BR, BL order. Zone bounds are normalized
shelf coordinates. This schema-2 configuration has no labeled angle samples.
It describes one planar shelf; moving the camera requires updating it.

```bash
shelfsense calibrate data/shelf.json --output outputs/shelf-validated.json
shelfsense track data/shelf.mp4 --weights models/yolo26n.pt \
  --gaze-weights models/gazelle_dinov2_vitb14_inout.pt \
  --head-weights models/yolo26n-pose.pt \
  --calibration outputs/shelf-validated.json --attention \
  --device cpu --output outputs/attention.jsonl
shelfsense init-db
shelfsense ingest outputs/attention.jsonl --calibration outputs/shelf-validated.json
```

Omit `--calibration` and `--attention` to export gaze targets before defining the
shelf. `map-gaze` can replay those exports with schema-2 shelf geometry later.
Start with `--max-frames 20` to check runtime. Every frame is processed; there is
no promise of live CCTV speed. Two YOLO passes plus Gaze-LLE add CPU work.

## Export compatibility

Frame schema 2/3/4 retain their existing meanings. New scene observations carry
`gaze_kind: image_target`, `head_bbox`, `target_image` (pixel x/y),
`in_frame_score`, status and reason. Frames record gaze model, threshold and
head-localization method. No angles are fabricated. Schema-3 shelf targets feed
the existing temporal engine, SQLite/PostgreSQL ingestion, API and dashboard.
Zone and point evaluation work; angular evaluation needs a direction-producing
model and is not applicable to Gaze-LLE.

Legacy L2CS remains available with the `l2cs` extra and explicit
`--gaze-backend l2cs`. Its schema-1 labeled calibration cannot be reused for
Gaze-LLE; a mismatch fails before inference. See [legacy setup](gaze.md).

## Validation

Core tests cover head association, unknowns, heatmap coordinates, geometry-only
mapping, replay, CLI selection and attention export ingestion into SQLite.
The opt-in pretrained smoke runs two supplied synthetic head boxes on a blank
frame and checks YOLO26n-pose on blank pixels:

```bash
docker build -f Dockerfile.vision-test --target gazelle-report \
  --output type=local,dest=outputs/gazelle-validation .
```

It exports package/model provenance and a single gaze-call duration. This does
not measure detection accuracy, gaze accuracy, full-pipeline FPS or sustained
consumer-CPU performance. Real shelf-zone accuracy still needs annotated store
footage. High-angle cameras and hidden heads may produce substantial unknowns.

The pretrained smoke passed on an Intel Core i5-1235U with Python 3.12.15,
PyTorch 2.14.1+cpu and two PyTorch threads. The measured call for two supplied
heads took 0.991 seconds. Model loading, downloads, video decoding, person
tracking and pose-based head localization are excluded from that timing.
This is one synthetic call, not a sustained real-time benchmark.
The decoder SHA-256 was
`451f76e5a1fa14429283d8612785d9aa94362f4754cb8a27d650482e3a2ac00e`;
the DINOv2 backbone SHA-256 was
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
