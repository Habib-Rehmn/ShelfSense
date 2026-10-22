# Retail Gaze validation

`evaluate-retail` compares Gaze-LLE with annotated head boxes against the existing
YOLO26n → ByteTrack → pose head localization → Gaze-LLE pipeline. Tracking sees
all decoded frames in order; gaze inference and scoring use only the selected
split. No models are trained or tuned by this command.

## Dataset

Download the videos, `samples.json` and `frames.json` from
[the Voxel51 export](https://huggingface.co/datasets/Voxel51/retail_gaze).
Our copy is under `data/retail_gaze`, revision
`08301f87f969d66676b2473ad1f7537513dee2fa`. Its download manifest records the
verified source file hashes. The export contains 42 videos and 3,922 labels.

The export's product masks are **not included**: `scene_mask.mask_path` points to
files on the publisher's computer. Fetch the referenced masks from the
[original Kaggle dataset](https://www.kaggle.com/datasets/dulanim/retailgaze):

```bash
PYTHONPATH=src python scripts/download_retail_masks.py data/retail_gaze \
  --sessions 1_060433 5_070017 9_073001
```

This downloads only missing PNG files and writes a SHA-256 manifest. Missing
masks do not prevent point scoring; they are excluded from region scoring and
counted in the report. Never execute or unpickle dataset files to use this loader.

### Incomplete video exports

For sessions `1_060433`, `5_070017` and `9_073001`, FFprobe decoded 112 frames,
but annotations extend to frame 113. Comparisons against original JPEGs at
indices 0, 4, 8 and 444 supported the documented indexing; the final JPEG at 448
was absent from each encoded sequence. The evaluation deliberately fails when
annotations extend past the available frames instead of silently dropping labels.

For these sessions, download all 113 original images and evaluate them directly:

```bash
python scripts/download_retail_frames.py data/retail_gaze/original_frames \
  --sessions 1_060433 5_070017 9_073001 --frame-count 113
```

The reader verifies a contiguous sequence `0.jpg, 4.jpg, ..., 448.jpg` and maps
its position to the annotation frame number. It does not modify downloaded
videos. Original images and their source/hash manifest stay under ignored `data/`.

## Run

In an environment with the `vision,gaze` extras installed:

```bash
TORCH_HOME="$PWD/models/torch-cache" python -m shelfsense evaluate-retail \
  data/retail_gaze --sessions 1_060433 5_070017 9_073001 --split test \
  --original-frames data/retail_gaze/original_frames \
  --weights models/yolo26n.pt --head-weights models/yolo26n-pose.pt \
  --gaze-weights models/gazelle_dinov2_vitb14_inout.pt \
  --device cpu --threads 2 --output-dir outputs/retail-gaze-validation
```

The output directory must be new. The first run may need Torch Hub downloads if
its cache is absent. With the previously prepared local `shelfsense:vision`
image, the diagnostic evaluation can also run without network access:

```bash
docker build -f Dockerfile.retail-eval --target report \
  --output type=local,dest=outputs/retail-gaze-validation .
```

That Dockerfile uses a local prepared image; it is not a standalone model download
recipe. Its companion ignore file admits the dataset inputs, including originals.

## Outputs and interpretation

- `report.json`: aggregate/per-session metrics, coverage, missing-prediction
  reasons, mask availability, timings and limitations.
- `predictions.jsonl`: one row per labeled frame, both modes, pixel coordinates,
  track association, automatic head IoU, confidence and errors.
- `previews/*.jpg`: green circle/box is ground truth; blue cross is gaze with
  annotated head input; red cross is gaze with automatic head input. Green tint
  shows the referenced target product region. Every scored frame gets a preview.
- `manifest.json`: model, annotation, video, used-mask and evaluation-source
  fingerprints, plus installed package versions.

FiftyOne frame numbers are one-based. The loader converts these to zero-based
video indices; it does **not** multiply them by four. Head boxes arrive as
normalized XYWH and are converted to XYXY. Gaze points are normalized XY.

Mean pixel error and normalized XY Euclidean error are reported only for accepted
predictions, alongside coverage. Normalized error is
`sqrt((dx/width)^2 + (dy/height)^2)`, not pixel error divided by image diagonal.
Region accuracy counts whether a predicted pixel lies inside the labeled target
mask. All-eligible accuracy includes abstentions as misses; accepted accuracy
excludes abstentions. A missing mask, absent annotation or ground-truth point
outside its mask is reported separately and excluded from region denominators.
There is no nearest-region snapping or inferred product-class label.

Automatic predictions are matched to the labeled shopper by the annotated head
center lying inside exactly one tracked person's upper 60%. The label is used
only for evaluation association, never as input to automatic gaze inference.
Ambiguous/missing person matches are abstentions. Track IDs are diagnostic only;
this dataset does not supply the identity labels needed for tracking metrics.

These three sessions (participants 1, 5 and 9) form a small diagnostic sample,
not a full benchmark. Source splits are frame-level, not participant-disjoint.
The videos use reconstructed 10 FPS timing, so the evaluation makes **no
attention-duration accuracy claim**. Shelf calibration, smoothing and database
analytics are outside this image-space evaluation.

## Initial measured baseline

Run on 2026-10-22 using pretrained Gaze-LLE ViT-B in/out, YOLO26n and
YOLO26n-pose, CPU with two Torch threads. All 339 original JPEGs were processed
for tracking; all 43 test labels in the three specified sessions were scored.
No weights, thresholds or annotations were changed to improve these results.

| Measure | Annotated head input | Automatic head input |
|---|---:|---:|
| Accepted predictions | 32 / 43 | 28 / 43 |
| Coverage | 74.4% | 65.1% |
| Mean pixel error, accepted predictions | 108.3 px | 96.2 px |
| Mean normalized XY error, accepted predictions | 0.1899 | 0.1693 |
| Target-region hits, all eligible labels | 0 / 31 | 0 / 31 |

The mean errors use different accepted subsets, so the smaller automatic-head
error does **not** establish that automatic heads are better. Images are 640×480.
Eleven annotated-head predictions were rejected as out-of-frame. Automatic
failures comprised ten missing heads, three out-of-frame predictions and two
non-unique/missing person matches.

Five labels have no mask annotation; seven labeled gaze points fall outside
their referenced foreground masks. Those twelve labels remain in point-error
and coverage calculations but are excluded from region accuracy. No referenced
mask was missing after the Kaggle download.

The run took approximately 102 seconds including gaze model initialization,
tracking, gaze inference, image decoding and preview writing (before final
manifest creation). This is a diagnostic processing time, not live-feed FPS:
gaze inference ran only on the 43 selected test frames, and both modes were run.

The results show that the current pretrained system is not ready for precise
product-region analytics in these scenes. Failure remains even with annotated
head boxes, so replacing the head detector alone would not resolve it. Inspect
more labeled scenes and audit the inconsistent masks before making a training
or model-selection decision. Preserve these sessions as the recorded baseline;
use a separate validation split for any tuning.

Artifacts are under `outputs/retail-gaze-validation/`: report, per-frame
predictions, manifests and 43 annotated JPEG previews. Example previews:
`previews/1_060433_0092.jpg`, `previews/5_070017_0098.jpg`, and
`previews/9_073001_0101.jpg`.

Validation: 97 unit tests discovered; 92 passed in the CPU vision image, with
five unrelated optional analytics/PostgreSQL checks skipped. The host passed 91
with six skips (also missing the optional OpenCV mask test). The real-data
inference run used `RUN --network=none` and cached pretrained weights.

## Annotated videos

Create dataset-label videos and three-panel comparison videos from the original
images and the saved predictions, without rerunning inference:

```bash
python -m shelfsense.retail_video data/retail_gaze \
  --original-frames data/retail_gaze/original_frames \
  --predictions outputs/retail-gaze-validation/predictions.jsonl \
  --sessions 1_060433 5_070017 9_073001 --fps 10 \
  --output-dir outputs/retail-videos-mpeg4
```

Or use the prepared local vision image:

```bash
docker build -f Dockerfile.retail-video --target videos \
  --output type=local,dest=outputs/retail-videos-mpeg4 .
```

Each session produces `*_labels.mp4` (green head box, gaze arrow and annotated
product mask) and `*_comparison.mp4` (labels, model with annotated heads, model
with automatic heads). The arrow starts at a heuristic position inside the head
box, not a measured eye position. Labels are explicitly distinguished from
predictions. The previous test-only evaluation provides predictions on 43 frames;
other frames say **Not evaluated**, with no stale predictions carried forward.
Absent model estimates show their unknown reason. Mask inconsistencies remain
visible in the annotation panel's status text.

Playback uses 10 FPS only for presentation. Each sequence has 113 frames and is
11.3 seconds long; this is not a claim about original recording duration.
OpenCV writes MPEG-4 Part 2 MP4 files. For browser-friendly H.264 copies:

```bash
mkdir -p outputs/retail-videos
for source in outputs/retail-videos-mpeg4/*.mp4; do
  ffmpeg -n -i "$source" -c:v libx264 -crf 20 -pix_fmt yuv420p \
    -movflags +faststart "outputs/retail-videos/$(basename "$source")"
done
```

The local `outputs/retail-videos/index.html` gallery links the six H.264 videos.
