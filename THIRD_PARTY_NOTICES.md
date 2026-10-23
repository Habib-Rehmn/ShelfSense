# Third-party components and media

ShelfSense's project license does not replace the licenses of dependencies,
model weights, datasets or third-party media. Preserve upstream notices when
redistributing their code or artifacts. The tables below identify major components;
the installed packages retain their own license metadata and notices.

## Models and model code

| Component | Source and terms | How used |
| --- | --- | --- |
| Ultralytics / YOLO26n / YOLO26n-pose | [AGPL-3.0 or separately obtained enterprise terms](https://www.ultralytics.com/license) | Optional detection/pose dependency and downloaded weights. |
| Gaze-LLE | [MIT code license](https://github.com/fkryan/gazelle/blob/main/LICENSE), copyright Fiona Ryan (2024) | Pinned source loaded through Torch Hub; decoder downloaded from the upstream release. |
| DINOv2 | [Apache-2.0 repository license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | Gaze-LLE backbone; pinned code and verified ViT-B weights in the CPU image. |
| L2CS-Net (legacy) | [Upstream repository and terms](https://github.com/Ahmednull/L2CS-Net) | Optional legacy adapter; not included in the locked CPU release image. |

The prepared CPU image retains upstream source archives' LICENSE/NOTICE files in
its Torch Hub cache. `scripts/prepare_vision.py` makes one documented source change
to Gaze-LLE's `gazelle/backbone.py`: it pins the nested DINOv2 Hub reference.
The prepared manifest records the modified file's SHA-256. Exact revisions, URLs
and original archive/checkpoint hashes live in `configs/vision-assets.json`.

PyTorch, torchvision, OpenCV, timm, NumPy, SciPy, scikit-learn and the other runtime
packages are installed dependencies, not relicensed ShelfSense source. Refer to
their distributed license files. Analytics dependencies are listed separately in
`requirements/analytics.lock`; CPU vision dependencies in `requirements/vision-cpu.lock`.

## Retail Gaze dataset and README media

The [original Kaggle dataset](https://www.kaggle.com/datasets/dulanim/retailgaze)
currently declares **CC0: Public Domain** in its
[dataset metadata](https://www.kaggle.com/api/v1/datasets/view/dulanim/retailgaze)
(checked 2026-10-22). This is the dataset publisher's declaration, not a license
inferred from the paper. The [Voxel51 export](https://huggingface.co/datasets/Voxel51/retail_gaze)
says that the paper does not explicitly state a license.

Dataset attribution: Shashimal Senarath, Primesh Pathirana, Dulani Meedeniya and
Sampath Jayarathna, “Retail Gaze: A Dataset for Gaze Estimation in Retail
Environments,” DASA 2022, DOI: 10.1109/DASA54658.2022.9765224.

`docs/assets/retail-gaze-annotated.mp4` and its GIF preview derive from session
`5_070017`, with ShelfSense's annotation overlays. Their underlying dataset media
are covered by the publisher's CC0 declaration, separately from the project code.
The overlays show dataset labels, not model predictions. The bulk dataset and
model weights are not committed. See `docs/assets/README.md` for media provenance.

A dataset copyright declaration does not establish permissions for every future
store-camera deployment. Contributors should use data they are entitled to share.
