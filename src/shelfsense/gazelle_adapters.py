"""Optional Gaze-LLE and pose-landmark head localization, with lazy imports."""

from math import isfinite

from .gazelle import decode_heatmap

GAZELLE_REVISION = "2d78f9f3bd2a5db360354954ec1a4c526ee8ab55"


class PoseHeadDetector:
    """Estimate head boxes from visible COCO facial landmarks, not body boxes.

    This is a localization baseline; absent landmarks (including many rear
    views) deliberately produce no head rather than an invented crop.
    """

    def __init__(self, weights, device="cpu", confidence=.25, keypoint_confidence=.5):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if self.model.task != "pose":
            raise ValueError("Head localization requires COCO pose weights, e.g. yolo26n-pose.pt")
        self.device, self.confidence = device, confidence
        self.keypoint_confidence = keypoint_confidence

    def detect(self, image):
        result = self.model.predict(image, device=self.device, conf=self.confidence, verbose=False)[0]
        if result.keypoints is None or result.keypoints.conf is None:
            return []
        heads = []
        for points, scores in zip(result.keypoints.xy.cpu().tolist(),
                                  result.keypoints.conf.cpu().tolist(), strict=True):
            visible = [(x, y) for (x, y), c in zip(points[:5], scores[:5], strict=True)
                       if c >= self.keypoint_confidence and isfinite(x) and isfinite(y)]
            if len(visible) < 2:
                continue
            xs, ys = zip(*visible)
            span = max(max(xs)-min(xs), max(ys)-min(ys))
            if span < 2:
                continue
            cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
            heads.append((cx-span, cy-span, cx+span, cy+span))
        return heads


class GazelleEstimator:
    model_name = "gazelle_dinov2_vitb14_inout"

    def __init__(self, weights, device="cpu", in_frame_threshold=.5):
        if not isfinite(in_frame_threshold) or not 0 <= in_frame_threshold <= 1:
            raise ValueError("In-frame threshold must be in [0, 1]")
        import torch
        self.torch = torch
        self.device = torch.device(f"cuda:{device}" if device.isdigit() else device)
        self.threshold = in_frame_threshold
        # Upstream downloads/caches DINOv2 backbone code and weights on first use.
        self.model, self.transform = torch.hub.load(
            f"fkryan/gazelle:{GAZELLE_REVISION}", self.model_name,
            pretrained=False, trust_repo=True)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        expected = {k for k in self.model.state_dict() if not k.startswith("backbone")}
        if set(state) != expected:
            raise ValueError("Expected the complete Gaze-LLE ViT-B in/out decoder state dictionary")
        self.model.load_gazelle_state_dict(state)
        self.model.to(self.device).eval()

    def predict(self, image, head_boxes):
        if not head_boxes:
            return []
        from PIL import Image
        h, w = image.shape[:2]
        bboxes = [(x1/w, y1/h, x2/w, y2/h) for x1, y1, x2, y2 in head_boxes]
        rgb = Image.fromarray(image[:, :, ::-1].copy())
        with self.torch.inference_mode():
            output = self.model({"images": self.transform(rgb).unsqueeze(0).to(self.device),
                                 "bboxes": [bboxes]})
        heatmaps = output["heatmap"][0].detach().cpu().tolist()
        scores = output["inout"][0].detach().cpu().tolist()
        if len(heatmaps) != len(head_boxes) or len(scores) != len(head_boxes):
            raise ValueError("Unexpected Gaze-LLE output dimensions")
        predictions = []
        for heatmap, score in zip(heatmaps, scores, strict=True):
            point, reason = decode_heatmap(heatmap, score, (w, h), self.threshold)
            predictions.append((point, score if isfinite(score) and 0 <= score <= 1 else None, reason))
        return predictions
