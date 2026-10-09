"""Scene gaze targets and conservative head-to-track association."""

from dataclasses import dataclass
from math import isfinite

from .gaze import clip_box


@dataclass(frozen=True)
class ImageGazeObservation:
    track_id: int
    status: str
    reason: str | None = None
    head_bbox: tuple | None = None
    target_image: tuple | None = None
    in_frame_score: float | None = None


def decode_heatmap(heatmap, score, image_size, threshold):
    """Use the global peak, never force an off-shelf target onto a shelf."""
    if not isfinite(score) or not 0 <= score <= 1:
        return None, "invalid_prediction"
    if score < threshold:
        return None, "out_of_frame"
    if not heatmap or not heatmap[0] or any(len(row) != len(heatmap[0]) for row in heatmap):
        return None, "invalid_prediction"
    values = [v for row in heatmap for v in row]
    if any(not isfinite(v) or not 0 <= v <= 1 for v in values):
        return None, "invalid_prediction"
    if max(values) - min(values) <= 1e-8:
        return None, "uninformative_heatmap"
    index = max(range(len(values)), key=values.__getitem__)
    h, w = len(heatmap), len(heatmap[0])
    # Pixel-cell centers avoid treating the first bin as the image boundary.
    return (((index % w + .5) / w * image_size[0]),
            ((index // w + .5) / h * image_size[1])), None


class SceneGazeAnalyzer:
    observation_kind = "image_target"

    def __init__(self, detector, estimator, min_head_size=16):
        if min_head_size <= 0:
            raise ValueError("Minimum head size must be positive")
        self.detector, self.estimator = detector, estimator
        self.min_head_size = min_head_size

    def observe(self, image, tracks):
        if not tracks:
            return []
        height, width = image.shape[:2]
        heads = [box for raw in self.detector.detect(image)
                 if (box := clip_box(raw, width, height)) is not None]
        owners = []
        for x1, y1, x2, y2 in heads:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            owners.append([i for i, t in enumerate(tracks)
                           if t.bbox[0] <= cx < t.bbox[2]
                           and t.bbox[1] <= cy < t.bbox[1] + .6*(t.bbox[3]-t.bbox[1])])
        observations, selected = [], []
        for i, track in enumerate(tracks):
            matches = [j for j, ids in enumerate(owners) if i in ids]
            reason, box = "no_head", None
            if matches:
                reason = "ambiguous_head"
                if len(matches) == 1 and len(owners[matches[0]]) == 1:
                    box = heads[matches[0]]
                    reason = "head_too_small"
                    if min(box[2]-box[0], box[3]-box[1]) >= self.min_head_size:
                        selected.append((i, box))
                        reason = "pending"
            observations.append(ImageGazeObservation(track.track_id, "unknown", reason, box))
        if selected:
            predictions = self.estimator.predict(image, [box for _, box in selected])
            if len(predictions) != len(selected):
                raise ValueError("Gaze-LLE prediction count differs from head count")
            for (i, box), (point, score, reason) in zip(selected, predictions, strict=True):
                observations[i] = ImageGazeObservation(
                    tracks[i].track_id, "estimated" if point is not None else "unknown",
                    reason, box, point, score)
        return observations
