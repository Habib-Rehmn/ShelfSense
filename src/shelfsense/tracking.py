"""Tracking contracts and an optional Ultralytics adapter."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self):
        if self.track_id < 0 or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid track ID or confidence")
        if len(self.bbox) != 4 or not all(isfinite(v) for v in self.bbox):
            raise ValueError("Bounding box must contain four finite coordinates")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box must have positive area")


class Tracker(Protocol):
    def track(self, frame: Any) -> list[Track]: ...


class YoloByteTracker:
    """Create one adapter per video; IDs are never cross-session identities."""

    def __init__(self, weights: str, confidence: float = 0.1, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        if self.model.names.get(0) != "person":
            raise ValueError("Use COCO-compatible weights with person at class 0")

    def track(self, frame: Any) -> list[Track]:
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            conf=self.confidence, device=self.device, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        return [
            Track(int(track_id), tuple(map(float, bbox)), float(confidence))
            for track_id, bbox, confidence in zip(
                boxes.id.cpu().tolist(), boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(), strict=True,
            )
        ]
