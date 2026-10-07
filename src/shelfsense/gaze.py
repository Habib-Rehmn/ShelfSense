"""Face association and gaze observations, independent of ML dependencies."""

from dataclasses import dataclass
from math import ceil, cos, floor, isfinite, pi, sin
from typing import Any, Protocol

from .tracking import Track

BBox = tuple[float, float, float, float]


class InvalidGazePrediction(ValueError):
    """Model produced unusable angles, rather than an infrastructure error."""


def clip_box(box: BBox, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Outward-round and clip pixel boxes; never use negative slice indices."""
    if len(box) != 4 or not all(isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    result = (max(0, floor(x1)), max(0, floor(y1)),
              min(width, ceil(x2)), min(height, ceil(y2)))
    return result if result[2] > result[0] and result[3] > result[1] else None


@dataclass(frozen=True)
class Angles:
    """Radians: positive yaw points image-left, positive pitch image-up."""

    yaw_rad: float
    pitch_rad: float

    def __post_init__(self):
        if not all(isfinite(v) and abs(v) <= pi for v in (self.yaw_rad, self.pitch_rad)):
            raise InvalidGazePrediction("Gaze angles must be finite radians in [-pi, pi]")

    @property
    def direction(self) -> tuple[float, float, float]:
        """Camera axes: x right, y down, z away; frontal gaze points -z."""
        yaw, pitch = self.yaw_rad, self.pitch_rad
        return (-cos(pitch) * sin(yaw), -sin(pitch), -cos(pitch) * cos(yaw))


@dataclass(frozen=True)
class GazeObservation:
    track_id: int
    status: str
    reason: str | None = None
    face_bbox: tuple[int, int, int, int] | None = None
    yaw_rad: float | None = None
    pitch_rad: float | None = None
    direction_camera: tuple[float, float, float] | None = None


class FaceDetector(Protocol):
    def detect(self, image: Any) -> list[BBox]: ...


class GazeEstimator(Protocol):
    def estimate(self, face_bgr: Any) -> Angles: ...


class GazeAnalyzer:
    def __init__(self, detector: FaceDetector, estimator: GazeEstimator,
                 min_face_size: int = 32):
        if min_face_size <= 0:
            raise ValueError("Minimum face size must be positive")
        self.detector = detector
        self.estimator = estimator
        self.min_face_size = min_face_size

    def observe(self, image: Any, tracks: list[Track]) -> list[GazeObservation]:
        if not tracks:
            return []
        height, width = image.shape[:2]
        faces = [clipped for box in self.detector.detect(image)
                 if (clipped := clip_box(box, width, height)) is not None]
        candidates: dict[int, list[int]] = {track.track_id: [] for track in tracks}
        owners: dict[int, list[int]] = {}
        for index, face in enumerate(faces):
            cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
            owners[index] = []
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                # Broad upper-body gate, not an estimate of head pose.
                if x1 <= cx < x2 and y1 <= cy < y1 + .6 * (y2 - y1):
                    candidates[track.track_id].append(index)
                    owners[index].append(track.track_id)
        observations = []
        for track in tracks:
            matched = candidates[track.track_id]
            if not matched:
                observations.append(GazeObservation(track.track_id, "unknown", "no_face"))
                continue
            if len(matched) != 1 or len(owners[matched[0]]) != 1:
                observations.append(GazeObservation(track.track_id, "unknown", "ambiguous_face"))
                continue
            box = faces[matched[0]]
            x1, y1, x2, y2 = box
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                observations.append(GazeObservation(track.track_id, "unknown", "face_too_small", box))
                continue
            try:
                angles = self.estimator.estimate(image[y1:y2, x1:x2])
            except InvalidGazePrediction:
                observations.append(GazeObservation(track.track_id, "unknown", "invalid_angles", box))
                continue
            observations.append(GazeObservation(
                track.track_id, "estimated", face_bbox=box,
                yaw_rad=angles.yaw_rad, pitch_rad=angles.pitch_rad,
                direction_camera=angles.direction,
            ))
        return observations
