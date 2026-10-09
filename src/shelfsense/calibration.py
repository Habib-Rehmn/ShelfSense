"""Supervised affine gaze mapping with explicit calibration coverage gates."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import pi
from pathlib import Path

from .geometry import ShelfGeometry, finite_vector, least_squares, validate_zones, zone_at


def features(face_bbox, yaw, pitch, image_size):
    x1, y1, x2, y2 = finite_vector(face_bbox, 4, "face_bbox")
    yaw, pitch = finite_vector((yaw, pitch), 2, "gaze angles")
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Face box must be inside the calibrated image")
    if abs(yaw) >= pi / 2 or abs(pitch) >= pi / 2:
        raise ValueError("Supervised baseline requires front-facing gaze angles")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), yaw, pitch)


@dataclass(frozen=True)
class ShelfTarget:
    track_id: int
    status: str
    reason: str | None = None
    shelf_point: tuple[float, float] | None = None
    zone_id: str | None = None


class Calibration:
    def __init__(self, document):
        # Persist the labeled samples, refit deterministically when loading, and
        # hash all input settings so every result identifies its calibration.
        if not isinstance(document, dict) or document.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported calibration schema; expected version 1 or 2")
        self.image_targets = document["schema_version"] == 2
        required = {"schema_version", "shelf_id", "image_size", "corners", "zones"}
        if not self.image_targets:
            required.add("samples")
        if set(document) != required:
            raise ValueError(f"Calibration requires exactly: {', '.join(sorted(required))}")
        if not isinstance(document["shelf_id"], str) or not document["shelf_id"].strip():
            raise ValueError("shelf_id must be a nonempty string")
        self.geometry = ShelfGeometry(document["image_size"], document["corners"])
        self.zones = validate_zones(document["zones"])
        if self.image_targets:
            self._json = json.dumps(document, sort_keys=True, allow_nan=False)
            self.calibration_id = sha256(self._json.encode()).hexdigest()
            self.shelf_id = document["shelf_id"]
            return
        samples = document["samples"]
        if not isinstance(samples, list) or len(samples) < 6:
            raise ValueError("Collect at least six labeled calibration samples")
        rows, targets = [], []
        for sample in samples:
            if not isinstance(sample, dict) or set(sample) != {"face_bbox", "yaw_rad", "pitch_rad", "target_image"}:
                raise ValueError("Each sample requires face_bbox, yaw_rad, pitch_rad, target_image")
            rows.append(features(sample["face_bbox"], sample["yaw_rad"], sample["pitch_rad"],
                                 self.geometry.image_size))
            target = self.geometry.to_shelf(sample["target_image"])
            if any(v < -1e-8 or v > 1 + 1e-8 for v in target):
                raise ValueError("Calibration targets must lie on the shelf")
            targets.append(tuple(max(0., min(1., v)) for v in target))
        self.minimum = [min(row[i] for row in rows) for i in range(4)]
        self.maximum = [max(row[i] for row in rows) for i in range(4)]
        self.active = [i for i in range(4) if self.maximum[i] - self.minimum[i] > 1e-6]
        if 2 not in self.active or 3 not in self.active:
            raise ValueError("Calibration must vary both yaw and pitch")
        if len(samples) < len(self.active) + 2:
            raise ValueError("Collect more samples than affine coefficients")
        design = [self._design(row) for row in rows]
        self.coefficients = [least_squares(design, [point[i] for point in targets]) for i in range(2)]
        # A read-only snapshot prevents caller mutations changing persisted calibration.
        self._json = json.dumps(document, sort_keys=True, allow_nan=False)
        self.calibration_id = sha256(self._json.encode()).hexdigest()
        self.shelf_id = document["shelf_id"]
        error = sum(sum((a-b)**2 for a, b in zip(self._predict(row), target))
                    for row, target in zip(rows, targets))
        self.fit_rmse = (error / len(rows)) ** .5

    def _design(self, row):
        return [1.] + [(row[i] - self.minimum[i]) / (self.maximum[i] - self.minimum[i])
                       for i in self.active]

    def _predict(self, row):
        design = self._design(row)
        return tuple(sum(a*b for a, b in zip(coefficients, design)) for coefficients in self.coefficients)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as output:
            output.write(json.dumps(json.loads(self._json), indent=2) + "\n")

    def summary(self):
        if self.image_targets:
            return {"calibration_id": self.calibration_id, "shelf_id": self.shelf_id,
                    "method": "image_target_homography_v1", "image_size": self.geometry.image_size}
        return {"calibration_id": self.calibration_id, "shelf_id": self.shelf_id,
                "method": "supervised_affine_v1", "image_size": self.geometry.image_size,
                "fit_rmse": self.fit_rmse, "fit_rmse_is_training_error": True}

    def map_observation(self, observation, image_size):
        track_id = observation.track_id
        if tuple(image_size) != self.geometry.image_size:
            raise ValueError("Frame resolution differs from shelf calibration")
        if observation.status != "estimated":
            return ShelfTarget(track_id, "unknown", "gaze_unavailable")
        if self.image_targets:
            try:
                x, y = finite_vector(observation.target_image, 2, "target_image")
                if not (0 <= x <= image_size[0] and 0 <= y <= image_size[1]):
                    raise ValueError("Target is outside the image")
                point = self.geometry.to_shelf((x, y))
            except (ValueError, AttributeError):
                return ShelfTarget(track_id, "unknown", "invalid_gaze")
            return self._map_point(track_id, point)
        try:
            row = features(observation.face_bbox, observation.yaw_rad, observation.pitch_rad, image_size)
        except ValueError:
            return ShelfTarget(track_id, "unknown", "invalid_gaze")
        # No extrapolation. A small tolerance only absorbs floating-point roundoff.
        if any(value < low - 1e-8 or value > high + 1e-8
               for value, low, high in zip(row, self.minimum, self.maximum)):
            return ShelfTarget(track_id, "unknown", "outside_calibration")
        return self._map_point(track_id, self._predict(row))

    def _map_point(self, track_id, point):
        if any(value < -1e-8 or value > 1 + 1e-8 for value in point):
            return ShelfTarget(track_id, "unknown", "off_shelf", shelf_point=point)
        point = tuple(max(0., min(1., value)) for value in point)
        zone = zone_at(point, self.zones)
        if zone is None:
            return ShelfTarget(track_id, "unknown", "unmapped_zone", shelf_point=point)
        return ShelfTarget(track_id, "estimated", shelf_point=point, zone_id=zone)

    def map_frame(self, observations, image_size):
        if tuple(image_size) != self.geometry.image_size:
            raise ValueError("Frame resolution differs from shelf calibration")
        return [asdict(self.map_observation(observation, image_size)) for observation in observations]
