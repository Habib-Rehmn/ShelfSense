"""Apply shelf calibration to stored schema-2 gaze frames without ML imports."""

import json
from math import isfinite

from .gaze import GazeObservation
from .gazelle import ImageGazeObservation


def map_gaze_file(source, output, calibration):
    previous = {}
    count = 0
    for line_number, line in enumerate(source, 1):
        try:
            record = json.loads(line)
            if record["schema_version"] != 2:
                raise ValueError("Expected schema-2 gaze frames")
            session = record["session_id"]
            index, timestamp = record["frame_index"], record["timestamp_s"]
            if (not isinstance(session, str) or not session
                    or type(index) is not int or index < 0
                    or isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                    or not isfinite(timestamp) or timestamp < 0):
                raise ValueError("Invalid session, frame index or timestamp")
            prior_index, prior_time = previous.get(session, (-1, -1))
            if index <= prior_index or timestamp <= prior_time:
                raise ValueError("Frame indices and timestamps must increase within a session")
            kind = record.get("gaze_kind", "angles")
            if kind not in {"angles", "image_target"}:
                raise ValueError("Unsupported gaze kind")
            if (kind == "image_target") != calibration.image_targets:
                raise ValueError("Gaze kind does not match calibration method")
            observation_type = ImageGazeObservation if kind == "image_target" else GazeObservation
            observations = [observation_type(**value) for value in record["gaze"]]
            ids = [observation.track_id for observation in observations]
            if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("Invalid or duplicate gaze track IDs")
            if any(o.status not in {"estimated", "unknown"} for o in observations):
                raise ValueError("Invalid gaze status")
            track_ids = [track["track_id"] for track in record["tracks"]]
            if ids != track_ids:
                raise ValueError("Gaze observations must match track IDs and order")
            record["shelf_targets"] = calibration.map_frame(observations, record["image_size"])
            record["schema_version"] = 3
            record["calibration"] = calibration.summary()
            output.write(json.dumps(record, allow_nan=False) + "\n")
            previous[session] = (index, timestamp)
            count += 1
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Gaze input line {line_number}: {exc}") from exc
    if count == 0:
        raise ValueError("Gaze input contains no frames")
    return count
