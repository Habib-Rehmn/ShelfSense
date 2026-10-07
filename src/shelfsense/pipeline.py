"""Stream one JSON record per frame without retaining video in memory."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, TextIO
import json
import uuid

from .tracking import Tracker


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_s: float
    image: Any


def process(frames: Iterable[Frame], tracker: Tracker, output: TextIO,
            session_id: str | None = None) -> int:
    session_id = session_id or str(uuid.uuid4())
    previous_index = -1
    previous_time = -1.0
    count = 0
    for frame in frames:
        if (frame.index <= previous_index or not isfinite(frame.timestamp_s)
                or frame.timestamp_s < 0 or frame.timestamp_s <= previous_time):
            raise ValueError("Frame indices and timestamps must increase strictly")
        tracks = tracker.track(frame.image)
        if len({track.track_id for track in tracks}) != len(tracks):
            raise ValueError("Duplicate track IDs in a frame")
        record = {
            "schema_version": 1, "session_id": session_id,
            "frame_index": frame.index, "timestamp_s": frame.timestamp_s,
            "tracks": [asdict(track) for track in tracks],
        }
        output.write(json.dumps(record, allow_nan=False) + "\n")
        previous_index, previous_time = frame.index, frame.timestamp_s
        count += 1
    return count
