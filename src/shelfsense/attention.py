"""Time-aware shelf attention episodes. These are not physiological fixations."""

from dataclasses import asdict, dataclass, field
from math import exp, hypot, isfinite
from uuid import uuid4

from .geometry import finite_vector, zone_at


@dataclass(frozen=True)
class AttentionConfig:
    smoothing_tau_s: float = .15
    min_duration_s: float = .5
    max_gap_s: float = .25
    max_radius: float = .08
    track_timeout_s: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.track_timeout_s < self.max_gap_s:
            raise ValueError("track_timeout_s must be at least max_gap_s")


@dataclass
class TrackState:
    last_seen: float
    last_valid: float | None = None
    point: tuple | None = None
    anchor: tuple | None = None
    zone: str | None = None
    start: float | None = None
    episode_id: str | None = None


@dataclass
class SessionState:
    last_index: int = -1
    last_time: float = -1
    tracks: dict[int, TrackState] = field(default_factory=dict)


class AttentionEngine:
    def __init__(self, calibration, config=None):
        self.calibration = calibration
        self.config = config or AttentionConfig()
        self.run_id = str(uuid4())
        self.sessions = {}
        self.sequence = 0
        self.closed = False

    def metadata(self):
        return {"run_id": self.run_id, "method": "time_ema_attention_v1",
                "config": asdict(self.config), "calibration_id": self.calibration.calibration_id}

    def _event(self, events, session, track_id, kind, timestamp, emitted, **fields):
        self.sequence += 1
        event = {"event_id": f"{self.run_id}:{self.sequence}", "type": kind,
                 "session_id": session, "track_id": track_id,
                 "shelf_id": self.calibration.shelf_id,
                 "timestamp_s": timestamp, "emitted_at_s": emitted, **fields}
        events.append(event)
        return event["event_id"]

    def _end(self, state, events, session, track_id, emitted, reason):
        if state.episode_id is not None:
            self._event(events, session, track_id, "ATTENTION_ENDED", state.last_valid, emitted,
                        episode_id=state.episode_id, zone_id=state.zone,
                        start_time_s=state.start, end_time_s=state.last_valid,
                        duration_s=state.last_valid-state.start, reason=reason)
        state.last_valid = state.point = state.anchor = state.zone = state.start = state.episode_id = None

    def _validate(self, record):
        if not isinstance(record, dict) or record.get("schema_version") != 3:
            raise ValueError("Attention requires schema-3 calibrated frames")
        if record.get("calibration", {}).get("calibration_id") != self.calibration.calibration_id:
            raise ValueError("Frame calibration ID differs from attention calibration")
        if record.get("image_size") not in (list(self.calibration.geometry.image_size),
                                            self.calibration.geometry.image_size):
            raise ValueError("Frame resolution differs from attention calibration")
        session, index, timestamp = record["session_id"], record["frame_index"], record["timestamp_s"]
        if (not isinstance(session, str) or not session or type(index) is not int or index < 0
                or isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not isfinite(timestamp) or timestamp < 0):
            raise ValueError("Invalid session, frame index or media timestamp")
        prior = self.sessions.get(session)
        if prior and (index <= prior.last_index or timestamp <= prior.last_time):
            raise ValueError("Frame indices and timestamps must increase within a session")
        targets = record["shelf_targets"]
        ids = [target["track_id"] for target in targets]
        if (any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids))
                or ids != [track["track_id"] for track in record["tracks"]]):
            raise ValueError("Shelf targets must match unique track IDs in order")
        for target in targets:
            if target["status"] not in {"estimated", "unknown"}:
                raise ValueError("Invalid shelf target status")
            if target["status"] == "estimated":
                point = finite_vector(target["shelf_point"], 2, "shelf_point")
                zone = zone_at(point, self.calibration.zones)
                if zone is None or target["zone_id"] != zone:
                    raise ValueError("Estimated target must lie in its declared shelf zone")
            elif target.get("zone_id") is not None:
                raise ValueError("Unknown shelf target must not declare a zone")
        return session, index, timestamp, targets

    def process(self, record):
        if self.closed:
            raise ValueError("Attention engine has already finished")
        try:
            session, index, timestamp, targets = self._validate(record)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Malformed attention frame: {exc}") from exc
        states = self.sessions.setdefault(session, SessionState())
        events, observations = [], []
        current_ids = {target["track_id"] for target in targets}
        for track_id, state in list(states.tracks.items()):
            # Expire even a returning ID if no observations arrived during timeout.
            if timestamp-state.last_seen >= self.config.track_timeout_s:
                self._end(state, events, session, track_id, timestamp, "track_timeout")
                self._event(events, session, track_id, "SHOPPER_LEFT", state.last_seen, timestamp,
                            reason="track_timeout")
                del states.tracks[track_id]
            elif track_id not in current_ids:
                self._end(state, events, session, track_id, timestamp, "track_missing")
        for target in targets:
            track_id = target["track_id"]
            if track_id not in states.tracks:
                states.tracks[track_id] = TrackState(timestamp)
                self._event(events, session, track_id, "SHOPPER_ENTERED", timestamp, timestamp)
            state = states.tracks[track_id]
            state.last_seen = timestamp
            if target["status"] != "estimated":
                self._end(state, events, session, track_id, timestamp, "gaze_unavailable")
                observations.append({"track_id": track_id, "status": "unknown",
                                     "reason": target.get("reason") or "gaze_unavailable",
                                     "shelf_point": None, "zone_id": None,
                                     "episode_id": None, "candidate_duration_s": 0.})
                continue
            if state.last_valid is not None and timestamp-state.last_valid > self.config.max_gap_s + 1e-9:
                self._end(state, events, session, track_id, timestamp, "observation_gap")
            raw = tuple(target["shelf_point"])
            point = raw
            if state.point is not None:
                alpha = 1-exp(-(timestamp-state.last_valid)/self.config.smoothing_tau_s)
                point = tuple(old+alpha*(new-old) for old, new in zip(state.point, raw))
            zone = zone_at(point, self.calibration.zones)
            if zone is None:
                self._end(state, events, session, track_id, timestamp, "unmapped_zone")
                observations.append({"track_id": track_id, "status": "unknown", "reason": "unmapped_zone",
                                     "shelf_point": point, "zone_id": None,
                                     "episode_id": None, "candidate_duration_s": 0.})
                continue
            if state.zone is not None and zone != state.zone:
                self._end(state, events, session, track_id, timestamp, "zone_changed")
            elif state.anchor is not None and hypot(point[0]-state.anchor[0], point[1]-state.anchor[1]) > self.config.max_radius:
                self._end(state, events, session, track_id, timestamp, "point_moved")
            if state.start is None:
                state.start, state.anchor, state.zone = timestamp, point, zone
            state.point, state.last_valid = point, timestamp
            duration = timestamp-state.start
            if state.episode_id is None and duration + 1e-9 >= self.config.min_duration_s:
                state.episode_id = self._event(
                    events, session, track_id, "ATTENTION_STARTED", state.start, timestamp,
                    zone_id=zone, start_time_s=state.start)
                events[-1]["episode_id"] = state.episode_id
            observations.append({"track_id": track_id, "status": "estimated", "reason": None,
                                 "shelf_point": point, "zone_id": zone,
                                 "episode_id": state.episode_id, "candidate_duration_s": duration})
        states.last_index, states.last_time = index, timestamp
        return {**record, "schema_version": 4, "record_type": "frame",
                "attention_metadata": self.metadata(), "attention": observations, "events": events}

    def finish(self):
        """Close at last observed media time, without adding a guessed frame duration."""
        if self.closed:
            return []
        summaries = []
        for session, states in self.sessions.items():
            events = []
            for track_id, state in states.tracks.items():
                self._end(state, events, session, track_id, states.last_time, "session_end")
                self._event(events, session, track_id, "SHOPPER_LEFT", state.last_seen, states.last_time,
                            reason="session_end")
            summaries.append({"schema_version": 4, "record_type": "session_end", "session_id": session,
                              "timestamp_s": states.last_time, "attention_metadata": self.metadata(),
                              "events": events})
        self.sessions.clear()
        self.closed = True
        return summaries
