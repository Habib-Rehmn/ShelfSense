"""Validate completed attention exports and reduce them into anonymous analytics."""

from hashlib import sha256
import json

from .attention import AttentionConfig, AttentionEngine

HEATMAP_COLUMNS = 12
HEATMAP_ROWS = 8


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_export(source, calibration):
    """Recompute temporal results before accepting events from an offline file.

    Keep derived events and aggregates, not frames or face/person bounding boxes.
    """
    engine = None
    sessions, events, episodes = {}, [], []
    summaries = None
    digest = sha256()
    for line_number, line in enumerate(source, 1):
        try:
            record = json.loads(line)
            if record["schema_version"] != 4:
                raise ValueError("Ingestion requires schema-4 attention output")
            metadata = record["attention_metadata"]
            if engine is None:
                run_id = metadata["run_id"]
                if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 200:
                    raise ValueError("Invalid run ID")
                engine = AttentionEngine(calibration, AttentionConfig(**metadata["config"]))
                engine.run_id = run_id
            if canonical(metadata) != canonical(engine.metadata()):
                raise ValueError("Run configuration or calibration changed within the file")
            if record["record_type"] == "frame":
                if summaries is not None:
                    raise ValueError("Frame follows session completion records")
                expected = engine.process({**record, "schema_version": 3})
                for key in ("attention", "events"):
                    if canonical(record[key]) != canonical(expected[key]):
                        raise ValueError(f"{key} does not match recomputed temporal output")
                sid, timestamp = record["session_id"], record["timestamp_s"]
                if sid not in sessions:
                    sessions[sid] = {
                        "session_id": sid, "first_time_s": timestamp, "last_time_s": timestamp,
                        "frame_count": 0, "track_observations": 0, "valid_observations": 0,
                        "tracked": set(), "viewed": set(), "confirmed": set(),
                        "zones": {z["id"]: {"episode_count": 0, "total_attention_s": 0.,
                                             "viewers": set()} for z in calibration.zones},
                        "heatmap": [[0.] * HEATMAP_COLUMNS for _ in range(HEATMAP_ROWS)],
                        "previous": {},
                    }
                session = sessions[sid]
                session["frame_count"] += 1
                session["last_time_s"] = timestamp
                current = {}
                for observation in record["attention"]:
                    tid = observation["track_id"]
                    session["tracked"].add(tid)
                    session["track_observations"] += 1
                    if observation["status"] != "estimated":
                        continue
                    session["viewed"].add(tid)
                    session["valid_observations"] += 1
                    current[tid] = (timestamp, observation["shelf_point"])
                    if tid in session["previous"]:
                        previous_time, point = session["previous"][tid]
                        dt = timestamp-previous_time
                        if dt <= engine.config.max_gap_s + 1e-9:
                            x = min(HEATMAP_COLUMNS-1, int(point[0]*HEATMAP_COLUMNS))
                            y = min(HEATMAP_ROWS-1, int(point[1]*HEATMAP_ROWS))
                            session["heatmap"][y][x] += dt
                session["previous"] = current
            elif record["record_type"] == "session_end":
                if summaries is None:
                    summaries = engine.finish()
                if not summaries or canonical(record) != canonical(summaries.pop(0)):
                    raise ValueError("Invalid or duplicate session completion record")
            else:
                raise ValueError("Invalid attention record type")
            for event in record["events"]:
                events.append(event)
                if event["type"] == "ATTENTION_ENDED":
                    episodes.append(event)
                    session = sessions[event["session_id"]]
                    session["confirmed"].add(event["track_id"])
                    zone = session["zones"][event["zone_id"]]
                    zone["episode_count"] += 1
                    zone["total_attention_s"] += event["duration_s"]
                    zone["viewers"].add(event["track_id"])
            digest.update((canonical(record)+"\n").encode())
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Attention import line {line_number}: {exc}") from exc
    if not sessions or summaries is None or summaries:
        raise ValueError("Incomplete export: all sessions require completion records")
    for session in sessions.values():
        session.pop("previous")
        for key in ("tracked", "viewed", "confirmed"):
            session[key] = len(session[key])
        for zone in session["zones"].values():
            zone["unique_tracks"] = len(zone.pop("viewers"))
    return {"run_id": engine.run_id, "digest": digest.hexdigest(),
            "metadata": {**engine.metadata(), "shelf_id": calibration.shelf_id,
                         "zones": calibration.zones},
            "sessions": list(sessions.values()), "events": events, "episodes": episodes}
