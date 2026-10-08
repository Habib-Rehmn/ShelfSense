"""Held-out, annotation-aligned evaluation with explicit abstention coverage."""

import json
from math import acos, degrees, hypot, isfinite

from .geometry import finite_vector, zone_at


def angular_error(predicted, actual):
    a = finite_vector(predicted, 3, "predicted direction")
    b = finite_vector(actual, 3, "labeled direction")
    na, nb = hypot(*a), hypot(*b)
    if na < 1e-12 or nb < 1e-12:
        raise ValueError("Direction vectors must be nonzero")
    cosine = sum((x/na)*(y/nb) for x, y in zip(a, b))
    return degrees(acos(max(-1., min(1., cosine))))


def key_for(value):
    sid, index, tid = value["session_id"], value["frame_index"], value["track_id"]
    if not isinstance(sid, str) or not sid or type(index) is not int or index < 0 or type(tid) is not int or tid < 0:
        raise ValueError("Annotation keys require session_id and nonnegative integer frame_index/track_id")
    return sid, index, tid


def validate_labels(document, calibration):
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or document.get("split") not in {"validation", "test"}
            or type(document.get("synthetic")) is not bool
            or not isinstance(document.get("dataset_id"), str) or not document["dataset_id"].strip()):
        raise ValueError("Labels require schema_version=1, dataset_id, validation/test split and synthetic boolean")
    calibration_sessions = document.get("calibration_session_ids")
    if (not isinstance(calibration_sessions, list) or not calibration_sessions
            or any(not isinstance(s, str) or not s for s in calibration_sessions)):
        raise ValueError("Declare the session IDs used to fit calibration")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("At least one held-out labeled sample is required")
    zones = {z["id"] for z in calibration.zones}
    keys = set()
    for sample in samples:
        key = key_for(sample)
        if key in keys:
            raise ValueError("Duplicate annotation key")
        if key[0] in calibration_sessions:
            raise ValueError("Evaluation sessions overlap declared calibration sessions")
        keys.add(key)
        if sample["zone_id"] not in zones:
            raise ValueError("Ground-truth zone must be a configured shelf zone")
        if sample.get("shelf_point") is not None:
            point = finite_vector(sample["shelf_point"], 2, "labeled shelf point")
            if zone_at(point, calibration.zones) != sample["zone_id"]:
                raise ValueError("Labeled shelf point and zone disagree")
        if sample.get("direction_camera") is not None:
            angular_error(sample["direction_camera"], sample["direction_camera"])
        conditions = sample.get("conditions", {})
        if not isinstance(conditions, dict):
            raise ValueError("conditions must be a JSON object")
        for name, value in conditions.items():
            if name not in {"camera_id", "distance_m", "camera_angle_deg", "lighting"}:
                raise ValueError("Unsupported evaluation condition")
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise ValueError("Condition values must be strings or finite numbers")
            if isinstance(value, (int, float)) and not isfinite(value):
                raise ValueError("Condition values must be finite")
    return samples


def read_predictions(source, calibration, kind="raw"):
    if kind not in {"raw", "smoothed"}:
        raise ValueError("Prediction kind must be raw or smoothed")
    predictions, previous, ended = {}, {}, set()
    schema, run_id = None, None
    for line_number, line in enumerate(source, 1):
        try:
            record = json.loads(line)
            version = record["schema_version"]
            if version not in (3, 4) or (kind == "smoothed" and version != 4):
                raise ValueError("Expected schema 3/4; smoothed evaluation requires schema 4")
            if schema is not None and version != schema:
                raise ValueError("Do not mix output schemas in one evaluation")
            schema = version
            sid = record["session_id"]
            if not isinstance(sid, str) or not sid:
                raise ValueError("Invalid session ID")
            if version == 4:
                metadata = record["attention_metadata"]
                if metadata["calibration_id"] != calibration.calibration_id:
                    raise ValueError("Calibration mismatch")
                if not isinstance(metadata["run_id"], str) or not metadata["run_id"]:
                    raise ValueError("Invalid run ID")
                if run_id is not None and metadata["run_id"] != run_id:
                    raise ValueError("Evaluate one processing run at a time")
                run_id = metadata["run_id"]
                if record["record_type"] == "session_end":
                    if sid not in previous or sid in ended or record["timestamp_s"] != previous[sid][1]:
                        raise ValueError("Invalid session ending")
                    ended.add(sid)
                    continue
                if record["record_type"] != "frame":
                    raise ValueError("Unknown record type")
            if sid in ended:
                raise ValueError("Frame follows its session ending")
            index, timestamp = record["frame_index"], record["timestamp_s"]
            if (type(index) is not int or index < 0 or isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float)) or not isfinite(timestamp) or timestamp < 0):
                raise ValueError("Invalid frame index or timestamp")
            old_index, old_time = previous.get(sid, (-1, -1))
            if index <= old_index or timestamp <= old_time:
                raise ValueError("Frame indices/timestamps must increase per session")
            previous[sid] = (index, timestamp)
            if record["calibration"]["calibration_id"] != calibration.calibration_id:
                raise ValueError("Calibration mismatch")
            if tuple(record["image_size"]) != calibration.geometry.image_size:
                raise ValueError("Prediction resolution differs from calibration")
            observations = record["shelf_targets" if kind == "raw" else "attention"]
            track_ids = [o["track_id"] for o in observations]
            if len(track_ids) != len(set(track_ids)) or track_ids != [t["track_id"] for t in record["tracks"]]:
                raise ValueError("Prediction track IDs must match frame tracks")
            gazes = {o["track_id"]: o for o in record.get("gaze", [])}
            for observation in observations:
                key = key_for({"session_id": sid, "frame_index": index, "track_id": observation["track_id"]})
                status = observation["status"]
                if status not in {"estimated", "unknown"}:
                    raise ValueError("Invalid prediction status")
                if status == "estimated":
                    point = finite_vector(observation["shelf_point"], 2, "predicted shelf point")
                    zone = zone_at(point, calibration.zones)
                    if zone is None or zone != observation["zone_id"]:
                        raise ValueError("Predicted zone and shelf point disagree")
                elif observation.get("zone_id") is not None:
                    raise ValueError("Unknown predictions cannot declare a zone")
                predictions[key] = {**observation,
                    "direction_camera": gazes.get(observation["track_id"], {}).get("direction_camera")}
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Prediction line {line_number}: {exc}") from exc
    if not previous:
        raise ValueError("Prediction file contains no frames")
    if schema == 4 and ended != set(previous):
        raise ValueError("Incomplete attention export: missing session endings")
    return predictions


def score(samples, predictions, zones):
    # Last confusion column is an explicit abstention, not a user-defined zone ID.
    confusion = [[0] * (len(zones)+1) for _ in zones]
    accepted, correct, missing = 0, 0, 0
    point_errors, angular_errors = [], []
    for sample in samples:
        prediction = predictions.get(key_for(sample))
        actual = zones.index(sample["zone_id"])
        predicted = len(zones)
        if prediction is None:
            missing += 1
        elif prediction["status"] == "estimated":
            predicted = zones.index(prediction["zone_id"])
            accepted += 1
            correct += actual == predicted
            if sample.get("shelf_point") is not None:
                point_errors.append(hypot(*(a-b for a, b in zip(prediction["shelf_point"], sample["shelf_point"]))))
        confusion[actual][predicted] += 1
        if prediction and sample.get("direction_camera") is not None and prediction.get("direction_camera") is not None:
            angular_errors.append(angular_error(prediction["direction_camera"], sample["direction_camera"]))
    per_zone = []
    for i, zone in enumerate(zones):
        tp, support = confusion[i][i], sum(confusion[i])
        predicted_count = sum(row[i] for row in confusion)
        precision = tp/predicted_count if predicted_count else 0.
        recall = tp/support if support else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        per_zone.append({"zone_id": zone, "support": support, "precision": precision, "recall": recall, "f1": f1})
    return {"samples": len(samples), "accepted_predictions": accepted, "missing_predictions": missing,
            "coverage": accepted/len(samples), "accuracy_all_labels": correct/len(samples),
            "accuracy_accepted": correct/accepted if accepted else None,
            "macro_f1": sum(z["f1"] for z in per_zone)/len(zones), "per_zone": per_zone,
            "confusion": {"true_zones": zones, "predicted_columns": zones+[None], "counts": confusion},
            "shelf_point_samples": len(point_errors),
            "mean_shelf_point_error": sum(point_errors)/len(point_errors) if point_errors else None,
            "angular_samples": len(angular_errors),
            "mean_angular_error_deg": sum(angular_errors)/len(angular_errors) if angular_errors else None}


def evaluate(source, labels, calibration, kind="raw"):
    try:
        samples = validate_labels(labels, calibration)
        predictions = read_predictions(source, calibration, kind)
        zones = [z["id"] for z in calibration.zones]
        groups = {}
        for name in {name for s in samples for name in s.get("conditions", {})}:
            values = sorted({str(s["conditions"][name]) for s in samples if name in s.get("conditions", {})})
            groups[name] = {value: score([s for s in samples if name in s.get("conditions", {})
                                         and str(s["conditions"][name]) == value],
                                         predictions, zones) for value in values}
        return {"schema_version": 1, "dataset_id": labels["dataset_id"], "split": labels["split"],
                "synthetic": labels["synthetic"], "prediction_kind": kind,
                "calibration_id": calibration.calibration_id, "metrics": score(samples, predictions, zones),
                "unlabeled_predictions": len(set(predictions)-{key_for(s) for s in samples}),
                "by_condition": groups}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed evaluation data: {exc}") from exc
