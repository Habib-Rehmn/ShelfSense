"""Read the public FiftyOne export without FiftyOne or untrusted pickle loading."""

import json
from collections import Counter
from math import hypot, isfinite
from pathlib import Path


def load_sessions(root, names, split="test"):
    root = Path(root)
    if split not in {"test", "val", "train", "all"}:
        raise ValueError("Invalid split")
    samples = json.loads((root / "samples.json").read_text())["samples"]
    sessions, ids = {}, {}
    for sample in samples:
        name = f"{sample['subject_id']}_{sample['session_id']}"
        if name not in names:
            continue
        relative = Path(sample["filepath"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Video paths must be relative to dataset root")
        sid = sample["_id"]["$oid"]
        if name in sessions or sid in ids:
            raise ValueError("Duplicate session")
        sessions[name] = {"video": root / relative, "labels": {}}
        ids[sid] = name
    if set(sessions) != set(names):
        raise ValueError("Unknown session requested")
    seen = set()
    for frame in json.loads((root / "frames.json").read_text())["frames"]:
        name = ids.get(frame["_sample_id"]["$oid"])
        if name is None:
            continue
        number = frame["frame_number"]
        if type(number) is not int or number < 1 or (name, number) in seen:
            raise ValueError("Invalid or duplicate frame number")
        seen.add((name, number))
        if split != "all" and frame["split"] != split:
            continue
        box = frame["head"]["bounding_box"]
        point = frame["gaze_target"]["keypoints"][0]["points"][0]
        if (len(box) != 4 or len(point) != 2
                or any(not isfinite(v) or not 0 <= v <= 1 for v in [*box, *point])
                or box[2] <= 0 or box[3] <= 0
                or box[0]+box[2] > 1.000001 or box[1]+box[3] > 1.000001):
            raise ValueError("Invalid normalized annotation")
        raw = (frame.get("scene_mask") or {}).get("mask_path")
        mask = None
        if raw:
            # The export contains the publisher's absolute paths, not local files.
            rel = Path(raw.split("/RetailGaze_V2/")[-1])
            subject, session = name.split("_", 1)
            if (len(rel.parts) != 4 or rel.parts[:3] != (subject, session, "masks")
                    or rel.suffix != ".png" or ".." in rel.parts):
                raise ValueError("Unexpected product mask path")
            mask = root / "masks" / rel
        sessions[name]["labels"][number-1] = {
            "head": (box[0], box[1], box[0]+box[2], box[1]+box[3]),
            "point": tuple(point), "mask": mask, "split": frame["split"]}
    if any(not s["labels"] for s in sessions.values()):
        raise ValueError("Each selected session must have labels in the requested split")
    return sessions


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap/union if union > 0 else 0.


def match_track(tracks, head):
    """Match by annotated head center within upper body; never choose by gaze."""
    cx, cy = (head[0]+head[2])/2, (head[1]+head[3])/2
    candidates = [t for t in tracks if t.bbox[0] <= cx < t.bbox[2]
                  and t.bbox[1] <= cy < t.bbox[1]+.6*(t.bbox[3]-t.bbox[1])]
    return candidates[0] if len(candidates) == 1 else None


def summarize(rows):
    accepted = [r for r in rows if r["point"] is not None]
    eligible = [r for r in rows if r["region_eligible"]]
    accepted_regions = [r for r in eligible if r["point"] is not None]
    correct = sum(r["region_hit"] is True for r in eligible)
    mean = lambda values: sum(values)/len(values) if values else None
    return {"samples": len(rows), "accepted": len(accepted),
            "coverage": len(accepted)/len(rows) if rows else None,
            "mean_error_px_accepted": mean([r["error_px"] for r in accepted]),
            "mean_error_normalized_xy_accepted": mean([r["error_normalized_xy"] for r in accepted]),
            "region_eligible": len(eligible), "region_hits": correct,
            "region_accuracy_all": correct/len(eligible) if eligible else None,
            "region_accuracy_accepted": correct/len(accepted_regions) if accepted_regions else None,
            "unknown_reasons": dict(Counter(r["reason"] for r in rows if r["point"] is None))}


def prediction_record(point, reason, actual, size, mask=None, mask_status="missing"):
    width, height = size
    eligible = mask_status == "valid"
    hit = None
    if eligible:
        hit = False
        if point is not None and 0 <= point[0] < width and 0 <= point[1] < height:
            hit = bool(mask[int(point[1])][int(point[0])])
    return {"point": point, "reason": reason,
            "error_px": hypot(point[0]-actual[0], point[1]-actual[1]) if point else None,
            "error_normalized_xy": hypot((point[0]-actual[0])/width, (point[1]-actual[1])/height) if point else None,
            "region_eligible": eligible, "mask_status": mask_status, "region_hit": hit}
