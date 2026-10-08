"""End-to-end synthetic mapping → events → SQLite → analytics → evaluation."""

import json
from pathlib import Path
from time import perf_counter

from .attention import AttentionEngine
from .attention_replay import derive_attention
from .calibration import Calibration
from .evaluation import evaluate
from .ingestion import read_export
from .provenance import environment_manifest
from .replay import map_gaze_file
from .storage import Store


def run_smoke(output_dir, fixtures_root=Path(".")):
    root, output = Path(fixtures_root), Path(output_dir)
    calibration_path = root/"configs/shelf.synthetic.json"
    gaze_path = root/"examples/attention-gaze.synthetic.jsonl"
    labels_path = root/"examples/labels.synthetic.json"
    calibration = Calibration.load(calibration_path)
    labels = json.loads(labels_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    timings = {}
    started = perf_counter()
    with gaze_path.open() as source, (output/"targets.jsonl").open("x") as sink:
        frames = map_gaze_file(source, sink, calibration)
    timings["mapping_s"] = perf_counter()-started
    started = perf_counter()
    with (output/"targets.jsonl").open() as source, (output/"attention.jsonl").open("x") as sink:
        derive_attention(source, sink, AttentionEngine(calibration))
    timings["attention_s"] = perf_counter()-started
    started = perf_counter()
    with (output/"attention.jsonl").open() as source:
        data = read_export(source, calibration)
    store = Store("sqlite:///"+str((output/"analytics.db").resolve()))
    store.initialize()
    store.ingest(data)
    if store.ingest(data)["imported"]:
        raise RuntimeError("Duplicate ingestion changed the run")
    report = store.report(data["run_id"])
    timings["ingestion_and_report_s"] = perf_counter()-started
    if abs(report["total_attention_s"]-1.8) > 1e-8 or sum(p["episode_count"] for p in report["products"]) != 2:
        raise RuntimeError("Synthetic attention metrics did not match expected results")
    evaluations = {}
    for kind in ("raw", "smoothed"):
        with (output/"attention.jsonl").open() as source:
            evaluations[kind] = evaluate(source, labels, calibration, kind)
    result = {"synthetic": True, "run_id": data["run_id"], "frames": frames,
              "stage_seconds": timings,
              "replay_frames_per_second": frames/(timings["mapping_s"]+timings["attention_s"]),
              "timing_scope": "Synthetic gaze replay only; excludes neural inference, video decode, API and dashboard",
              "analytics": report, "evaluation": evaluations}
    with (output/"report.json").open("x") as sink:
        json.dump(result, sink, indent=2, allow_nan=False)
    with (output/"manifest.json").open("x") as sink:
        json.dump(environment_manifest([calibration_path, gaze_path, labels_path]), sink, indent=2)
    return result
