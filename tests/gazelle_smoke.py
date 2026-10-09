"""Opt-in pretrained CPU smoke. Synthetic pixels do not measure accuracy."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from shelfsense.gazelle_adapters import GAZELLE_REVISION, GazelleEstimator, PoseHeadDetector
from shelfsense.provenance import environment_manifest

torch.set_num_threads(2)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
head_weights = '/models/yolo26n-pose.pt'
gaze_weights = '/models/gazelle_dinov2_vitb14_inout.pt'
heads = PoseHeadDetector(head_weights)
assert heads.detect(frame) == []
estimator = GazelleEstimator(gaze_weights)
start = time.perf_counter()
predictions = estimator.predict(frame, [(100, 100, 160, 180), (400, 100, 460, 180)])
elapsed = time.perf_counter() - start
assert len(predictions) == 2
for point, score, reason in predictions:
    assert 0 <= score <= 1
    assert point is None or (0 <= point[0] <= 640 and 0 <= point[1] <= 480)
backbone_weights = list((Path(torch.hub.get_dir())/'checkpoints').glob('dinov2_vitb14*.pth'))
cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
            if line.startswith('model name')), 'unknown')
result = {'cpu': cpu, 'synthetic': True, 'accuracy_measured': False, 'model': estimator.model_name,
          'heads': 2, 'cpu_threads': 2, 'gaze_forward_seconds': elapsed,
          'predictions': predictions, 'gazelle_source': {'repository': 'fkryan/gazelle', 'revision': GAZELLE_REVISION},
          'manifest': environment_manifest([head_weights, gaze_weights, *backbone_weights])}
Path('/tmp/gazelle-smoke-report.json').write_text(json.dumps(result, indent=2))
print(f'Pretrained Gaze-LLE CPU smoke passed: two synthetic heads in {elapsed:.3f}s; no accuracy claim.')
