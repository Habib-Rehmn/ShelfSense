"""Opt-in L2CS dependency/adapter check, using synthetic, untrained weights."""

from importlib import metadata
import json
from math import isclose, pi
from pathlib import Path
import tempfile

import numpy as np
import torch
from l2cs.utils import getArch

from shelfsense.gaze_adapters import L2CSEstimator
from shelfsense.provenance import environment_manifest


def run():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        weights = Path(directory) / "synthetic-l2cs-resnet50.pkl"
        model = getArch("ResNet50", 90)
        # Distinct, known output bins catch swapped heads, signs and units while
        # still exercising real checkpoint loading and the full forward pass.
        with torch.no_grad():
            for head, peak in ((model.fc_yaw_gaze, 50), (model.fc_pitch_gaze, 35)):
                head.weight.zero_()
                head.bias.fill_(-100)
                head.bias[peak] = 100
        torch.save(model.state_dict(), weights)
        del model
        estimator = L2CSEstimator(str(weights), device="cpu", arch="ResNet50")
        crop = np.zeros((80, 96, 3), dtype=np.uint8)
        crop[:, :, 2] = 255
        observed = []

        def check_input(module, inputs):
            tensor = inputs[0]
            expected = torch.tensor([(1 - .485) / .229, -.456 / .224, -.406 / .225])
            if tensor.shape != (1, 3, 448, 448):
                raise AssertionError(f"Unexpected L2CS input shape: {tensor.shape}")
            if not torch.allclose(tensor, expected.reshape(1, 3, 1, 1).expand_as(tensor)):
                raise AssertionError("BGR-to-RGB conversion or input normalization changed")
            if torch.is_grad_enabled():
                raise AssertionError("Inference unexpectedly enabled gradients")
            observed.append(True)

        hook = estimator.model.register_forward_pre_hook(check_input)
        try:
            angles = estimator.estimate(crop)
        finally:
            hook.remove()
        if observed != [True]:
            raise AssertionError("Expected exactly one real model forward pass")
        if not (isclose(angles.yaw_rad, 20 * pi / 180, abs_tol=1e-6)
                and isclose(angles.pitch_rad, -40 * pi / 180, abs_tol=1e-6)):
            raise AssertionError(f"Unexpected synthetic head outputs: {angles}")
        if estimator.model.training:
            raise AssertionError("L2CS model was not switched to evaluation mode")
        sources = {}
        for package in ("l2cs", "face_detection"):
            origin = metadata.distribution(package).read_text("direct_url.json")
            sources[package] = json.loads(origin) if origin else None
        result = {"synthetic": True, "pretrained": False, "accuracy_measured": False,
                  "architecture": "ResNet50", "yaw_rad": angles.yaw_rad,
                  "pitch_rad": angles.pitch_rad, "sources": sources,
                  "manifest": environment_manifest([weights])}
        Path("/tmp/gaze-smoke-report.json").write_text(json.dumps(result, indent=2))
        print("L2CS CPU adapter smoke passed with synthetic weights; no gaze accuracy claim.")


if __name__ == "__main__":
    run()
