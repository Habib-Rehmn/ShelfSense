"""Optional OpenCV and L2CS-Net implementations; imports stay lazy."""

from math import pi
from pathlib import Path

from .gaze import Angles


class OpenCVFaceDetector:
    """Frontal-face Haar baseline. No calibrated face confidence is available."""

    def __init__(self):
        import cv2

        self.cv2 = cv2
        self.cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        if self.cascade.empty():
            raise RuntimeError("OpenCV frontal-face cascade could not be loaded")

    def detect(self, image):
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(16, 16),
        )
        return [(float(x), float(y), float(x + w), float(y + h))
                for x, y, w, h in faces]


class L2CSEstimator:
    """Run upstream model/preprocessing directly, avoiding Pipeline.step's
    empty-face handling and separating localization from gaze inference.
    """

    def __init__(self, weights: str, device: str = "cpu", arch: str = "ResNet50"):
        if arch not in {"ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"}:
            raise ValueError("Unsupported L2CS architecture")
        import cv2
        import torch
        from l2cs.utils import getArch, prep_input_numpy

        self.cv2 = cv2
        self.torch = torch
        self.prepare = prep_input_numpy
        self.device = torch.device(f"cuda:{device}" if device.isdigit() else device)
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        self.indices = torch.arange(90, dtype=torch.float32, device=self.device)

    def estimate(self, face_bgr):
        rgb = self.cv2.cvtColor(face_bgr, self.cv2.COLOR_BGR2RGB)
        resized = self.cv2.resize(rgb, (224, 224))
        with self.torch.inference_mode():
            tensor = self.prepare(resized, self.device)
            first, second = self.model(tensor)
            # Upstream calls first output "pitch", but renders it horizontally.
            # Canonical ShelfSense yaw follows that horizontal convention.
            values = [(self.torch.sum(self.torch.softmax(logits, dim=1) * self.indices,
                                      dim=1) * 4 - 180).item() * pi / 180
                      for logits in (first, second)]
        return Angles(yaw_rad=values[0], pitch_rad=values[1])
