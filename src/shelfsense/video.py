"""Local constant-frame-rate video reader with explicit resource ownership."""

from math import isfinite
from .pipeline import Frame


class VideoReader:
    def __init__(self, path: str, fps_override: float | None = None):
        self.path = path
        self.fps_override = fps_override

    def __enter__(self):
        import cv2

        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video: {self.path}")
        self.fps = (self.fps_override if self.fps_override is not None
                    else self.capture.get(cv2.CAP_PROP_FPS))
        if not isfinite(self.fps) or self.fps <= 0:
            self.capture.release()
            raise ValueError("Video has no valid FPS; supply --fps")
        return self

    def __iter__(self):
        index = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            yield Frame(index, index / self.fps, image)
            index += 1

    def __exit__(self, *args):
        self.capture.release()
