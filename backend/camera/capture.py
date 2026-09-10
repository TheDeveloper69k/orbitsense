"""
capture.py

Thin wrapper around OpenCV's VideoCapture. Keeps the camera handle in one
place so app.py doesn't have to deal with cv2 directly.
"""

import cv2
import threading


class CameraStream:
    """
    Wraps cv2.VideoCapture. Not doing threaded buffering here on purpose —
    SIH demo rigs are usually single-camera and this keeps behavior
    predictable. Swap in a threaded reader later if you need higher FPS.
    """

    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.lock = threading.Lock()

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source {self.source}. "
                f"Check that it's connected and not in use by another app."
            )
        return self

    def read_frame(self):
        """Returns a single BGR frame (numpy array), or None if the read failed."""
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call .open() first.")
        with self.lock:
            success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def encode_jpeg(self, frame, quality=80):
        """Encode a frame to JPEG bytes for streaming over Socket.IO / HTTP."""
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return None
        return buffer.tobytes()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()