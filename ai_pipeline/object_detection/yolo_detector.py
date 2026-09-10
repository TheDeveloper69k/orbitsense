"""
Object detector for OrbitSense demo: locates the "water" and "acid" containers
in a camera frame and returns their bounding boxes.

TRADEOFF NOTE (read this before demo day):
--------------------------------------------------------------------------
A custom-trained YOLOv8 model needs a labeled dataset of your exact
containers under your exact lighting, which you don't have time to build
and train reliably in one night. Training on generic COCO classes won't
recognize "labeled beaker with ACID sticker" as a class at all.

For a HACKATHON DEMO, the reliable move is: stick colored tape/stickers
on the two containers (e.g. BLUE tape = water, RED tape = acid) and use
simple HSV color-range detection. It is deterministic, needs zero
training data, and will not randomly fail on stage.

This file ships BOTH:
  - ColorBasedDetector  -> use this for tomorrow's demo (DEFAULT)
  - YoloDetector        -> stub using ultralytics YOLOv8, wire in later
                            once you have a trained model / more time

Swap which one `pipeline.py` uses via the DETECTOR_MODE constant below.
--------------------------------------------------------------------------
"""

import cv2
import numpy as np

# "color" = reliable demo mode (recommended for tomorrow)
# "yolo"  = use a trained YOLOv8 model (only if you have one ready)
DETECTOR_MODE = "color"

# HSV color ranges — TUNE THESE to your actual tape/sticker colors.
# Test with a quick script that prints HSV values from your webcam first.
COLOR_RANGES = {
    "water": {
        "lower": np.array([100, 120, 70]),   # blue range
        "upper": np.array([130, 255, 255]),
    },
    "acid": {
        "lower": np.array([0, 120, 70]),     # red range (lower hue band)
        "upper": np.array([10, 255, 255]),
    },
}

MIN_CONTOUR_AREA = 800  # ignore tiny color specks/noise


class ColorBasedDetector:
    """Detects labeled containers via HSV color thresholding. Reliable, no training needed."""

    def __init__(self, color_ranges=None):
        self.color_ranges = color_ranges or COLOR_RANGES

    def detect(self, frame):
        """
        Returns a list of detections:
        [{"label": "water", "bbox": (x, y, w, h), "confidence": 1.0}, ...]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []

        for label, bounds in self.color_ranges.items():
            mask = cv2.inRange(hsv, bounds["lower"], bounds["upper"])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(largest)
            detections.append({"label": label, "bbox": (x, y, w, h), "confidence": 1.0})

        return detections


class YoloDetector:
    """
    Optional upgrade path: real YOLOv8 object detection via Ultralytics.
    Only use this if you have a trained model checkpoint (.pt file) for
    "water" and "acid" classes. Falls back gracefully if not available.
    """

    def __init__(self, model_path="ai_pipeline/activity_classifier/weights/orbitsense_yolo.pt"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
        except Exception as e:
            print(f"[YoloDetector] Model not available ({e}). Falling back to color detection.")
            self.model = None
            self.available = False
            self._fallback = ColorBasedDetector()

    def detect(self, frame):
        if not self.available:
            return self._fallback.detect(frame)

        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "confidence": conf,
            })
        return detections


def get_detector():
    """Factory — returns the active detector based on DETECTOR_MODE."""
    if DETECTOR_MODE == "yolo":
        return YoloDetector()
    return ColorBasedDetector()


if __name__ == "__main__":
    # Quick manual test using your webcam. Press 'q' to quit.
    detector = get_detector()
    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for det in detector.detect(frame):
            x, y, w, h = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, det["label"], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("OrbitSense - Object Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()