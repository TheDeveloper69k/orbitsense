"""
Generic object detector for OrbitSense demo — informational overlay only.

Uses YOLOv8's PRETRAINED weights (trained on the COCO dataset) to recognize
everyday objects like "bottle", "cup", "scissors", "cell phone", etc. This
is separate from the color-based water/acid detector in yolo_detector.py.

IMPORTANT DISTINCTION:
- yolo_detector.py's ColorBasedDetector -> drives the actual safety logic
  (water/acid sequence checking). Reliable, deterministic, demo-safe.
- This file's GenericObjectDetector -> just labels whatever tools/objects
  are visible on screen (bottle, tape, scissors, etc.) for visual richness
  on the dashboard. Does NOT feed into the rule engine.

Why two separate detectors instead of one:
COCO (the dataset YOLOv8 is pretrained on) has classes like "bottle" and
"cup" but has NO concept of "container labeled ACID" — that requires
either custom training (no time) or your own labeling scheme (the color
tape approach). So: color detection for safety-critical logic, generic
YOLO for "look, it also recognizes lab objects" wow-factor.

SETUP NOTE: the first time you run this, ultralytics will download the
yolov8n.pt weights file (~6MB) automatically. Do this tonight while you
still have internet — it will NOT download at the venue if there's no
wifi. Run this file once now to trigger the download and cache it.
"""

import cv2

# Only keep labels relevant to a lab/experiment setting, to avoid
# cluttering the screen with "person", "chair", "tv", etc. every frame.
# Full COCO class list: https://docs.ultralytics.com/datasets/detect/coco/
RELEVANT_LABELS = {
    "bottle", "cup", "wine glass", "bowl", "scissors",
    "cell phone", "remote", "keyboard", "book", "spoon",
    "fork", "knife", "vase", "clock",
}

CONFIDENCE_THRESHOLD = 0.35


class GenericObjectDetector:
    def __init__(self, model_name="yolov8n.pt", filter_relevant=True):
        from ultralytics import YOLO
        self.model = YOLO(model_name)  # auto-downloads on first run
        self.filter_relevant = filter_relevant

    def detect(self, frame):
        """
        Returns a list of detections:
        [{"label": "bottle", "bbox": (x, y, w, h), "confidence": 0.82}, ...]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            if self.filter_relevant and label not in RELEVANT_LABELS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "confidence": conf,
            })

        return detections


if __name__ == "__main__":
    # Run this once tonight to download+cache the model, and to test it.
    # Press 'q' to quit.
    print("Loading YOLOv8 (downloads weights on first run, needs internet once)...")
    detector = GenericObjectDetector()
    print("Model ready. Opening webcam...")

    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for det in detector.detect(frame):
            x, y, w, h = det["bbox"]
            label_text = f'{det["label"]} {det["confidence"]:.2f}'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("OrbitSense - Generic Object Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()