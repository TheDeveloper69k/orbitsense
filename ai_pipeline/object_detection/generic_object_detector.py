"""
Generic object detector for OrbitSense.

Uses YOLOv8 COCO for normal objects and a custom visual detector
for the physical tape roll.

YOLO detects:
    bottle, cup, cell phone, scissors, etc.

Custom detector detects:
    tape

The tape detector does NOT depend on the tape being blue.
It looks for the circular/elliptical shape and center hole.
"""

import cv2
import numpy as np


# ------------------------------------------------------------
# YOLO SETTINGS
# ------------------------------------------------------------

RELEVANT_LABELS = {
    "bottle",
    "cup",
    "wine glass",
    "bowl",
    "scissors",
    "cell phone",
    "remote",
    "keyboard",
    "book",
    "spoon",
    "fork",
    "knife",
    "vase",
    "clock",
}

CONFIDENCE_THRESHOLD = 0.35


class GenericObjectDetector:

    def __init__(
        self,
        model_name="yolov8n.pt",
        filter_relevant=True
    ):
        from ultralytics import YOLO

        print("Loading YOLOv8...")
        self.model = YOLO(model_name)
        print("Model ready.")

        self.filter_relevant = filter_relevant

    # ========================================================
    # MAIN DETECTOR
    # ========================================================

    def detect(self, frame):
        """
        Detect normal COCO objects + physical tape.

        Returns:

        [
            {
                "label": "bottle",
                "bbox": (x, y, w, h),
                "confidence": 0.82
            },
            {
                "label": "tape",
                "bbox": (x, y, w, h),
                "confidence": 0.85
            }
        ]
        """

        detections = []

        # ----------------------------------------------------
        # 1. YOLO OBJECT DETECTION
        # ----------------------------------------------------

        results = self.model(
            frame,
            verbose=False
        )[0]

        for box in results.boxes:

            conf = float(box.conf[0])

            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            # Ignore irrelevant COCO objects.
            if (
                self.filter_relevant
                and label not in RELEVANT_LABELS
            ):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "label": label,
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2 - x1),
                        int(y2 - y1)
                    ),
                    "confidence": conf,
                }
            )

        # ----------------------------------------------------
        # 2. CUSTOM TAPE DETECTION
        # ----------------------------------------------------

        tape_detection = self._detect_tape(frame)

        if tape_detection is not None:
            detections.append(tape_detection)

        return detections

    # ========================================================
    # TAPE DETECTOR
    # ========================================================

    def _detect_tape(self, frame):
        """
        Detect a physical roll of tape.

        The tape is detected using:
            - circular/elliptical shape
            - center hole
            - edge structure

        This intentionally does NOT depend on tape color.
        """

        if frame is None or frame.size == 0:
            return None

        frame_height, frame_width = frame.shape[:2]

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (7, 7),
            1.5
        )

        # ----------------------------------------------------
        # Detect circles
        # ----------------------------------------------------

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=100,
            param2=30,
            minRadius=25,
            maxRadius=100
        )

        if circles is None:
            return None

        circles = np.round(
            circles[0]
        ).astype(int)

        candidates = []

        # ----------------------------------------------------
        # Examine every detected circle
        # ----------------------------------------------------

        for cx, cy, radius in circles:

            # Make sure circle is inside image.
            if (
                cx - radius < 0
                or cy - radius < 0
                or cx + radius >= frame_width
                or cy + radius >= frame_height
            ):
                continue

            if radius < 25 or radius > 100:
                continue

            # ------------------------------------------------
            # Center hole
            # ------------------------------------------------

            inner_radius = max(
                8,
                int(radius * 0.25)
            )

            y1 = max(
                0,
                cy - inner_radius
            )

            y2 = min(
                frame_height,
                cy + inner_radius
            )

            x1 = max(
                0,
                cx - inner_radius
            )

            x2 = min(
                frame_width,
                cx + inner_radius
            )

            center_region = gray[
                y1:y2,
                x1:x2
            ]

            if center_region.size == 0:
                continue

            center_brightness = float(
                np.mean(center_region)
            )

            # ------------------------------------------------
            # Ring region
            # ------------------------------------------------

            ring_mask = np.zeros_like(
                gray,
                dtype=np.uint8
            )

            outer_ring_radius = int(
                radius * 0.82
            )

            inner_ring_radius = int(
                radius * 0.42
            )

            cv2.circle(
                ring_mask,
                (cx, cy),
                outer_ring_radius,
                255,
                -1
            )

            cv2.circle(
                ring_mask,
                (cx, cy),
                inner_ring_radius,
                0,
                -1
            )

            ring_pixels = gray[
                ring_mask > 0
            ]

            if ring_pixels.size == 0:
                continue

            ring_brightness = float(
                np.mean(ring_pixels)
            )

            brightness_difference = abs(
                ring_brightness - center_brightness
            )

            # ------------------------------------------------
            # Edge strength
            # ------------------------------------------------

            edge_region = cv2.Canny(
                gray,
                50,
                150
            )

            circle_mask = np.zeros_like(
                gray,
                dtype=np.uint8
            )

            cv2.circle(
                circle_mask,
                (cx, cy),
                int(radius * 0.90),
                255,
                2
            )

            edge_pixels = edge_region[
                circle_mask > 0
            ]

            if edge_pixels.size == 0:
                continue

            edge_score = float(
                np.mean(edge_pixels > 0)
            )

            # ------------------------------------------------
            # Candidate score
            # ------------------------------------------------

            hole_score = min(
                brightness_difference / 60.0,
                1.0
            )

            size_score = min(
                radius / 65.0,
                1.0
            )

            shape_score = min(
                edge_score * 3.0,
                1.0
            )

            final_score = (
                hole_score * 0.50
                + shape_score * 0.30
                + size_score * 0.20
            )

            # Reject very weak candidates.
            if final_score < 0.35:
                continue

            candidates.append(
                (
                    final_score,
                    cx,
                    cy,
                    radius
                )
            )

        # ----------------------------------------------------
        # No tape found
        # ----------------------------------------------------

        if not candidates:
            return None

        # ----------------------------------------------------
        # Best candidate
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        score, cx, cy, radius = candidates[0]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x = max(
            0,
            int(cx - radius)
        )

        y = max(
            0,
            int(cy - radius)
        )

        right = min(
            frame_width,
            int(cx + radius)
        )

        bottom = min(
            frame_height,
            int(cy + radius)
        )

        w = right - x
        h = bottom - y

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = min(
            0.95,
            max(
                0.60,
                0.60 + score * 0.35
            )
        )

        return {
            "label": "tape",
            "bbox": (
                x,
                y,
                w,
                h
            ),
            "confidence": confidence
        }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    detector = GenericObjectDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        raise SystemExit

    print()
    print("OrbitSense Generic Object Detection Test")
    print("----------------------------------------")
    print("Hold the tape roll in front of the camera.")
    print("Press Q to quit.")
    print()

    while True:

        ok, frame = cap.read()

        if not ok:
            print("ERROR: Could not read webcam frame.")
            break

        detections = detector.detect(frame)

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        for det in detections:

            x, y, w, h = det["bbox"]

            label = det["label"]
            confidence = det["confidence"]

            label_text = (
                f"{label} {confidence:.2f}"
            )

            # Tape gets a different box.
            if label == "tape":
                box_color = (0, 255, 0)
                thickness = 3

                text_y = max(
                    25,
                    y - 10
                )

            else:
                box_color = (0, 200, 255)
                thickness = 2

                text_y = max(
                    25,
                    y - 10
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                thickness
            )

            cv2.putText(
                frame,
                label_text,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "OrbitSense - Generic Object Detection Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()

    print("Camera closed.")