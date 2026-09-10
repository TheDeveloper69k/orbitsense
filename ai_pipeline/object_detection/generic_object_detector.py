"""
OrbitSense Generic + Custom Object Detector

Detects:
    - bottle       -> YOLOv8 COCO
    - bottle cap   -> custom shape detection
    - tape         -> custom color/shape detection
    - water        -> custom blue container/label detection
    - acid         -> custom red container/label detection

IMPORTANT:
YOLOv8 COCO does NOT contain classes for:
    tape
    water
    acid
    bottle cap

Therefore those objects are detected using custom OpenCV logic.

The detector also uses temporal smoothing so labels do not randomly
appear/disappear from frame to frame.
"""

import cv2
import numpy as np
from collections import defaultdict


# ============================================================
# SETTINGS
# ============================================================

YOLO_MODEL = "yolov8n.pt"

# Minimum YOLO confidence
YOLO_CONFIDENCE = 0.45

# Number of frames required before a detection becomes stable
STABLE_FRAMES = 3

# Number of frames a detection can disappear before being removed
MAX_MISSING_FRAMES = 5

# Minimum area for custom color detection
MIN_COLOR_AREA = 500

# Relevant YOLO classes
RELEVANT_LABELS = {
    "bottle",
    "cup",
    "bowl",
    "scissors",
    "cell phone",
    "book",
    "spoon",
    "fork",
    "knife",
}

# Deliberately NOT including "clock".
# COCO frequently mistakes circular objects for clocks.


# ============================================================
# GENERIC OBJECT DETECTOR
# ============================================================

class GenericObjectDetector:

    def __init__(
        self,
        model_name=YOLO_MODEL,
        filter_relevant=True
    ):
        print("[OrbitSense] Loading YOLOv8...")

        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.filter_relevant = filter_relevant

        print("[OrbitSense] YOLOv8 ready.")

        # ----------------------------------------------------
        # Temporal tracking
        # ----------------------------------------------------

        self.tracks = {}

        """
        tracks structure:

        {
            "bottle": {
                "bbox": (...),
                "confidence": 0.82,
                "seen": 4,
                "missing": 0
            }
        }
        """

    # ========================================================
    # MAIN DETECTION
    # ========================================================

    def detect(self, frame):

        raw_detections = []

        # ----------------------------------------------------
        # 1. YOLO detection
        # ----------------------------------------------------

        yolo_detections = self._detect_yolo(frame)

        raw_detections.extend(yolo_detections)

        # ----------------------------------------------------
        # 2. Custom tape detection
        # ----------------------------------------------------

        tape_detection = self._detect_tape(frame)

        if tape_detection is not None:
            raw_detections.append(tape_detection)

        # ----------------------------------------------------
        # 3. Custom water detection
        # ----------------------------------------------------

        water_detection = self._detect_colored_object(
            frame,
            label="water",
            color="blue"
        )

        if water_detection is not None:
            raw_detections.append(water_detection)

        # ----------------------------------------------------
        # 4. Custom acid detection
        # ----------------------------------------------------

        acid_detection = self._detect_colored_object(
            frame,
            label="acid",
            color="red"
        )

        if acid_detection is not None:
            raw_detections.append(acid_detection)

        # ----------------------------------------------------
        # 5. Bottle cap detection
        # ----------------------------------------------------

        cap_detection = self._detect_bottle_cap(frame)

        if cap_detection is not None:
            raw_detections.append(cap_detection)

        # ----------------------------------------------------
        # 6. Temporal stabilization
        # ----------------------------------------------------

        stable_detections = self._stabilize_detections(
            raw_detections
        )

        return stable_detections

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    def _detect_yolo(self, frame):

        detections = []

        results = self.model(
            frame,
            verbose=False,
            conf=YOLO_CONFIDENCE,
            iou=0.50
        )[0]

        if results.boxes is None:
            return detections

        for box in results.boxes:

            conf = float(box.conf[0])

            if conf < YOLO_CONFIDENCE:
                continue

            cls_id = int(box.cls[0])

            label = self.model.names[cls_id]

            # ------------------------------------------------
            # Never allow YOLO to report clock
            # ------------------------------------------------

            if label == "clock":
                continue

            # ------------------------------------------------
            # Filter irrelevant COCO classes
            # ------------------------------------------------

            if self.filter_relevant and label not in RELEVANT_LABELS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            # Ignore extremely tiny detections
            if w < 20 or h < 20:
                continue

            detections.append({
                "label": label,
                "bbox": (x, y, w, h),
                "confidence": conf,
            })

        return detections

    # ========================================================
    # TAPE DETECTION
    # ========================================================

    def _detect_tape(self, frame):
        """
        Detect physical tape rolls using circular/elliptical geometry.

        This is NOT based on YOLO's 'tape' prediction.

        A tape roll normally has:
            - circular outer boundary
            - hole in the center
            - reasonably compact shape

        Returns:
            {"label": "tape", ...}
        or None
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=100,
            param2=45,
            minRadius=20,
            maxRadius=150
        )

        if circles is None:
            return None

        best = None
        best_score = 0

        h, w = gray.shape

        for circle in np.round(circles[0]).astype(int):

            cx, cy, radius = circle

            if radius < 20:
                continue

            if radius > 150:
                continue

            # Must be inside image
            if (
                cx - radius < 0
                or cy - radius < 0
                or cx + radius >= w
                or cy + radius >= h
            ):
                continue

            # ------------------------------------------------
            # Check for center hole
            # ------------------------------------------------

            inner_radius = max(5, int(radius * 0.30))

            inner_mask = np.zeros_like(gray)

            cv2.circle(
                inner_mask,
                (cx, cy),
                inner_radius,
                255,
                -1
            )

            inner_pixels = gray[inner_mask > 0]

            if len(inner_pixels) == 0:
                continue

            # Outer ring
            outer_mask = np.zeros_like(gray)

            cv2.circle(
                outer_mask,
                (cx, cy),
                int(radius * 0.85),
                255,
                -1
            )

            cv2.circle(
                outer_mask,
                (cx, cy),
                int(radius * 0.40),
                0,
                -1
            )

            outer_pixels = gray[outer_mask > 0]

            if len(outer_pixels) == 0:
                continue

            inner_std = float(np.std(inner_pixels))
            outer_std = float(np.std(outer_pixels))

            # A real hole usually has different appearance
            # from the tape material around it.
            contrast_score = min(
                1.0,
                abs(
                    float(np.mean(inner_pixels))
                    - float(np.mean(outer_pixels))
                ) / 80.0
            )

            # Reject very flat/unconvincing circles
            if contrast_score < 0.15:
                continue

            # ------------------------------------------------
            # Size score
            # ------------------------------------------------

            size_score = min(1.0, radius / 70.0)

            score = (
                contrast_score * 0.65
                + size_score * 0.35
            )

            if score > best_score:
                best_score = score
                best = (cx, cy, radius)

        if best is None:
            return None

        cx, cy, radius = best

        x = max(0, int(cx - radius))
        y = max(0, int(cy - radius))
        w_box = int(radius * 2)
        h_box = int(radius * 2)

        confidence = min(
            0.95,
            0.55 + best_score * 0.40
        )

        return {
            "label": "tape",
            "bbox": (x, y, w_box, h_box),
            "confidence": confidence,
        }

    # ========================================================
    # WATER / ACID DETECTION
    # ========================================================

    def _detect_colored_object(
        self,
        frame,
        label,
        color
    ):
        """
        Detects a strongly colored region.

        water -> blue
        acid  -> red

        This should normally correspond to the colored tape/
        sticker used on the experimental containers.
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if color == "blue":

            lower = np.array(
                [95, 110, 60],
                dtype=np.uint8
            )

            upper = np.array(
                [135, 255, 255],
                dtype=np.uint8
            )

            mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

        elif color == "red":

            lower1 = np.array(
                [0, 110, 60],
                dtype=np.uint8
            )

            upper1 = np.array(
                [10, 255, 255],
                dtype=np.uint8
            )

            lower2 = np.array(
                [170, 110, 60],
                dtype=np.uint8
            )

            upper2 = np.array(
                [179, 255, 255],
                dtype=np.uint8
            )

            mask1 = cv2.inRange(
                hsv,
                lower1,
                upper1
            )

            mask2 = cv2.inRange(
                hsv,
                lower2,
                upper2
            )

            mask = cv2.bitwise_or(
                mask1,
                mask2
            )

        else:
            return None

        # ----------------------------------------------------
        # Clean mask
        # ----------------------------------------------------

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Sort largest first
        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_COLOR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            # Reject extremely thin noise
            if w < 15 or h < 15:
                continue

            # Reject gigantic background regions
            frame_area = frame.shape[0] * frame.shape[1]

            if area > frame_area * 0.35:
                continue

            # ------------------------------------------------
            # Confidence based on area
            # ------------------------------------------------

            area_score = min(
                1.0,
                area / 10000.0
            )

            confidence = min(
                0.95,
                0.55 + area_score * 0.40
            )

            return {
                "label": label,
                "bbox": (x, y, w, h),
                "confidence": confidence,
            }

        return None

    # ========================================================
    # BOTTLE CAP DETECTION
    # ========================================================

    def _detect_bottle_cap(self, frame):
        """
        Detects a bottle cap as a small circular object.

        This is intentionally conservative because random circles
        should NOT constantly be labelled as bottle caps.
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (7, 7),
            1.5
        )

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=45,
            param1=100,
            param2=42,
            minRadius=8,
            maxRadius=45
        )

        if circles is None:
            return None

        h, w = gray.shape

        best_circle = None
        best_score = 0

        for circle in np.round(
            circles[0]
        ).astype(int):

            cx, cy, radius = circle

            if radius < 8 or radius > 45:
                continue

            if (
                cx - radius < 0
                or cy - radius < 0
                or cx + radius >= w
                or cy + radius >= h
            ):
                continue

            # Bottle caps are normally fairly small.
            size_score = 1.0 - abs(
                radius - 20
            ) / 30.0

            size_score = max(
                0.0,
                min(1.0, size_score)
            )

            # Check circular edge strength
            mask = np.zeros_like(gray)

            cv2.circle(
                mask,
                (cx, cy),
                radius,
                255,
                2
            )

            edge_strength = float(
                np.mean(
                    cv2.Canny(
                        gray,
                        50,
                        150
                    )[mask > 0]
                )
            ) / 255.0

            score = (
                size_score * 0.55
                + edge_strength * 0.45
            )

            if score > best_score:
                best_score = score
                best_circle = (
                    cx,
                    cy,
                    radius
                )

        if best_circle is None:
            return None

        # Be conservative.
        if best_score < 0.55:
            return None

        cx, cy, radius = best_circle

        x = int(cx - radius)
        y = int(cy - radius)
        box_w = int(radius * 2)
        box_h = int(radius * 2)

        confidence = min(
            0.90,
            0.55 + best_score * 0.30
        )

        return {
            "label": "bottle cap",
            "bbox": (
                x,
                y,
                box_w,
                box_h
            ),
            "confidence": confidence,
        }

    # ========================================================
    # TEMPORAL STABILIZATION
    # ========================================================

    def _stabilize_detections(
        self,
        detections
    ):
        """
        Prevents labels from flickering.

        A detection must appear repeatedly before being shown.
        """

        current = {}

        # ----------------------------------------------------
        # Process current detections
        # ----------------------------------------------------

        for det in detections:

            label = det["label"]

            if label not in current:
                current[label] = det
                continue

            # Keep highest confidence detection
            if (
                det["confidence"]
                > current[label]["confidence"]
            ):
                current[label] = det

        # ----------------------------------------------------
        # Update existing tracks
        # ----------------------------------------------------

        for label, det in current.items():

            if label not in self.tracks:

                self.tracks[label] = {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "seen": 1,
                    "missing": 0,
                }

            else:

                track = self.tracks[label]

                # Smooth bounding box
                old_bbox = track["bbox"]
                new_bbox = det["bbox"]

                smooth_bbox = tuple(
                    int(
                        old_bbox[i] * 0.65
                        + new_bbox[i] * 0.35
                    )
                    for i in range(4)
                )

                track["bbox"] = smooth_bbox

                track["confidence"] = (
                    track["confidence"] * 0.70
                    + det["confidence"] * 0.30
                )

                track["seen"] += 1
                track["missing"] = 0

        # ----------------------------------------------------
        # Increase missing counter
        # ----------------------------------------------------

        for label in list(self.tracks.keys()):

            if label not in current:

                self.tracks[label]["missing"] += 1

                if (
                    self.tracks[label]["missing"]
                    > MAX_MISSING_FRAMES
                ):
                    del self.tracks[label]

        # ----------------------------------------------------
        # Only output stable detections
        # ----------------------------------------------------

        output = []

        for label, track in self.tracks.items():

            if track["seen"] < STABLE_FRAMES:
                continue

            if track["missing"] > 2:
                continue

            output.append({
                "label": label,
                "bbox": track["bbox"],
                "confidence": round(
                    track["confidence"],
                    2
                ),
            })

        return output


# ============================================================
# STANDALONE WEBCAM TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Loading YOLOv8..."
    )

    detector = GenericObjectDetector()

    print(
        "Model ready. Opening webcam..."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        raise SystemExit

    while True:

        ok, frame = cap.read()

        if not ok:
            print(
                "ERROR: Could not read webcam frame."
            )
            break

        detections = detector.detect(
            frame
        )

        for det in detections:

            x, y, w, h = det["bbox"]

            label = det["label"]
            confidence = det["confidence"]

            # ------------------------------------------------
            # Different display colors
            # ------------------------------------------------

            if label == "bottle":
                box_color = (0, 255, 0)

            elif label == "tape":
                box_color = (255, 0, 255)

            elif label == "water":
                box_color = (255, 100, 0)

            elif label == "acid":
                box_color = (0, 0, 255)

            elif label == "bottle cap":
                box_color = (0, 255, 255)

            else:
                box_color = (0, 200, 255)

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label_text = (
                f"{label} "
                f"{confidence:.2f}"
            )

            text_y = max(
                25,
                y - 10
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
        # Status
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "OrbitSense Object Detection",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "OrbitSense - Generic Object Detection Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()