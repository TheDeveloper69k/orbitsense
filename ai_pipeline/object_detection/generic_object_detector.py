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
    "remote",
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
        # 5. Lemon detection
        # ----------------------------------------------------

        lemon_detection = self._detect_lemon(frame)

        if lemon_detection is not None:
            raw_detections.append(lemon_detection)

        # ----------------------------------------------------
        # 6. Bottle cap detection
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
        Conservative physical tape-roll detector.

        A tape roll is treated as tape only when OpenCV finds a reasonably
        large circular/elliptical object AND a contrasting inner opening.
        This deliberately avoids using the YOLO "remote", "cell phone" or
        other COCO predictions as tape.

        The detector is intentionally conservative: a missed tape is much
        better than a remote, face, clock, or bottle cap being called tape.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=110,
            param2=52,
            minRadius=28,
            maxRadius=125
        )

        if circles is None:
            return None

        image_h, image_w = gray.shape
        edges = cv2.Canny(gray, 60, 150)

        best = None
        best_score = 0.0

        for circle in np.round(circles[0]).astype(int):
            cx, cy, radius = map(int, circle)

            if radius < 28 or radius > 125:
                continue

            if (
                cx - radius < 3
                or cy - radius < 3
                or cx + radius >= image_w - 3
                or cy + radius >= image_h - 3
            ):
                continue

            # Tape rolls are substantially larger than normal bottle caps.
            # This also prevents small circular buttons from becoming tape.
            if radius < 32:
                continue

            # ---- Inner-hole test ----------------------------------------
            inner_radius = max(7, int(radius * 0.25))

            inner_mask = np.zeros_like(gray)
            cv2.circle(inner_mask, (cx, cy), inner_radius, 255, -1)

            inner_pixels = gray[inner_mask > 0]
            if inner_pixels.size < 30:
                continue

            # ---- Tape material ring ------------------------------------
            ring_mask = np.zeros_like(gray)
            cv2.circle(
                ring_mask,
                (cx, cy),
                int(radius * 0.82),
                255,
                -1
            )
            cv2.circle(
                ring_mask,
                (cx, cy),
                int(radius * 0.43),
                0,
                -1
            )

            ring_pixels = gray[ring_mask > 0]
            if ring_pixels.size < 100:
                continue

            inner_mean = float(np.mean(inner_pixels))
            ring_mean = float(np.mean(ring_pixels))
            contrast = abs(inner_mean - ring_mean) / 255.0

            # ---- Check for circular edges around the inner hole --------
            inner_edge_mask = np.zeros_like(gray)
            cv2.circle(
                inner_edge_mask,
                (cx, cy),
                int(radius * 0.29),
                255,
                2
            )

            edge_values = edges[inner_edge_mask > 0]
            edge_density = (
                float(np.mean(edge_values > 0))
                if edge_values.size
                else 0.0
            )

            # A tape roll should have an actual opening, not just a solid
            # circular object.
            if contrast < 0.14:
                continue

            if edge_density < 0.05:
                continue

            # ---- Compactness / circle score ----------------------------
            # Hough already gives us a circle. We add a size preference so
            # tiny circles are not promoted to tape.
            size_score = min(1.0, max(0.0, (radius - 30) / 65.0))

            score = (
                contrast * 0.55
                + edge_density * 0.25
                + size_score * 0.20
            )

            if score > best_score:
                best_score = score
                best = (cx, cy, radius)

        if best is None or best_score < 0.24:
            return None

        cx, cy, radius = best

        x = max(0, int(cx - radius))
        y = max(0, int(cy - radius))
        box_w = min(int(radius * 2), image_w - x)
        box_h = min(int(radius * 2), image_h - y)

        confidence = min(
            0.96,
            0.70 + best_score * 0.35
        )

        return {
            "label": "tape",
            "bbox": (x, y, box_w, box_h),
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

    def _detect_lemon(self, frame):
        """
        Detect a real lemon using yellow color + compact blob geometry.

        This is deliberately conservative. It is meant for a bright yellow
        lemon placed reasonably close to the camera, not for every yellow
        object in the room.
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow lemon range. Saturation is kept fairly high to avoid
        # treating warm/cream wall lighting as a lemon.
        lower = np.array([18, 95, 80], dtype=np.uint8)
        upper = np.array([42, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        frame_area = frame.shape[0] * frame.shape[1]
        best = None
        best_score = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if area < 700:
                continue

            if area > frame_area * 0.18:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue

            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter)
            )

            x, y, w, h = cv2.boundingRect(contour)

            if w < 20 or h < 15:
                continue

            aspect = w / float(h)
            if aspect < 0.45 or aspect > 2.2:
                continue

            # Lemons are compact but not perfect circles.
            if circularity < 0.35:
                continue

            area_score = min(1.0, area / 12000.0)
            shape_score = min(1.0, max(0.0, circularity / 0.75))

            score = (
                area_score * 0.45
                + shape_score * 0.35
                + 0.20
            )

            if score > best_score:
                best_score = score
                best = (x, y, w, h)

        if best is None or best_score < 0.58:
            return None

        x, y, w, h = best

        confidence = min(
            0.94,
            0.68 + best_score * 0.25
        )

        return {
            "label": "lemon",
            "bbox": (x, y, w, h),
            "confidence": confidence,
        }

    # ========================================================
    # BOTTLE CAP DETECTION
    # ========================================================

    def _detect_bottle_cap(self, frame):
        """
        Conservative bottle-cap detector.

        It is intentionally restricted to small circular objects. Large
        circular objects (such as a tape roll) are rejected here.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=55,
            param1=110,
            param2=48,
            minRadius=9,
            maxRadius=28
        )

        if circles is None:
            return None

        image_h, image_w = gray.shape
        edges = cv2.Canny(gray, 60, 150)

        best_circle = None
        best_score = 0.0

        for circle in np.round(circles[0]).astype(int):
            cx, cy, radius = map(int, circle)

            if radius < 9 or radius > 28:
                continue

            if (
                cx - radius < 2
                or cy - radius < 2
                or cx + radius >= image_w - 2
                or cy + radius >= image_h - 2
            ):
                continue

            # Prefer the size of a normal bottle cap in this webcam view.
            size_score = 1.0 - abs(radius - 18) / 18.0
            size_score = max(0.0, min(1.0, size_score))

            edge_mask = np.zeros_like(gray)
            cv2.circle(
                edge_mask,
                (cx, cy),
                radius,
                255,
                2
            )

            values = edges[edge_mask > 0]
            edge_strength = (
                float(np.mean(values > 0))
                if values.size
                else 0.0
            )

            # Cap should be a compact filled object. Compare the centre
            # against the immediate outside ring to reject face/remote
            # details as much as possible.
            center_mask = np.zeros_like(gray)
            cv2.circle(
                center_mask,
                (cx, cy),
                max(3, int(radius * 0.55)),
                255,
                -1
            )

            center_mean = float(np.mean(gray[center_mask > 0]))

            ring_mask = np.zeros_like(gray)
            cv2.circle(
                ring_mask,
                (cx, cy),
                int(radius * 0.95),
                255,
                -1
            )
            cv2.circle(
                ring_mask,
                (cx, cy),
                int(radius * 0.65),
                0,
                -1
            )

            ring_values = gray[ring_mask > 0]
            if ring_values.size == 0:
                continue

            ring_mean = float(np.mean(ring_values))
            uniformity = max(
                0.0,
                1.0 - abs(center_mean - ring_mean) / 100.0
            )

            score = (
                size_score * 0.50
                + edge_strength * 0.30
                + uniformity * 0.20
            )

            if score > best_score:
                best_score = score
                best_circle = (cx, cy, radius)

        if best_circle is None or best_score < 0.63:
            return None

        cx, cy, radius = best_circle

        x = max(0, int(cx - radius))
        y = max(0, int(cy - radius))
        box_w = min(int(radius * 2), image_w - x)
        box_h = min(int(radius * 2), image_h - y)

        confidence = min(
            0.92,
            0.66 + best_score * 0.28
        )

        return {
            "label": "bottle cap",
            "bbox": (x, y, box_w, box_h),
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

            elif label == "lemon":
                box_color = (0, 255, 255)

            elif label == "remote":
                box_color = (255, 180, 0)

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