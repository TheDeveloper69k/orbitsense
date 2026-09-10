#mediapipe_pose.py

"""
Hand tracking + full-body pose tracking for OrbitSense demo.

HandTracker uses MediaPipe Hands to find hand landmarks, then checks
proximity between the hand (index fingertip + wrist) and each detected
object's bounding box to figure out which container is being picked
up / poured.

PoseTracker uses MediaPipe Pose to draw the full-body skeleton on the
live monitoring feed (does not affect object-pickup detection logic).
"""

import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# How close (in pixels) the hand needs to be to an object's bbox center
# to count as "holding" it. Tune this based on your camera distance.
PROXIMITY_THRESHOLD_PX = 120


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def get_hand_points(self, frame):
        """
        Returns a list of (x, y) pixel coordinates for detected hand
        keypoints (wrist + index fingertip) per hand found in the frame.
        Empty list if no hand detected.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hand_points = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                hand_points.append({
                    "wrist": (int(wrist.x * w), int(wrist.y * h)),
                    "index_tip": (int(index_tip.x * w), int(index_tip.y * h)),
                    "landmarks": hand_landmarks,  # kept for optional drawing
                })

        return hand_points

    def draw_landmarks(self, frame, hand_points):
        """Optional: draw hand skeleton on frame for the live dashboard feed."""
        for hp in hand_points:
            mp_drawing.draw_landmarks(frame, hp["landmarks"], mp_hands.HAND_CONNECTIONS)
        return frame


class PoseTracker:
    """
    Tracks full-body pose landmarks using MediaPipe Pose and draws the
    body skeleton on the frame. Separate from HandTracker — does not
    feed into find_picked_object, only used for the visual overlay.
    """

    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._last_results = None

    def get_pose_points(self, frame):
        """
        Runs pose detection on a BGR frame and returns a list of
        (x, y, visibility) tuples for the 33 body landmarks, or an
        empty list if no person is detected.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        self._last_results = results

        points = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                points.append((int(lm.x * w), int(lm.y * h), lm.visibility))

        return points

    def draw_landmarks(self, frame, pose_points=None):
        """Draws the last-detected body skeleton onto the frame."""
        if self._last_results and self._last_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                self._last_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame


def _bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def find_picked_object(hand_points, detections, threshold=PROXIMITY_THRESHOLD_PX):
    """
    Given hand points and detected objects, returns the label of the object
    currently being picked up/held (closest object within threshold), or
    None if no hand is close enough to any object.
    """
    if not hand_points or not detections:
        return None

    closest_label = None
    closest_dist = float("inf")

    for hp in hand_points:
        hand_ref = hp["index_tip"]  # fingertip is the most reliable "pointing at" signal
        for det in detections:
            center = _bbox_center(det["bbox"])
            dist = _distance(hand_ref, center)
            if dist < threshold and dist < closest_dist:
                closest_dist = dist
                closest_label = det["label"]

    return closest_label


if __name__ == "__main__":
    # Quick manual test using your webcam. Press 'q' to quit.
    from ai_pipeline.object_detection.yolo_detector import get_detector

    tracker = HandTracker()
    pose_tracker = PoseTracker()
    detector = get_detector()
    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hand_points = tracker.get_hand_points(frame)
        pose_points = pose_tracker.get_pose_points(frame)
        detections = detector.detect(frame)
        picked = find_picked_object(hand_points, detections)

        frame = pose_tracker.draw_landmarks(frame, pose_points)  # body first
        frame = tracker.draw_landmarks(frame, hand_points)       # hands on top
        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if picked:
            cv2.putText(frame, f"Picked up: {picked}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("OrbitSense - Hand Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()