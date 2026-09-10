"""
OrbitSense AI Pipeline — main entry point.

Wires together: object detection -> hand tracking -> rule engine ->
voice alerts, and produces a single JSON result per frame that the
backend (Flask-SocketIO) can broadcast to the dashboard.

Usage from backend/app.py:

    from ai_pipeline.pipeline import OrbitSensePipeline
    pipeline = OrbitSensePipeline()
    result = pipeline.process_frame(frame)   # frame = OpenCV BGR image
    if result:
        socketio.emit("activity_update", result)
"""

import cv2

from ai_pipeline.object_detection.yolo_detector import get_detector
from ai_pipeline.pose_detection.mediapipe_pose import HandTracker, find_picked_object
from ai_pipeline.experiment_validation.rule_engine import RuleEngine
from ai_pipeline.voice_alerts.tts_engine import speak


class OrbitSensePipeline:
    def __init__(self, protocol_path=None, enable_voice=True, enable_generic_objects=True):
        self.detector = get_detector()  # color-based: drives safety logic
        self.hand_tracker = HandTracker()
        self.rule_engine = RuleEngine(protocol_path) if protocol_path else RuleEngine()
        self.enable_voice = enable_voice

        # Generic YOLO object labeling (bottle, scissors, etc.) — informational
        # only, does not affect rule_engine decisions. Optional/gracefully
        # disabled if ultralytics/model isn't available.
        self.generic_detector = None
        if enable_generic_objects:
            try:
                from ai_pipeline.object_detection.generic_object_detector import GenericObjectDetector
                self.generic_detector = GenericObjectDetector()
            except Exception as e:
                print(f"[Pipeline] Generic object detection unavailable ({e}). Continuing without it.")

    def reset(self):
        """Restart the demo sequence (e.g. before a fresh run on stage)."""
        self.rule_engine.reset()

    def process_frame(self, frame):
        """
        Runs the full pipeline on a single frame.
        Returns a result dict (and speaks aloud) if a new event occurred,
        otherwise returns None.

        Result shape:
        {
            "step": int,
            "object": "water" | "acid",
            "status": "ok" | "violation",
            "message": str
        }
        """
        detections = self.detector.detect(frame)
        hand_points = self.hand_tracker.get_hand_points(frame)
        picked_object = find_picked_object(hand_points, detections)

        result = self.rule_engine.evaluate(picked_object)

        if result and self.enable_voice:
            speak(result["message"])

        return result

    def annotate_frame(self, frame):
        """
        Draws bounding boxes + hand landmarks on the frame for the live
        camera feed shown on the dashboard. Call this separately from
        process_frame if you want a clean video stream either way.

        Draws TWO kinds of boxes:
        - Color-based water/acid boxes (the ones driving safety logic)
        - Generic YOLO object labels (bottle, scissors, etc.) if enabled —
          purely informational, shown in a different color to distinguish
          them from the safety-critical boxes.
        """
        # Safety-critical detections (color-based)
        detections = self.detector.detect(frame)
        hand_points = self.hand_tracker.get_hand_points(frame)

        for det in detections:
            x, y, w, h = det["bbox"]
            color = (255, 100, 0) if det["label"] == "water" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, det["label"].upper(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Generic object labels (informational only, doesn't affect rule engine)
        if self.generic_detector:
            generic_detections = self.generic_detector.detect(frame)
            for det in generic_detections:
                x, y, w, h = det["bbox"]
                label_text = f'{det["label"]} {det["confidence"]:.2f}'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 1)
                cv2.putText(frame, label_text, (x, y + h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        frame = self.hand_tracker.draw_landmarks(frame, hand_points)
        return frame


if __name__ == "__main__":
    # Standalone test: run the full pipeline live on your webcam.
    # This is the fastest way to rehearse the demo before wiring up Flask.
    pipeline = OrbitSensePipeline()
    cap = cv2.VideoCapture(0)

    print("OrbitSense pipeline running. Press 'r' to reset demo, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pipeline.process_frame(frame)
        if result:
            print(result)

        annotated = pipeline.annotate_frame(frame)

        status_text = "VIOLATION!" if pipeline.rule_engine.violated else "Monitoring..."
        status_color = (0, 0, 255) if pipeline.rule_engine.violated else (0, 255, 0)
        cv2.putText(annotated, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("OrbitSense - Live Pipeline", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            pipeline.reset()
            print("Demo reset.")

    cap.release()
    cv2.destroyAllWindows()