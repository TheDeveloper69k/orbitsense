"""
HSV Calibration Tool for OrbitSense.

Run this, click on your actual "water" object in the video feed, and it
will print the exact HSV range to paste into yolo_detector.py's
COLOR_RANGES. Do this under the same lighting you'll use for the demo.

Usage:
    python calibrate_color.py

Click and drag a small box over just the colored object (avoid skin/background).
Press 'q' when done to print the final range.
"""

import cv2
import numpy as np

samples = []
drawing = False
start_point = None
current_frame = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, samples, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if x2 - x1 > 2 and y2 - y1 > 2:
            roi = current_frame[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            samples.append(hsv_roi.reshape(-1, 3))
            print(f"Sampled region ({x1},{y1})-({x2},{y2}): "
                  f"{len(hsv_roi.reshape(-1, 3))} pixels captured")


def main():
    global current_frame

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Calibrate - drag box over object, press q when done")
    cv2.setMouseCallback("Calibrate - drag box over object, press q when done", mouse_callback)

    print("Drag a small rectangle over your colored object (avoid skin/background).")
    print("Sample multiple times from different angles/lighting for better accuracy.")
    print("Press 'q' when done to see the computed HSV range.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current_frame = frame.copy()

        cv2.imshow("Calibrate - drag box over object, press q when done", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("No samples captured. Run again and drag a box over the object.")
        return

    all_pixels = np.vstack(samples)
    h, s, v = all_pixels[:, 0], all_pixels[:, 1], all_pixels[:, 2]

    # Add a small margin around observed range for lighting variation
    h_margin, s_margin, v_margin = 8, 40, 40

    lower = [
        max(0, int(np.min(h)) - h_margin),
        max(0, int(np.min(s)) - s_margin),
        max(0, int(np.min(v)) - v_margin),
    ]
    upper = [
        min(179, int(np.max(h)) + h_margin),
        min(255, int(np.max(s)) + s_margin),
        min(255, int(np.max(v)) + v_margin),
    ]

    print("\n" + "=" * 60)
    print("Paste this into COLOR_RANGES in yolo_detector.py:")
    print("=" * 60)
    print(f'"lower": np.array({lower}),')
    print(f'"upper": np.array({upper}),')
    print("=" * 60)


if __name__ == "__main__":
    main()