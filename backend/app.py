"""
app.py

Flask + Flask-SocketIO server for OrbitSense.

Responsibilities:
  - Open the camera (backend/camera/capture.py)
  - On a background thread, grab frames, run them through
    ai_pipeline.pipeline.process_frame(), and:
      * emit the pipeline result on Socket.IO event "activity_update"
      * emit the raw frame (as base64 JPEG) on event "camera_frame"
      * insert a row into SQLite for any "violation" result
  - Serve REST logs via backend/routes/log_routes.py (GET /api/logs)

Run with:  python backend/app.py
(see README / instructions for full setup)
"""

import base64
import sys
import os
import time
import threading

# Allow `python backend/app.py` to resolve `backend.xxx` imports regardless
# of the working directory it's launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

from backend.camera.capture import CameraStream
from backend.routes.log_routes import log_routes
from backend.services import db_service
from ai_pipeline.pipeline import process_frame

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAMERA_SOURCE = int(os.environ.get("ORBITSENSE_CAMERA_SOURCE", 0))  # 0 = default webcam
FRAME_INTERVAL_SEC = float(os.environ.get("ORBITSENSE_FRAME_INTERVAL", 0.2))  # ~5 fps
JPEG_QUALITY = int(os.environ.get("ORBITSENSE_JPEG_QUALITY", 70))

# ---------------------------------------------------------------------------
# App / SocketIO setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("ORBITSENSE_SECRET_KEY", "orbitsense-dev-secret")

CORS(app)  # allow the frontend dev server (different port) to talk to us
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

app.register_blueprint(log_routes)

camera = CameraStream(source=CAMERA_SOURCE)

_capture_thread = None
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Background capture + pipeline loop
# ---------------------------------------------------------------------------
def capture_loop():
    """
    Runs on a background thread. Grabs a frame, runs the AI pipeline on it,
    emits the result + the frame over Socket.IO, and logs violations to SQLite.
    """
    camera.open()
    print(f"[OrbitSense] Camera opened (source={CAMERA_SOURCE}). Starting capture loop.")

    try:
        while not _stop_event.is_set():
            frame = camera.read_frame()
            if frame is None:
                print("[OrbitSense] Frame read failed, retrying...")
                time.sleep(0.5)
                continue

            # --- run the AI pipeline ---
            try:
                result = process_frame(frame)
            except Exception as e:
                print(f"[OrbitSense] process_frame() raised: {e}")
                time.sleep(FRAME_INTERVAL_SEC)
                continue

            # --- emit pipeline result to all connected clients ---
            socketio.emit("activity_update", result)

            # --- log violations to SQLite ---
            if result.get("status") == "violation":
                try:
                    db_service.insert_log(result)
                except Exception as e:
                    print(f"[OrbitSense] Failed to insert log: {e}")

            # --- stream the raw frame as base64 JPEG (optional, for live video) ---
            jpeg_bytes = camera.encode_jpeg(frame, quality=JPEG_QUALITY)
            if jpeg_bytes is not None:
                b64_frame = base64.b64encode(jpeg_bytes).decode("utf-8")
                socketio.emit("camera_frame", {"image": b64_frame})

            time.sleep(FRAME_INTERVAL_SEC)

    finally:
        camera.release()
        print("[OrbitSense] Camera released. Capture loop stopped.")


def start_capture_thread():
    global _capture_thread
    if _capture_thread is None or not _capture_thread.is_alive():
        _stop_event.clear()
        _capture_thread = threading.Thread(target=capture_loop, daemon=True)
        _capture_thread.start()


def stop_capture_thread():
    _stop_event.set()
    if _capture_thread is not None:
        _capture_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Socket.IO connection handlers
# ---------------------------------------------------------------------------
@socketio.on("connect")
def handle_connect():
    print("[OrbitSense] Client connected.")
    # Start capturing lazily on first client connection, so the camera
    # isn't held open with nobody watching.
    start_capture_thread()


@socketio.on("disconnect")
def handle_disconnect():
    print("[OrbitSense] Client disconnected.")


# ---------------------------------------------------------------------------
# Basic health check
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return {"status": "ok", "service": "OrbitSense backend"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    db_service.init_db()
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    finally:
        stop_capture_thread()