"""
Offline text-to-speech for OrbitSense demo, using pyttsx3.

Runs speech in a background thread so it never blocks the camera/detection
loop — important because pyttsx3's runAndWait() is blocking by default,
which would freeze your video feed mid-warning otherwise.
"""

import threading
import queue
import pyttsx3


class TTSEngine:
    def __init__(self, rate=170, volume=1.0, voice_index=None):
        self._queue = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        # pyttsx3 engine must be created inside the thread that uses it
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        if self._voice_index is not None:
            voices = engine.getProperty("voices")
            if 0 <= self._voice_index < len(voices):
                engine.setProperty("voice", voices[self._voice_index].id)

        while True:
            message = self._queue.get()
            if message is None:
                break
            engine.say(message)
            engine.runAndWait()

    def speak(self, message: str):
        """Queue a message to be spoken. Non-blocking — safe to call from the main loop."""
        if message:
            self._queue.put(message)

    def stop(self):
        self._queue.put(None)


# Module-level singleton so pipeline.py can just call speak() directly
_engine_instance = None


def get_tts_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TTSEngine()
    return _engine_instance


def speak(message: str):
    """Convenience function: speak(message) from anywhere in the pipeline."""
    get_tts_engine().speak(message)


if __name__ == "__main__":
    import time
    speak("Step 1 complete. Water detected. Proceed to add acid.")
    time.sleep(4)
    speak("Warning! Acid detected before water. Stop immediately and add water first.")
    time.sleep(5)