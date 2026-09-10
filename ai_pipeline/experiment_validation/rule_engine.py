"""
Rule engine for OrbitSense demo.

Loads the experiment protocol from JSON and tracks which objects have been
picked up, in what order, flagging a violation if the required sequence
is broken (e.g. acid picked up before water).
"""

import json
import os
import time

DEFAULT_PROTOCOL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "protocols", "bas_experiment_steps.json"
)


class RuleEngine:
    def __init__(self, protocol_path=DEFAULT_PROTOCOL_PATH):
        self.protocol_path = protocol_path
        self.steps = self._load_protocol()
        self.completed_objects = set()   # objects successfully validated in order
        self.violated = False
        self.last_event_time = 0
        self.cooldown_seconds = 3  # avoid re-triggering on every single frame

    def _load_protocol(self):
        with open(self.protocol_path, "r") as f:
            data = json.load(f)
        return data["steps"]

    def reset(self):
        """Call this to restart the demo for a fresh run."""
        self.completed_objects = set()
        self.violated = False
        self.last_event_time = 0

    def _find_step(self, object_label):
        for step in self.steps:
            if step["object"] == object_label:
                return step
        return None

    def evaluate(self, picked_object):
        """
        Given the object label currently picked up, returns a result dict:
        {"step": int, "object": str, "status": "ok"|"violation"|"ignored", "message": str}
        or None if nothing new to report (cooldown / no object / already validated).
        """
        if picked_object is None:
            return None

        now = time.time()
        if now - self.last_event_time < self.cooldown_seconds:
            return None  # debounce: don't spam repeated detections

        if picked_object in self.completed_objects:
            return None  # already handled this object, don't re-fire

        step = self._find_step(picked_object)
        if step is None:
            return None  # unknown object, nothing to validate

        # Check if this step requires an earlier step to have happened first
        required_first_steps = [
            s for s in self.steps
            if s["step"] < step["step"] and s.get("required_first")
        ]
        missing_prereqs = [
            s for s in required_first_steps if s["object"] not in self.completed_objects
        ]

        self.last_event_time = now

        if missing_prereqs and not step.get("required_first"):
            # Violation: this step happened before its prerequisite
            self.violated = True
            return {
                "step": step["step"],
                "object": picked_object,
                "status": "violation",
                "message": step["violation_message"],
            }

        # OK path
        self.completed_objects.add(picked_object)
        return {
            "step": step["step"],
            "object": picked_object,
            "status": "ok",
            "message": step["ok_message"],
        }


if __name__ == "__main__":
    # Quick manual test simulating a wrong-order sequence
    engine = RuleEngine()

    print(engine.evaluate("acid"))   # should be a violation (water not done yet)
    time.sleep(4)                    # clear cooldown for demo purposes
    print(engine.evaluate("water"))  # should be ok
    time.sleep(4)
    print(engine.evaluate("acid"))   # should be ok now