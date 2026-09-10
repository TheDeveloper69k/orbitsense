"""
pipeline.py (STUB)

Replace this with your real detection/classification pipeline. The contract
app.py depends on is just:

    process_frame(frame: np.ndarray) -> dict
"""

import itertools

_counter = itertools.count()

_FAKE_RESULTS = [
    {"step": 1, "object": "gloves", "status": "ok", "message": "PPE check passed."},
    {"step": 2, "object": "acid", "status": "violation", "message": "Acid handled without gloves."},
    {"step": 3, "object": "beaker", "status": "warning", "message": "Beaker placement unstable."},
]


def process_frame(frame):
    """STUB implementation. Swap this out for your real model inference."""
    i = next(_counter) % len(_FAKE_RESULTS)
    return _FAKE_RESULTS[i]