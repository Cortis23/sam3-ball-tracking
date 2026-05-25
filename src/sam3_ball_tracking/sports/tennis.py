"""Tennis-specific ball-in-play scoring."""

from typing import Dict

import numpy as np

ANGLE_THRESHOLD = 30.0
SPEED_THRESHOLD = 3.0
NMS_WINDOW = 3
REQUIRES_POSE = False


def compute_changepoints(
    centroids: Dict[int, np.ndarray],
    track_masks: Dict[int, np.ndarray],
    pose_data: Dict[int, Dict[int, dict]],
    nms_window: int = NMS_WINDOW,
) -> Dict[int, float]:
    frames = sorted(centroids.keys())
    if len(frames) < 3:
        return {}

    angles: Dict[int, float] = {}
    for i in range(1, len(frames) - 1):
        f_prev, f, f_next = frames[i - 1], frames[i], frames[i + 1]
        if f - f_prev != 1 or f_next - f != 1:
            continue
        v_in = centroids[f] - centroids[f_prev]
        v_out = centroids[f_next] - centroids[f]
        s_in = float(np.linalg.norm(v_in))
        s_out = float(np.linalg.norm(v_out))
        if s_in < SPEED_THRESHOLD or s_out < SPEED_THRESHOLD:
            continue
        cos = np.clip(np.dot(v_in, v_out) / (s_in * s_out), -1.0, 1.0)
        a = float(np.degrees(np.arccos(cos)))
        if a >= ANGLE_THRESHOLD:
            angles[f] = a

    peaks: Dict[int, float] = {}
    for f, a in angles.items():
        if all(angles.get(f2, 0) <= a for f2 in range(f - nms_window, f + nms_window + 1) if f2 != f):
            peaks[f] = a

    return peaks
