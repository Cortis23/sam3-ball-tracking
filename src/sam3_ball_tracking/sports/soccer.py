"""Soccer-specific ball-in-play scoring."""

from typing import Dict

import numpy as np

ANGLE_THRESHOLD = 30.0
CONTACT_THRESHOLD = 10.0
KP_CONFIDENCE = 0.3
REQUIRES_POSE = True


def compute_changepoints(
    centroids: Dict[int, np.ndarray],
    track_masks: Dict[int, np.ndarray],
    pose_data: Dict[int, Dict[int, dict]],
    nms_window: int = 3,
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
        if s_in < 1e-6 or s_out < 1e-6:
            continue
        cos = np.clip(np.dot(v_in, v_out) / (s_in * s_out), -1.0, 1.0)
        angles[f] = float(np.degrees(np.arccos(cos)))

    above = {f: a for f, a in angles.items() if a >= ANGLE_THRESHOLD}

    gated: Dict[int, float] = {}
    for f, a in above.items():
        mask = track_masks.get(f)
        if mask is None:
            continue
        if _min_mask_kp_dist(mask, pose_data, f) <= CONTACT_THRESHOLD:
            gated[f] = a

    peaks: Dict[int, float] = {}
    for f, a in gated.items():
        if all(gated.get(f2, 0) <= a for f2 in range(f - nms_window, f + nms_window + 1) if f2 != f):
            peaks[f] = a

    return peaks


def _min_mask_kp_dist(mask: np.ndarray, pose_data: Dict[int, Dict[int, dict]], frame_idx: int) -> float:
    if not mask.any():
        return float("inf")
    ys, xs = np.where(mask)
    mask_coords = np.stack([xs, ys], axis=1).astype(np.float32)
    best = float("inf")
    for pose in pose_data.get(frame_idx, {}).values():
        kps = pose["keypoints"]
        scores = pose["scores"]
        for idx in range(len(scores)):
            if scores[idx] < KP_CONFIDENCE:
                continue
            kp = kps[idx][:2].astype(np.float32)
            d = float(np.linalg.norm(mask_coords - kp, axis=1).min())
            if d < best:
                best = d
    return best
