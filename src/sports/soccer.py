from typing import Dict

import cv2
import numpy as np

from vision.players import PlayerMasks, player_mask_union

ANGLE_THRESHOLD = 30.0
CONTACT_THRESHOLD = 10


def compute_changepoints(
    centroids: Dict[int, np.ndarray],
    track_masks: Dict[int, np.ndarray],
    player_masks: PlayerMasks,
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
        if _touches_player_region(mask, player_masks, f):
            gated[f] = a

    peaks: Dict[int, float] = {}
    for f, a in gated.items():
        if all(gated.get(f2, 0) <= a for f2 in range(f - nms_window, f + nms_window + 1) if f2 != f):
            peaks[f] = a

    return peaks


def _touches_player_region(
    ball_mask: np.ndarray,
    player_masks: PlayerMasks,
    frame_idx: int,
) -> bool:
    if not ball_mask.any():
        return False

    players = player_mask_union(player_masks, frame_idx, ball_mask.shape)
    if players is None or not players.any():
        return False

    kernel_size = CONTACT_THRESHOLD * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    contact_region = cv2.dilate(players.astype(np.uint8), kernel).astype(bool)
    return bool(np.any(ball_mask & contact_region))
