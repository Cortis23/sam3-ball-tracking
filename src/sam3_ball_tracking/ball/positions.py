"""Derive per-frame ball centroids from selected ball masks."""

from typing import Dict, Optional

import numpy as np


def derive_ball_positions(ball_masks: Dict[int, dict]) -> Dict[int, Optional[np.ndarray]]:
    positions = {}
    for frame, entry in ball_masks.items():
        mask = entry["mask"]
        if mask is not None and mask.any():
            ys, xs = np.where(mask)
            positions[frame] = np.array([xs.mean(), ys.mean()])
    return positions
