from typing import Dict, Optional

import numpy as np


def centroid_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return np.array([xs.mean(), ys.mean()])


def compute_centroids(tracks: Dict[int, Dict[int, np.ndarray]]) -> Dict[int, Dict[int, np.ndarray]]:
    centroids: Dict[int, Dict[int, np.ndarray]] = {}
    for obj_id, frame_masks in tracks.items():
        centroids[obj_id] = {}
        for frame_idx, mask in frame_masks.items():
            c = centroid_from_mask(mask)
            if c is not None:
                centroids[obj_id][frame_idx] = c
    return centroids
