from typing import Dict

import numpy as np

from sam3_ball_tracking.detect import detect
from sam3_ball_tracking.video import Video


def detect_player_bboxes(
    video: Video,
    confidence_threshold: float = 0.5,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Return per-frame player boxes as {frame: {det_id: [x1,y1,x2,y2]}}."""
    detections, _ = detect(
        video,
        prompt="player",
        confidence_threshold=confidence_threshold,
        keep_masks=False,
    )

    bboxes: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx, dets in detections.items():
        bboxes[frame_idx] = {
            det_id: det[:4].astype(float)
            for det_id, det in enumerate(dets)
        }
    return bboxes


def mask_centroid_inside_any_bbox(
    mask: np.ndarray,
    frame_idx: int,
    bboxes: Dict[int, Dict[int, np.ndarray]],
) -> bool:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return False
    cx, cy = float(xs.mean()), float(ys.mean())
    for bbox in bboxes.get(frame_idx, {}).values():
        x1, y1, x2, y2 = bbox[:4]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False
