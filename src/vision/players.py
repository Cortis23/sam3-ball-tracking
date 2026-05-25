from typing import Dict, List, Optional

import numpy as np

from vision.detect import detect
from utils.video import Video

PlayerBBoxes = Dict[int, Dict[int, np.ndarray]]
PlayerMasks = Dict[int, List[dict]]


def detect_players(
    video: Video,
    confidence_threshold: float = 0.5,
) -> tuple[PlayerBBoxes, PlayerMasks]:
    """Return SAM3 player boxes and cropped player masks using the pipeline artifact shape."""
    detections, masks = detect(
        video,
        prompt="player",
        confidence_threshold=confidence_threshold,
        keep_masks=True,
    )

    bboxes: PlayerBBoxes = {}
    for frame_idx, dets in detections.items():
        bboxes[frame_idx] = {
            det_id: det[:4].astype(float)
            for det_id, det in enumerate(dets)
        }
    return bboxes, masks or {}


def detect_player_bboxes(
    video: Video,
    confidence_threshold: float = 0.5,
) -> PlayerBBoxes:
    """Return per-frame player boxes as {frame: {det_id: [x1,y1,x2,y2]}}."""
    bboxes, _ = detect_players(video, confidence_threshold=confidence_threshold)
    return bboxes


def mask_centroid_inside_any_bbox(
    mask: np.ndarray,
    frame_idx: int,
    bboxes: PlayerBBoxes,
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


def player_mask_union(
    player_masks: PlayerMasks,
    frame_idx: int,
    shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Expand cropped SAM3 player masks into one full-frame boolean mask."""
    union = np.zeros(shape, dtype=bool)
    has_mask = False

    for det in player_masks.get(frame_idx, []):
        cropped = np.asarray(det["mask"]).astype(bool)
        if cropped.size == 0:
            continue

        x1, y1, x2, y2 = det["box"].astype(int)
        h, w = cropped.shape[:2]
        y2 = y1 + h
        x2 = x1 + w

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(shape[1], x2)
        dst_y2 = min(shape[0], y2)
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            continue

        src_x1 = dst_x1 - x1
        src_y1 = dst_y1 - y1
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        union[dst_y1:dst_y2, dst_x1:dst_x2] |= cropped[src_y1:src_y2, src_x1:src_x2]
        has_mask = True

    return union if has_mask else None
