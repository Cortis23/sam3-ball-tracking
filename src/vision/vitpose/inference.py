from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.video import Segment, Video

TORSO_KP_INDICES = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip

VITPOSE_MODEL = "usyd-community/vitpose-plus-base"


def _run_pose(image, boxes_xyxy: np.ndarray, processor, model) -> List[dict]:
    """Run ViTPose on an image with given bounding boxes.

    Returns list of {"keypoints": (17,2), "scores": (17,)} per box.
    """
    if len(boxes_xyxy) == 0:
        return []

    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    inputs = processor(image, boxes=[boxes_xywh], return_tensors="pt").to(model.device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
        inputs = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in inputs.items()}
        dataset_index = torch.zeros(len(boxes_xyxy), dtype=torch.long, device=model.device)
        outputs = model(**inputs, dataset_index=dataset_index)

    if outputs.heatmaps.dtype == torch.bfloat16:
        outputs.heatmaps = outputs.heatmaps.float()

    pose_results = processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])

    results = []
    for pose_result in pose_results[0]:
        results.append({
            "keypoints": pose_result["keypoints"].cpu().numpy(),
            "scores": pose_result["scores"].cpu().numpy(),
        })
    return results


def estimate_all_poses(
    video: Video,
    track_bboxes: Dict[int, Dict[int, np.ndarray]],
    processor,
    model,
    segment: Segment,
) -> Dict[int, Dict[int, dict]]:
    """Run ViTPose on track bboxes directly.

    Returns {frame_idx: {track_id: {"keypoints": (17,2), "scores": (17,)}}}.
    """
    total_frames = segment.end - segment.start
    pose_data: Dict[int, Dict[int, dict]] = {}

    for frame_idx, frame in enumerate(
        tqdm(video.iter_frames(segment.start, segment.end), total=total_frames, desc="Pose")
    ):
        frame_tracks = track_bboxes.get(frame_idx)
        if not frame_tracks:
            continue

        track_ids = list(frame_tracks.keys())
        boxes_xyxy = np.array([frame_tracks[tid] for tid in track_ids])

        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        valid = (widths > 0) & (heights > 0)

        if not valid.any():
            continue

        valid_indices = np.where(valid)[0]
        valid_boxes = boxes_xyxy[valid]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _run_pose(image, valid_boxes, processor, model)

        frame_poses: Dict[int, dict] = {}
        for i, result in zip(valid_indices, results):
            frame_poses[track_ids[i]] = result
        pose_data[frame_idx] = frame_poses

    return pose_data


def is_legible(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_confidence: float = 0.5,
) -> bool:
    """Check if all 4 torso keypoints are confident enough for OCR."""
    torso_scores = scores[TORSO_KP_INDICES]
    return bool((torso_scores > min_confidence).all())


def crop_torso(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    padding: int = 5,
    min_confidence: float = 0.3,
    min_visible: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Crop torso region defined by shoulder+hip keypoints.

    Padding on left, right, and bottom only (per Koshkina & Elder 2024).
    Returns (BGR crop, (x1, y1, x2, y2)) or (None, None).
    """
    torso_kps = keypoints[TORSO_KP_INDICES]
    torso_scores = scores[TORSO_KP_INDICES]

    visible = torso_scores > min_confidence
    if visible.sum() < min_visible:
        return None, None

    pts = torso_kps[visible]
    x1 = int(pts[:, 0].min()) - padding
    y1 = int(pts[:, 1].min())
    x2 = int(pts[:, 0].max()) + padding
    y2 = int(pts[:, 1].max()) + padding

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
