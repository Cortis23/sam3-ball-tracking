from typing import Dict, Optional

import cv2
import numpy as np

from ball.changepoints import compute_centroids
from ball.positions import derive_ball_positions
from utils.rendering import render_frames
from utils.video import Segment, Video
from vision.players import PlayerMasks

BALL_COLOR = (0, 255, 255)
TRACK_COLORS = [
    (0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255),
    (0, 165, 255), (255, 255, 0), (128, 0, 255), (0, 200, 200),
    (200, 0, 130), (100, 200, 0), (255, 100, 0),
]
TRAIL_LENGTH = 100
PLAYER_MASK_COLORS = [
    (0, 200, 100), (255, 120, 0), (180, 80, 255), (80, 180, 255),
    (120, 220, 40), (255, 80, 160), (40, 220, 220), (180, 180, 40),
]


def render_selected_ball(
    video: Video,
    segment: Segment,
    output_path: str,
    ball_masks: Dict[int, dict],
    tracks: Dict[int, Dict[int, np.ndarray]],
) -> None:
    ball_positions = derive_ball_positions(ball_masks)
    trail_length = 30

    track_ids = sorted(tracks.keys())
    frame_color: Dict[int, tuple] = {}
    for f, entry in ball_masks.items():
        bm = entry["mask"]
        for i, tid in enumerate(track_ids):
            tm = tracks[tid].get(f)
            if tm is not None and np.array_equal(bm, tm):
                frame_color[f] = TRACK_COLORS[i % len(TRACK_COLORS)]
                break

    def _color_at(f: int) -> tuple:
        return frame_color.get(f, BALL_COLOR)

    def draw_frame(frame: np.ndarray, frame_idx: int) -> None:
        trail_start = max(0, frame_idx - trail_length)
        points = []
        for t in range(trail_start, frame_idx + 1):
            pos = ball_positions.get(t)
            if pos is not None:
                points.append((t, int(pos[0]), int(pos[1])))
        for i in range(1, len(points)):
            alpha = (points[i][0] - trail_start) / trail_length
            c = tuple(int(v * alpha) for v in _color_at(points[i][0]))
            cv2.line(frame, (points[i - 1][1], points[i - 1][2]), (points[i][1], points[i][2]), c, 2, cv2.LINE_AA)

        _draw_ball_mask(frame, ball_masks.get(frame_idx), ball_positions.get(frame_idx), _color_at(frame_idx))
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    render_frames(video, output_path, draw_frame, desc="Rendering selected ball", segment=segment)


def render_ball_candidates(
    video: Video,
    segment: Segment,
    output_path: str,
    tracks: Dict[int, Dict[int, np.ndarray]],
    drift_kills: list,
    player_masks: Optional[PlayerMasks] = None,
) -> None:
    centroids = compute_centroids(tracks)
    track_ids = sorted(tracks.keys())

    drift_ranges: Dict[int, tuple] = {}
    for kill in drift_kills:
        if kill["reason"] == "drift_kill" and "drift_onset" in kill:
            drift_ranges[kill["obj_id"]] = (kill["drift_onset"], kill["frame_idx"])

    width = video.width

    def _track_color(tid: int, frame_idx: int, i: int) -> tuple:
        r = drift_ranges.get(tid)
        if r is not None and r[0] <= frame_idx <= r[1]:
            return (0, 0, 255)
        return TRACK_COLORS[i % len(TRACK_COLORS)]

    def draw_frame(frame: np.ndarray, frame_idx: int) -> None:
        _draw_player_masks(frame, player_masks, frame_idx)

        overlay = frame.copy()
        for i, tid in enumerate(track_ids):
            mask = tracks[tid].get(frame_idx)
            if mask is not None and mask.any():
                color = _track_color(tid, frame_idx, i)
                overlay[mask] = color
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 0), 1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        for i, tid in enumerate(track_ids):
            trail_start = frame_idx - TRAIL_LENGTH
            prev = None
            for f in range(max(0, trail_start), frame_idx + 1):
                curr = centroids[tid].get(f)
                if prev is not None and curr is not None and f - prev[0] == 1:
                    color = _track_color(tid, f, i)
                    cv2.line(frame, (int(prev[1][0]), int(prev[1][1])), (int(curr[0]), int(curr[1])), color, 2, cv2.LINE_AA)
                if curr is not None:
                    prev = (f, curr)
                else:
                    prev = None

        cv2.putText(frame, f"Frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        y = 60
        for i, tid in enumerate(track_ids):
            color = _track_color(tid, frame_idx, i)
            alive = frame_idx in tracks[tid]
            label = f"T{tid} ({len(tracks[tid])}f)" + (" *" if alive else "")
            cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            y += 22

        kill_y = 30
        for kill in drift_kills:
            kill_frame = kill["frame_idx"]
            if kill_frame <= frame_idx < kill_frame + 60:
                label = f"{kill['reason'].upper()} T{kill['obj_id']} at f{kill_frame}"
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(frame, label, (width - tw - 20, kill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                kill_y += 25

    render_frames(video, output_path, draw_frame, desc="Rendering ball candidates", segment=segment)


def _draw_player_masks(
    frame: np.ndarray,
    player_masks: Optional[PlayerMasks],
    frame_idx: int,
) -> None:
    if player_masks is None:
        return

    detections = player_masks.get(frame_idx, [])
    if not detections:
        return

    overlay = frame.copy()
    has_mask = False
    for i, det in enumerate(detections):
        cropped = np.asarray(det["mask"]).astype(bool)
        if cropped.size == 0:
            continue

        x1, y1, _, _ = det["box"].astype(int)
        h, w = cropped.shape[:2]
        x2 = x1 + w
        y2 = y1 + h

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(frame.shape[1], x2)
        dst_y2 = min(frame.shape[0], y2)
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            continue

        src_x1 = dst_x1 - x1
        src_y1 = dst_y1 - y1
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        mask = cropped[src_y1:src_y2, src_x1:src_x2]
        if not mask.any():
            continue

        color = PLAYER_MASK_COLORS[i % len(PLAYER_MASK_COLORS)]
        overlay[dst_y1:dst_y2, dst_x1:dst_x2][mask] = color
        has_mask = True

    if has_mask:
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)


def _draw_ball_mask(
    frame: np.ndarray,
    entry: Optional[dict],
    ball_pos: Optional[np.ndarray],
    color: tuple,
) -> None:
    if entry is not None:
        mask = entry["mask"]
        if mask.any():
            overlay = frame.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 0, 0), 1)
    elif ball_pos is not None:
        bx, by = int(ball_pos[0]), int(ball_pos[1])
        cv2.circle(frame, (bx, by), 8, color, -1)
        cv2.circle(frame, (bx, by), 8, (0, 0, 0), 1)
