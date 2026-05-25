from typing import Callable, Dict, List, Tuple

import numpy as np

from ball.changepoints import compute_centroids


def select_match_ball(
    tracks: Dict[int, Dict[int, np.ndarray]],
    pose_data: Dict[int, Dict[int, dict]],
    total_frames: int,
    compute_changepoints: Callable,
) -> Dict[int, dict]:
    centroids = compute_centroids(tracks)
    touches: Dict[int, List[int]] = {}
    for tid, track_masks in tracks.items():
        cp = compute_changepoints(centroids[tid], track_masks, pose_data)
        touches[tid] = sorted(cp.keys())
        print(f"  T{tid}: {len(touches[tid])} selection changepoints")

    active = _chronological_selection(tracks, touches)

    ball_masks: Dict[int, dict] = {}
    for f in range(total_frames):
        tid = active.get(f)
        if tid is not None and f in tracks[tid]:
            mask = tracks[tid][f]
            if mask.any():
                ball_masks[f] = {"mask": mask, "interpolated": False, "track_id": tid}

    print(f"  Selection: {len(ball_masks)}/{total_frames} frames covered")
    return ball_masks


def _chronological_selection(
    tracks: Dict[int, Dict[int, np.ndarray]],
    touches: Dict[int, List[int]],
) -> Dict[int, int]:
    if not tracks:
        return {}

    bounds: Dict[int, Tuple[int, int]] = {
        tid: (min(frames.keys()), max(frames.keys()))
        for tid, frames in tracks.items()
    }
    order = sorted(tracks.keys(), key=lambda tid: bounds[tid][0])

    transitions: List[Tuple[int, int]] = []
    current = None

    for tid in order:
        if not touches[tid]:
            continue

        new_start = bounds[tid][0]
        if current is None:
            current = tid
            transitions.append((new_start, tid))
            continue

        win_end = max(bounds[current][1], bounds[tid][1])
        new_in_win = sum(1 for t in touches[tid] if new_start <= t <= win_end)
        cur_in_win = sum(1 for t in touches[current] if new_start <= t <= win_end)
        if new_in_win > cur_in_win:
            current = tid
            transitions.append((new_start, tid))

    active: Dict[int, int] = {}
    for i, (from_frame, tid) in enumerate(transitions):
        next_from = transitions[i + 1][0] if i + 1 < len(transitions) else None
        to_frame = (next_from - 1) if next_from is not None else bounds[tid][1]
        to_frame = min(to_frame, bounds[tid][1])
        for f in range(from_frame, to_frame + 1):
            active[f] = tid

    return active
