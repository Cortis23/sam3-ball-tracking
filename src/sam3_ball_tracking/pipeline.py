from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from sam3_ball_tracking.artifacts import save_pickle
from sam3_ball_tracking.ball.render import render_ball_candidates, render_selected_ball
from sam3_ball_tracking.ball.select import select_match_ball
from sam3_ball_tracking.ball.track import track_ball_candidates, trim_drift_killed_tracks
from sam3_ball_tracking.players import detect_player_bboxes, mask_centroid_inside_any_bbox
from sam3_ball_tracking.pose import estimate_poses
from sam3_ball_tracking.video import Video


@dataclass(frozen=True)
class TrackingResult:
    player_bboxes: Dict[int, Dict[int, np.ndarray]]
    pose_data: Dict[int, Dict[int, dict]]
    ball_tracks: Dict[int, Dict[int, np.ndarray]]
    ball_drift_kills: list
    ball_masks: Dict[int, dict]


def run_tracking(
    input_video: str | Path,
    sport: str,
    output: str | Path,
    debug_dir: str | Path | None = None,
    player_confidence: float = 0.5,
) -> TrackingResult:
    """Run the public single-video tracking pipeline."""
    sport_module = _sport_module(sport)
    video = Video.open(input_video)
    segment = video.full_segment()

    if debug_dir is None:
        debug_dir = Path(output).with_suffix("")
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    print(f"Input: {video.path}")
    print(f"Video: {video.num_frames} frames, {video.fps:.2f} fps, {video.width}x{video.height}")
    print(f"Sport: {sport}")

    print("\nStep 1/5: detecting players")
    player_bboxes = detect_player_bboxes(video, confidence_threshold=player_confidence)
    save_pickle(player_bboxes, debug_path / "player_bboxes.pkl")

    print("\nStep 2/5: estimating pose" if sport_module.REQUIRES_POSE else "\nStep 2/5: skipping pose")
    if sport_module.REQUIRES_POSE:
        pose_data = estimate_poses(video, player_bboxes, segment)
    else:
        pose_data = {}
    save_pickle(pose_data, debug_path / "pose_data.pkl")

    def is_attached_to_player(ball_mask: np.ndarray, frame_idx: int) -> bool:
        return mask_centroid_inside_any_bbox(ball_mask, frame_idx, player_bboxes)

    print("\nStep 3/5: tracking ball candidates")
    ball_tracks, drift_kills = track_ball_candidates(
        video,
        segment,
        is_attached_to_player=is_attached_to_player,
    )
    save_pickle(ball_tracks, debug_path / "ball_tracks.pkl")
    save_pickle(drift_kills, debug_path / "ball_drift_kills.pkl")

    print("\nStep 4/5: selecting ball in play")
    selection_tracks = trim_drift_killed_tracks(ball_tracks, drift_kills)
    ball_masks = select_match_ball(
        selection_tracks,
        pose_data,
        video.num_frames,
        compute_changepoints=sport_module.compute_changepoints,
    )
    save_pickle(ball_masks, debug_path / "ball_masks.pkl")

    print("\nStep 5/5: rendering outputs")
    render_ball_candidates(video, segment, str(debug_path / "all_candidates.mp4"), ball_tracks, drift_kills)
    render_selected_ball(video, segment, str(output), ball_masks, ball_tracks)

    return TrackingResult(
        player_bboxes=player_bboxes,
        pose_data=pose_data,
        ball_tracks=ball_tracks,
        ball_drift_kills=drift_kills,
        ball_masks=ball_masks,
    )


def _sport_module(sport: str):
    if sport == "soccer":
        from sam3_ball_tracking.sports import soccer

        return soccer
    if sport == "tennis":
        from sam3_ball_tracking.sports import tennis

        return tennis
    raise ValueError(f"Unsupported sport: {sport}")
