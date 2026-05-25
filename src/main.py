"""Ball-in-play tracking for one soccer or tennis video.

CLI examples:
    # Soccer: player detections + ViTPose-gated ball selection
    uv run sam3-ball-track examples/videos/soccer/clip-1.mp4 --sport soccer

    # Tennis: player detections + motion-gated ball selection
    uv run sam3-ball-track examples/videos/tennis/clip-1.mp4 --sport tennis

Python example:
    from main import run_tracking

    result = run_tracking(
        input_video="examples/videos/soccer/clip-1.mp4",
        sport="soccer",
    )
    print(result.result_dir)

Model access:
    SAM3 weights are downloaded through Hugging Face Hub on first use. Request
    access to facebook/sam3, run hf auth login once, then run the CLI.
    ViTPose is loaded from the public usyd-community/vitpose-plus-base model.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np

from ball.render import render_ball_candidates, render_selected_ball
from ball.select import select_match_ball
from ball.track import track_ball_candidates, trim_drift_killed_tracks
from utils.artifacts import save_artifact
from utils.video import Video
from vision.players import detect_player_bboxes, mask_centroid_inside_any_bbox
from vision.pose import estimate_poses


@dataclass(frozen=True)
class TrackingResult:
    result_dir: Path
    ball_in_play_video: Path
    all_balls_video: Path
    player_bboxes: dict[int, dict[int, np.ndarray]]
    pose_data: dict[int, dict[int, dict]]
    ball_tracks: dict[int, dict[int, np.ndarray]]
    ball_drift_kills: list
    ball_masks: dict[int, dict]


def run_tracking(
    input_video: str | Path,
    sport: str,
    player_confidence: float = 0.5,
) -> TrackingResult:
    sport_module = _sport_module(sport)
    video = Video.open(input_video)
    segment = video.full_segment()

    result_path = _create_result_dir(input_video, sport)
    ball_in_play_video = result_path / "ball-in-play.mp4"
    all_balls_video = result_path / "all-balls.mp4"

    print(f"Input: {video.path}")
    print(f"Video: {video.num_frames} frames, {video.fps:.2f} fps, {video.width}x{video.height}")
    print(f"Sport: {sport}")
    print(f"Result dir: {result_path}")

    print("\nStep 1/5: detecting players")
    player_bboxes = detect_player_bboxes(video, confidence_threshold=player_confidence)
    save_artifact(player_bboxes, result_path / "player-bboxes.pkl.zst")

    print("\nStep 2/5: estimating pose" if sport_module.REQUIRES_POSE else "\nStep 2/5: skipping pose")
    if sport_module.REQUIRES_POSE:
        pose_data = estimate_poses(video, player_bboxes, segment)
    else:
        pose_data = {}
    save_artifact(pose_data, result_path / "pose-data.pkl.zst")

    def is_attached_to_player(ball_mask: np.ndarray, frame_idx: int) -> bool:
        return mask_centroid_inside_any_bbox(ball_mask, frame_idx, player_bboxes)

    print("\nStep 3/5: tracking ball candidates")
    ball_tracks, drift_kills = track_ball_candidates(
        video,
        segment,
        is_attached_to_player=is_attached_to_player,
    )
    save_artifact(ball_tracks, result_path / "ball-tracks.pkl.zst")
    save_artifact(drift_kills, result_path / "ball-drift-kills.pkl.zst")

    print("\nStep 4/5: selecting ball in play")
    selection_tracks = trim_drift_killed_tracks(ball_tracks, drift_kills)
    ball_masks = select_match_ball(
        selection_tracks,
        pose_data,
        video.num_frames,
        compute_changepoints=sport_module.compute_changepoints,
    )
    save_artifact(ball_masks, result_path / "ball-in-play-masks.pkl.zst")

    print("\nStep 5/5: rendering outputs")
    render_ball_candidates(video, segment, str(all_balls_video), ball_tracks, drift_kills)
    render_selected_ball(video, segment, str(ball_in_play_video), ball_masks, ball_tracks)

    return TrackingResult(
        result_dir=result_path,
        ball_in_play_video=ball_in_play_video,
        all_balls_video=all_balls_video,
        player_bboxes=player_bboxes,
        pose_data=pose_data,
        ball_tracks=ball_tracks,
        ball_drift_kills=drift_kills,
        ball_masks=ball_masks,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sam3-ball-track",
        description="Track the ball in play in soccer or tennis broadcast video.",
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--sport", choices=["soccer", "tennis"], required=True)
    parser.add_argument(
        "--player-confidence",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold for player detections.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_tracking(
        input_video=args.input,
        sport=args.sport,
        player_confidence=args.player_confidence,
    )


def _create_result_dir(input_video: str | Path, sport: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    clip_name = _slug(Path(input_video).stem)
    result_path = Path("results") / f"{timestamp}-{sport}-{clip_name}"
    result_path.mkdir(parents=True, exist_ok=False)
    return result_path


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Input video must have a non-empty file stem")
    return slug


def _sport_module(sport: str):
    if sport == "soccer":
        from sports import soccer

        return soccer
    if sport == "tennis":
        from sports import tennis

        return tennis
    raise ValueError(f"Unsupported sport: {sport}")


if __name__ == "__main__":
    main()
