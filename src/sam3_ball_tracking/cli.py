"""Command-line entrypoint."""

import argparse
from pathlib import Path

from sam3_ball_tracking.pipeline import run_tracking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sam3-ball-track",
        description="Track the ball in play in soccer or tennis broadcast video.",
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--sport", choices=["soccer", "tennis"], required=True)
    parser.add_argument("--output", required=True, help="Annotated selected-ball output mp4")
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Directory for all-candidates render and pickle artifacts. Defaults beside output.",
    )
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
        output=Path(args.output),
        debug_dir=args.debug_dir,
        player_confidence=args.player_confidence,
    )


if __name__ == "__main__":
    main()
