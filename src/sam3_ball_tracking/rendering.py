"""Frame-by-frame video rendering."""

import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from tqdm import tqdm

from sam3_ball_tracking.video import Segment, Video

FFMPEG_PATH = "ffmpeg"


def render_frames(
    video: Video,
    output_path: str | Path,
    draw_frame: Callable[[np.ndarray, int], None],
    desc: str = "Rendering",
    segment: Optional[Segment] = None,
) -> None:
    start = segment.start if segment is not None else 0
    stop = segment.end if segment is not None else video.num_frames
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out.parent / f"temp_{out.name}"

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video.fps,
        (video.width, video.height),
    )

    total = stop - start
    for frame_idx, frame in enumerate(
        tqdm(video.iter_frames(start, stop), total=total, desc=desc)
    ):
        draw_frame(frame, frame_idx)
        writer.write(frame)

    writer.release()
    subprocess.run(
        [
            FFMPEG_PATH,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temp_path),
            "-vcodec",
            "libx264",
            "-crf",
            "28",
            str(out),
        ],
        check=True,
    )
    temp_path.unlink()
    print(f"Rendered: {out}")
