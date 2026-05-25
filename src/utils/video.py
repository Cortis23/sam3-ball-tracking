from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
from torchcodec.decoders import VideoDecoder

Segment = namedtuple("Segment", ["start", "end", "label"])


@dataclass(frozen=True)
class Video:
    path: Path
    decoder: VideoDecoder
    num_frames: int
    fps: float
    width: int
    height: int

    @classmethod
    def open(cls, mp4_path: str | Path) -> "Video":
        path = Path(mp4_path).resolve()
        decoder = VideoDecoder(str(path), dimension_order="NHWC", seek_mode="exact")
        md = decoder.metadata
        return cls(
            path=path,
            decoder=decoder,
            num_frames=md.num_frames,
            fps=float(md.average_fps),
            width=md.width,
            height=md.height,
        )

    def __len__(self) -> int:
        return self.num_frames

    def full_segment(self) -> Segment:
        return Segment(0, self.num_frames, "full")

    def iter_frames(self, start: int = 0, stop: int | None = None) -> Iterator[np.ndarray]:
        if stop is None:
            stop = self.num_frames
        for i in range(start, stop):
            yield _rgb_tensor_to_bgr(self.decoder[i])

    def read_range(self, start: int, stop: int) -> np.ndarray:
        block = self.decoder.get_frames_in_range(start, stop).data
        return np.ascontiguousarray(block.numpy()[:, :, :, ::-1])

    def read_indices(self, indices: Sequence[int]) -> List[np.ndarray]:
        if not indices:
            return []
        block = self.decoder.get_frames_at(list(indices)).data
        arr = block.numpy()[:, :, :, ::-1]
        return [np.ascontiguousarray(arr[k]) for k in range(arr.shape[0])]


def _rgb_tensor_to_bgr(frame) -> np.ndarray:
    return np.ascontiguousarray(frame.numpy()[:, :, ::-1])
