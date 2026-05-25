import concurrent.futures
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3_ball_tracking.video import Segment, Video

DRIFT_KILL_LOOKBACK = 10


class LazyFrameLoader:
    """SAM3 text-grounded frame source with bounded memory."""

    PREFETCH_AHEAD = 2

    _IMG_MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16).view(3, 1, 1)
    _IMG_STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float16).view(3, 1, 1)

    def __init__(self, video: Video, segment: Segment, image_size: int):
        self._video = video
        self._start = segment.start
        self._num_frames = segment.end - segment.start
        self._image_size = image_size
        self.video_height = video.height
        self.video_width = video.width

        self._cached_idx: Optional[int] = None
        self._cached_tensor: Optional[torch.Tensor] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._futures: Dict[int, concurrent.futures.Future] = {}
        self._next_to_prefetch = 0
        self._fill_prefetch()

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx == self._cached_idx and self._cached_tensor is not None:
            return self._cached_tensor

        future = self._futures.pop(idx, None)
        cpu_tensor = future.result() if future is not None else self._load_cpu(idx)

        gpu_tensor = cpu_tensor.cuda()
        self._cached_idx = idx
        self._cached_tensor = gpu_tensor

        self._fill_prefetch()
        return gpu_tensor

    def __enter__(self) -> "LazyFrameLoader":
        return self

    def __exit__(self, *exc) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _fill_prefetch(self) -> None:
        while (
            len(self._futures) < self.PREFETCH_AHEAD
            and self._next_to_prefetch < self._num_frames
        ):
            i = self._next_to_prefetch
            self._futures[i] = self._executor.submit(self._load_cpu, i)
            self._next_to_prefetch += 1

    def _load_cpu(self, idx: int) -> torch.Tensor:
        rgb_np = self._video.decoder[self._start + idx].numpy()
        img_pil = Image.fromarray(rgb_np)
        img_np = np.array(img_pil.convert("RGB").resize((self._image_size, self._image_size)))
        img_np = img_np / 255.0
        img = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float16)
        img -= self._IMG_MEAN
        img /= self._IMG_STD
        return img


def track_ball_candidates(
    video: Video,
    segment: Segment,
    is_attached_to_player: Callable[[np.ndarray, int], bool],
    prompt: str = "ball",
) -> Tuple[Dict[int, Dict[int, np.ndarray]], List[dict]]:
    """Run SAM3 per-frame to detect and track all ball-like objects."""
    total_frames = segment.end - segment.start

    attached_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=DRIFT_KILL_LOOKBACK))
    drift_kill_obj_ids: set = set()
    prev_ka: Dict[int, int] = {}
    last_confirmed_frame: Dict[int, int] = {}
    ka_history: Dict[int, List[tuple]] = defaultdict(list)

    def drift_kill_fn(obj_id: int, frame_idx: int, keep_alive: int) -> bool:
        prev = prev_ka.get(obj_id)
        if prev is None or (keep_alive >= prev and keep_alive > -1):
            last_confirmed_frame[obj_id] = frame_idx
        prev_ka[obj_id] = keep_alive
        ka_history[obj_id].append((frame_idx, keep_alive))

        if keep_alive > -1:
            return False
        history = attached_history.get(obj_id, ())
        if len(history) < DRIFT_KILL_LOOKBACK:
            return False
        if all(history):
            drift_kill_obj_ids.add(obj_id)
            return True
        return False

    from sam3_ball_tracking.models import get_sam3_predictor

    predictor = get_sam3_predictor()
    model = predictor.model
    model.drift_kill_fn = drift_kill_fn
    model.hotstart_delay = 0

    with (
        LazyFrameLoader(video, segment, model.image_size) as frames,
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        inference_state = model.init_state(resource_path=frames)
        model.add_prompt(inference_state, frame_idx=0, text_str=prompt)

        print(f"Ball: propagating {total_frames} frames with text prompt '{prompt}'...")
        tracks: Dict[int, Dict[int, np.ndarray]] = {}
        removed_obj_ids = set()
        removals = []

        for frame_idx in tqdm(range(total_frames), desc="SAM3 ball"):
            out = model._run_single_frame_inference(inference_state, frame_idx, reverse=False)

            newly_removed = out.get("removed_obj_ids", set()) - removed_obj_ids
            for obj_id in newly_removed:
                if int(obj_id) in tracks:
                    reason = "drift_kill" if int(obj_id) in drift_kill_obj_ids else "sam3"
                    removal = {"obj_id": int(obj_id), "frame_idx": frame_idx, "reason": reason}
                    if reason == "drift_kill":
                        onset = last_confirmed_frame.get(int(obj_id), frame_idx) + 1
                        removal["drift_onset"] = onset
                        print(f"  {reason}: T{int(obj_id)} at f{frame_idx} (onset f{onset})")
                    else:
                        print(f"  {reason}: T{int(obj_id)} at f{frame_idx}")
                    removals.append(removal)
            removed_obj_ids.update(out.get("removed_obj_ids", set()))

            postprocessed = model._postprocess_output(
                inference_state,
                out,
                removed_obj_ids=removed_obj_ids,
                suppressed_obj_ids=out.get("suppressed_obj_ids"),
                unconfirmed_obj_ids=out.get("unconfirmed_obj_ids"),
            )

            obj_ids = postprocessed["out_obj_ids"]
            masks = postprocessed["out_binary_masks"]
            if hasattr(obj_ids, "tolist"):
                obj_ids = obj_ids.tolist()
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()

            for i, obj_id in enumerate(obj_ids):
                mask = masks[i].copy()
                if mask.any():
                    tracks.setdefault(obj_id, {})[frame_idx] = mask
                    attached_history[obj_id].append(is_attached_to_player(mask, frame_idx))
                else:
                    attached_history[obj_id].append(False)

    for obj_id, frames in sorted(tracks.items()):
        print(f"  Ball track {obj_id}: {len(frames)} frames ({min(frames)}-{max(frames)})")
    print(f"Ball: {len(tracks)} total tracks")

    return tracks, removals


def trim_drift_killed_tracks(
    tracks: Dict[int, Dict[int, np.ndarray]],
    drift_kills: List[dict],
) -> Dict[int, Dict[int, np.ndarray]]:
    selection_tracks = dict(tracks)
    for kill in drift_kills:
        if kill["reason"] != "drift_kill":
            continue
        tid = kill["obj_id"]
        onset = kill["drift_onset"]
        if tid in selection_tracks:
            trimmed = {f: m for f, m in selection_tracks[tid].items() if f < onset}
            if trimmed:
                selection_tracks[tid] = trimmed
            else:
                del selection_tracks[tid]
    return selection_tracks
