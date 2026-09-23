import concurrent.futures
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.video import Segment, Video

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

    def __getitem__(self, idx) -> torch.Tensor:
        if isinstance(idx, torch.Tensor):
            if idx.ndim == 0:
                return self._get_one(int(idx.item()))
            return self._get_many(int(i) for i in idx.detach().cpu().tolist())
        if isinstance(idx, np.ndarray):
            if idx.ndim == 0:
                return self._get_one(int(idx.item()))
            return self._get_many(int(i) for i in idx.tolist())
        if isinstance(idx, slice):
            return self._get_many(range(*idx.indices(self._num_frames)))
        if isinstance(idx, (list, tuple)):
            return self._get_many(int(i) for i in idx)
        return self._get_one(int(idx))

    def __enter__(self) -> "LazyFrameLoader":
        return self

    def __exit__(self, *exc) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _get_one(self, idx: int) -> torch.Tensor:
        if idx == self._cached_idx and self._cached_tensor is not None:
            return self._cached_tensor

        future = self._futures.pop(idx, None)
        cpu_tensor = future.result() if future is not None else self._load_cpu(idx)

        gpu_tensor = cpu_tensor.cuda()
        self._cached_idx = idx
        self._cached_tensor = gpu_tensor

        self._fill_prefetch()
        return gpu_tensor

    def _get_many(self, indices) -> torch.Tensor:
        frames = [self._get_one(i) for i in indices]
        if not frames:
            return torch.empty(
                0,
                3,
                self._image_size,
                self._image_size,
                dtype=torch.float16,
                device="cuda",
            )
        return torch.stack(frames)

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
    sam_version: str = "sam3",
) -> Tuple[Dict[int, Dict[int, np.ndarray]], List[dict]]:
    """Run SAM per-frame to detect and track all ball-like objects."""
    total_frames = segment.end - segment.start

    attached_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=DRIFT_KILL_LOOKBACK))
    drift_kill_obj_ids: set[int] = set()
    prev_ka: Dict[int, int] = {}
    last_confirmed_frame: Dict[int, int] = {}
    ka_history: Dict[int, List[tuple]] = defaultdict(list)

    from vision.models import get_sam3_predictor

    predictor = get_sam3_predictor(sam_version)
    model = predictor.model
    kill_keep_alive = getattr(model, "min_trk_keep_alive", -1)

    def should_drift_kill(obj_id: int, frame_idx: int, keep_alive: Optional[int]) -> bool:
        if keep_alive is None:
            return False

        prev = prev_ka.get(obj_id)
        if prev is None or (keep_alive >= prev and keep_alive > kill_keep_alive):
            last_confirmed_frame[obj_id] = frame_idx
        prev_ka[obj_id] = keep_alive
        ka_history[obj_id].append((frame_idx, keep_alive))

        if keep_alive > kill_keep_alive:
            return False
        history = attached_history.get(obj_id, ())
        if len(history) < DRIFT_KILL_LOOKBACK:
            return False
        if all(history):
            drift_kill_obj_ids.add(obj_id)
            return True
        return False

    if sam_version == "sam3":
        model.drift_kill_fn = should_drift_kill
    model.hotstart_delay = 0

    if torch.backends.mps.is_available():
        _device = "mps"
        _dtype = torch.float16
    elif torch.cuda.is_available():
        _device = "cuda"
        _dtype = torch.bfloat16
    else:
        _device = "cpu"
        _dtype = torch.float32

    with (
        LazyFrameLoader(video, segment, model.image_size) as frames,
        torch.inference_mode(),
        torch.autocast(
            device_type=_device,
            dtype=_dtype,
            enabled=_device != "cpu",
        ),
    ):
    #    LazyFrameLoader(video, segment, model.image_size) as frames,
    #    torch.inference_mode(),
    #    torch.autocast("cuda", dtype=torch.bfloat16),
    #):
        inference_state = model.init_state(resource_path=frames)
        model.add_prompt(inference_state, frame_idx=0, text_str=prompt)

        print(f"Ball: propagating {total_frames} frames with {sam_version.upper()} text prompt '{prompt}'...")
        tracks: Dict[int, Dict[int, np.ndarray]] = {}
        removed_obj_ids: set[int] = set()
        removals = []

        for frame_idx in tqdm(range(total_frames), desc=f"{sam_version.upper()} ball"):
            out = model._run_single_frame_inference(inference_state, frame_idx, reverse=False)

            newly_removed = _int_set(out.get("removed_obj_ids", set())) - removed_obj_ids
            for obj_id in newly_removed:
                if obj_id in tracks:
                    reason = "drift_kill" if obj_id in drift_kill_obj_ids else "sam3"
                    _record_removal(removals, obj_id, frame_idx, reason, last_confirmed_frame)
            removed_obj_ids.update(_int_set(out.get("removed_obj_ids", set())))

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

            for i, raw_obj_id in enumerate(obj_ids):
                obj_id = int(raw_obj_id)
                mask = masks[i].copy()
                if mask.any():
                    tracks.setdefault(obj_id, {})[frame_idx] = mask
                    attached_history[obj_id].append(is_attached_to_player(mask, frame_idx))
                else:
                    attached_history[obj_id].append(False)

                if sam_version != "sam3" and should_drift_kill(
                    obj_id, frame_idx, _keep_alive_for_obj(inference_state, obj_id)
                ):
                    _remove_object(model, inference_state, obj_id, frame_idx)
                    removed_obj_ids.add(obj_id)
                    _record_removal(
                        removals,
                        obj_id,
                        frame_idx,
                        "drift_kill",
                        last_confirmed_frame,
                    )

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


def _int_set(values) -> set[int]:
    if values is None:
        return set()
    return {int(v) for v in values}


def _record_removal(
    removals: List[dict],
    obj_id: int,
    frame_idx: int,
    reason: str,
    last_confirmed_frame: Dict[int, int],
) -> None:
    if any(r["obj_id"] == obj_id and r["reason"] == reason for r in removals):
        return

    removal = {"obj_id": obj_id, "frame_idx": frame_idx, "reason": reason}
    if reason == "drift_kill":
        onset = last_confirmed_frame.get(obj_id, frame_idx) + 1
        removal["drift_onset"] = onset
        print(f"  {reason}: T{obj_id} at f{frame_idx} (onset f{onset})")
    else:
        print(f"  {reason}: T{obj_id} at f{frame_idx}")
    removals.append(removal)


def _keep_alive_for_obj(inference_state: dict, obj_id: int) -> Optional[int]:
    metadata = inference_state.get("tracker_metadata", {})
    rank0_metadata = metadata.get("rank0_metadata", {})
    keep_alive = rank0_metadata.get("trk_keep_alive")
    if isinstance(keep_alive, dict):
        value = keep_alive.get(obj_id)
        return int(value) if value is not None else None

    gpu_metadata = metadata.get("gpu_metadata", {})
    obj_ids = metadata.get("obj_ids_all_gpu")
    keep_alive = gpu_metadata.get("trk_keep_alive")
    if obj_ids is None or keep_alive is None:
        return None

    obj_id_list = (
        [int(v) for v in obj_ids.tolist()]
        if hasattr(obj_ids, "tolist")
        else [int(v) for v in obj_ids]
    )
    if obj_id not in obj_id_list:
        return None
    value = keep_alive[obj_id_list.index(obj_id)]
    return int(value.item()) if hasattr(value, "item") else int(value)


def _remove_object(model, inference_state: dict, obj_id: int, frame_idx: int) -> None:
    try:
        model.remove_object(
            inference_state,
            obj_id,
            frame_idx=frame_idx,
            is_user_action=False,
        )
    except TypeError:
        model.remove_object(inference_state, obj_id, is_user_action=False)
