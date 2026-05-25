from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.video import Video


def detect(
    video: Video,
    prompt: str,
    confidence_threshold: float = 0.5,
    keep_masks: bool = False,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, List[dict]]]]:
    from sam3.model.sam3_image_processor import Sam3Processor
    from vision.models import get_sam3_image_model

    model = get_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    detections: Dict[int, np.ndarray] = {}
    masks: Optional[Dict[int, List[dict]]] = {} if keep_masks else None

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame_idx in tqdm(range(video.num_frames), desc=f"SAM3 detect '{prompt}'"):
            rgb = video.decoder[frame_idx].numpy()
            image = Image.fromarray(rgb)
            state = processor.set_image(image)
            state = processor.set_text_prompt(state=state, prompt=prompt)

            boxes = state["boxes"].float().cpu().numpy()
            scores = state["scores"].float().cpu().numpy()

            if keep_masks and masks is not None:
                raw_masks = state["masks"].squeeze(1).cpu().numpy()
                frame_masks = []
                for i in range(len(scores)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    frame_masks.append({
                        "mask": raw_masks[i, y1:y2, x1:x2].copy(),
                        "box": boxes[i],
                        "score": float(scores[i]),
                    })
                masks[frame_idx] = frame_masks

            detections[frame_idx] = (
                np.column_stack([boxes[:len(scores)], scores])
                if len(scores)
                else np.empty((0, 5))
            )

    total_dets = sum(len(d) for d in detections.values())
    avg_dets = total_dets / len(detections) if detections else 0
    print(f"SAM3 detect: {total_dets} detections across {len(detections)} frames (avg {avg_dets:.1f}/frame)")
    return detections, masks
