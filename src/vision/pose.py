from typing import Dict

import numpy as np

from vision.models import get_vitpose_model
from utils.video import Segment, Video


def estimate_poses(
    video: Video,
    bboxes: Dict[int, Dict[int, np.ndarray]],
    segment: Segment,
) -> Dict[int, Dict[int, dict]]:
    from vision.vitpose.inference import estimate_all_poses

    processor, model = get_vitpose_model()
    return estimate_all_poses(video, bboxes, processor, model, segment)
