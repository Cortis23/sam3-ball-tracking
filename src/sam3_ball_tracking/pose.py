from typing import Dict

import numpy as np

from sam3_ball_tracking.models import get_vitpose_model
from sam3_ball_tracking.video import Segment, Video


def estimate_poses(
    video: Video,
    bboxes: Dict[int, Dict[int, np.ndarray]],
    segment: Segment,
) -> Dict[int, Dict[int, dict]]:
    from sam3_ball_tracking.vitpose.inference import estimate_all_poses

    processor, model = get_vitpose_model()
    return estimate_all_poses(video, bboxes, processor, model, segment)
