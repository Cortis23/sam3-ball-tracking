# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Euclidean Distance Transform (EDT).

Original SAM3 uses a CUDA/Triton implementation. This version provides
a CPU/SciPy fallback so SAM3 can run on Apple MPS devices.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def edt_triton(data: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of
    binary images.

    This is an MPS/CPU-compatible replacement for SAM3's CUDA/Triton
    implementation.

    Args:
        data:
            Tensor of shape (B, H, W) containing binary masks.

    Returns:
        Tensor of shape (B, H, W) containing Euclidean distances.

    Equivalent to applying:
        cv2.distanceTransform(input, cv2.DIST_L2, 0)
    independently to every image in the batch.
    """

    assert data.dim() == 3, (
        f"Expected EDT input with shape (B, H, W), got {tuple(data.shape)}"
    )

    original_device = data.device

    # scipy operates on CPU / NumPy arrays.
    data_np = data.detach().to("cpu").numpy()

    # scipy.ndimage.distance_transform_edt computes distance to the
    # nearest zero, which matches the behavior expected by SAM3.
    result = np.empty(data_np.shape, dtype=np.float32)

    for i in range(data_np.shape[0]):
        result[i] = distance_transform_edt(data_np[i]).astype(
            np.float32,
            copy=False,
        )

    # Return to the device SAM3 was using (MPS, CUDA, or CPU).
    return torch.from_numpy(result).to(original_device)