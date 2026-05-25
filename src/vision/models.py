from pathlib import Path
from typing import Any, Optional, Tuple

from utils.weights import resolve_sam3_checkpoint

_sam3_predictor: Optional[Any] = None
_sam3_image_model: Optional[Any] = None
_vitpose_model: Optional[Tuple[Any, Any]] = None


def _sam3_bpe_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def get_sam3_predictor():
    global _sam3_predictor
    if _sam3_predictor is None:
        from sam3.model_builder import build_sam3_video_predictor

        print("Loading SAM3 video predictor...")
        _sam3_predictor = build_sam3_video_predictor(
            checkpoint_path=resolve_sam3_checkpoint(),
            bpe_path=_sam3_bpe_path(),
        )
        print("SAM3 video predictor loaded.")
    return _sam3_predictor


def get_sam3_image_model():
    global _sam3_image_model
    if _sam3_image_model is None:
        from sam3.model_builder import build_sam3_image_model

        print("Loading SAM3 image model...")
        _sam3_image_model = build_sam3_image_model(
            checkpoint_path=resolve_sam3_checkpoint(),
            bpe_path=_sam3_bpe_path(),
        )
        print("SAM3 image model loaded.")
    return _sam3_image_model


def get_vitpose_model():
    global _vitpose_model
    if _vitpose_model is None:
        from vision.vitpose.loader import load_vitpose_model

        _vitpose_model = load_vitpose_model()
    return _vitpose_model
