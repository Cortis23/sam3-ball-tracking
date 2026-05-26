import contextlib
import io
from pathlib import Path
from typing import Any, Optional

from utils.weights import resolve_sam3_checkpoint

_sam3_predictors: dict[str, Any] = {}
_sam3_image_model: Optional[Any] = None


def _sam3_bpe_path() -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def get_sam3_predictor(version: str = "sam3"):
    if version not in _sam3_predictors:
        from sam3.model_builder import build_sam3_predictor

        print(f"Loading {version.upper()} video predictor...")
        build_kwargs = {
            "checkpoint_path": resolve_sam3_checkpoint(version),
            "bpe_path": _sam3_bpe_path(),
            "version": version,
            "async_loading_frames": False,
        }
        if version == "sam3.1":
            build_kwargs["use_fa3"] = False

        if version == "sam3.1":
            with contextlib.redirect_stdout(io.StringIO()):
                _sam3_predictors[version] = build_sam3_predictor(**build_kwargs)
            _sam3_predictors[version].model.batched_grounding_batch_size = 8
            print("SAM3.1 video predictor loaded (FA3 disabled, grounding batch size 8).")
        else:
            _sam3_predictors[version] = build_sam3_predictor(**build_kwargs)
            print(f"{version.upper()} video predictor loaded.")
    return _sam3_predictors[version]


def get_sam3_image_model():
    global _sam3_image_model
    if _sam3_image_model is None:
        from sam3.model_builder import build_sam3_image_model

        print("Loading SAM3 image model...")
        _sam3_image_model = build_sam3_image_model(
            checkpoint_path=resolve_sam3_checkpoint("sam3"),
            bpe_path=_sam3_bpe_path(),
        )
        print("SAM3 image model loaded.")
    return _sam3_image_model
