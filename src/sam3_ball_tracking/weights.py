import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_ROOT = Path(os.environ.get("SAM3_BALL_WEIGHTS", PACKAGE_ROOT / "weights"))

SAM3_CHECKPOINT = WEIGHTS_ROOT / "sam3.pt"
VITPOSE_WEIGHTS = WEIGHTS_ROOT / "vitpose-plus-base-wholebody.safetensors"
VITPOSE_CONFIG = WEIGHTS_ROOT / "vitpose-plus-base-wholebody-config.json"
