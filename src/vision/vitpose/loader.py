import json

from safetensors.torch import load_file

from vision.vitpose.model import VitPoseConfig, VitPoseForPoseEstimation
from vision.vitpose.processor import VitPoseProcessor
from utils.weights import WEIGHTS_ROOT

WEIGHTS_PATH = WEIGHTS_ROOT / "vitpose-plus-base-wholebody.safetensors"
CONFIG_PATH = WEIGHTS_ROOT / "vitpose-plus-base-wholebody-config.json"


def load_vitpose_model() -> tuple:
    """Load ViTPose model and processor.

    Returns:
        (processor, model) tuple compatible with the transformers API.
    """
    with open(CONFIG_PATH) as f:
        config = VitPoseConfig.from_hf_dict(json.load(f))

    model = VitPoseForPoseEstimation(config)
    model.load_state_dict(load_file(str(WEIGHTS_PATH)))
    model.eval()
    model = model.to("cuda")

    print(f"ViTPose loaded on {next(model.parameters()).device}.")
    return VitPoseProcessor(), model
