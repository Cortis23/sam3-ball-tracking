import json

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vision.vitpose.model import VitPoseConfig, VitPoseForPoseEstimation
from vision.vitpose.processor import VitPoseProcessor

VITPOSE_REPO_ID = "usyd-community/vitpose-plus-base"
VITPOSE_WEIGHTS = "model.safetensors"
VITPOSE_CONFIG = "config.json"


def load_vitpose_model() -> tuple:
    config_path = hf_hub_download(repo_id=VITPOSE_REPO_ID, filename=VITPOSE_CONFIG)
    weights_path = hf_hub_download(repo_id=VITPOSE_REPO_ID, filename=VITPOSE_WEIGHTS)

    with open(config_path) as f:
        config = VitPoseConfig.from_hf_dict(json.load(f))

    model = VitPoseForPoseEstimation(config)
    model.load_state_dict(load_file(weights_path))
    model.eval()
    model = model.to("cuda")

    print(f"ViTPose loaded on {next(model.parameters()).device}.")
    return VitPoseProcessor(), model
