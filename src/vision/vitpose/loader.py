VITPOSE_MODEL = "usyd-community/vitpose-plus-base"


def load_vitpose_model() -> tuple:
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    processor = AutoProcessor.from_pretrained(VITPOSE_MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(VITPOSE_MODEL)
    model.eval()
    model = model.to("cuda")

    print(f"ViTPose loaded on {next(model.parameters()).device}.")
    return processor, model
