import torch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_autocast_context():
    device = get_device()

    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )


    if device.type == "mps":
        return torch.autocast(
            device_type="mps",
            dtype=torch.float16,
        )

    return torch.autocast(
        device_type="cpu",
        enabled=False,
    )