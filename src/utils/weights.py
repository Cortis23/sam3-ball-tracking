from huggingface_hub import hf_hub_download

SAM3_CHECKPOINTS = {
    "sam3": ("facebook/sam3", "sam3.pt"),
    "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
}


def resolve_sam3_checkpoint(version: str = "sam3") -> str:
    try:
        repo_id, filename = SAM3_CHECKPOINTS[version]
    except KeyError as exc:
        supported = ", ".join(sorted(SAM3_CHECKPOINTS))
        raise ValueError(f"Unsupported SAM version {version!r}; expected one of: {supported}") from exc
    return hf_hub_download(repo_id=repo_id, filename=filename)
