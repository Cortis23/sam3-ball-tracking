from huggingface_hub import hf_hub_download

SAM3_REPO_ID = "facebook/sam3"
SAM3_FILENAME = "sam3.pt"


def resolve_sam3_checkpoint() -> str:
    return hf_hub_download(repo_id=SAM3_REPO_ID, filename=SAM3_FILENAME)
