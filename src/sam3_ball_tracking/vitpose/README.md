# ViTPose Standalone

Standalone ViTPose++ implementation — pure PyTorch, no HuggingFace transformers or mmpose dependencies. This avoids the ~5 minute import time of transformers and the mmpose dependency chain.

## Models

We use ViTPose++ WholeBody models (133 keypoints including foot keypoints: big toe, small toe, heel). Stored as safetensors in `pipeline/weights/` (mirrored to R2 via `python -m src.infra.upload_weights`).

| Variant | Params | File |
|---------|--------|------|
| Base    | ~90M   | `vitpose-plus-base-wholebody.safetensors` |

Only the base variant is mirrored to R2. Large is supported by `convert.py` but not currently in use.

## Why WholeBody?

Standard COCO pose (17 keypoints) stops at the ankle — no foot keypoints. For soccer touch attribution, we need to know where the player's foot is, not just their ankle. COCO-WholeBody adds 6 foot keypoints per foot (big toe, small toe, heel), giving us the actual contact surface.

## How to get the models

The ViTPose++ models on HuggingFace (`usyd-community/vitpose-plus-base`) only support COCO 17-keypoint output. WholeBody (133 keypoints) is not available on HuggingFace (known issue). We get the weights from the original ViTPose repo instead.

### Steps

1. **Clone the ViTPose repo:**
   ```
   git clone https://github.com/ViTAE-Transformer/ViTPose.git
   ```

2. **Download the multi-task checkpoint from OneDrive** (links in the ViTPose README):
   - Base: `https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G`
   - Large: `https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ`

   These are multi-task ViTPose++ checkpoints with MoE experts for 6 datasets.

3. **Split into single-task checkpoints** using the ViTPose repo's tool:
   ```
   cd ViTPose
   python tools/model_split.py --source vitpose_base.pth --target .
   ```
   This produces `wholebody.pth` — the wholebody expert merged into the backbone with the 133-keypoint decoder head. No MoE routing needed at inference.

4. **Convert to our safetensors format:**
   ```
   cd pipeline
   python -m src.vitpose.convert base
   python -m src.vitpose.convert large
   ```
   This maps the mmpose state dict keys to our model's key format (split fused QKV, rename layers) and saves as safetensors with a config JSON.

### Why this process?

The original ViTPose++ is a multi-task MoE model — one checkpoint, 6 expert branches, 6 decoder heads. `model_split.py` extracts a single task into a standalone model by merging the selected expert into the MLP weights and keeping only the relevant decoder head. The result is a simpler, faster model that doesn't need MoE routing at inference.

We then convert the mmpose key format to match our standalone PyTorch implementation, which was ported from the HuggingFace transformers source but runs without any HuggingFace dependencies.
