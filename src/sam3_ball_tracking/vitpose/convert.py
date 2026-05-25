"""Convert ViTPose++ wholebody checkpoint (mmpose split format) to safetensors.

Not wired into the pipeline. Kept as a utility: run this when adopting a new
ViTPose variant. mmpose ships weights in a split-tensor format; our loader
(vitpose/loader.py) reads safetensors, so a new variant needs a one-time
conversion here. Writes the .safetensors file + a matching config.json to
pipeline/weights/ that loader.py picks up. Afterwards run
`python -m src.infra.upload_weights` to mirror to R2.

Usage:
    cd /workspace/athleticintuition/pipeline
    python -m sam3_ball_tracking.vitpose.convert base
    python -m sam3_ball_tracking.vitpose.convert large
"""

import json
import sys
import torch
from safetensors.torch import save_file

MODELS = {
    "base": {
        "src": "/workspace/athleticintuition/ViTPose/wholebody_base.pth",
        "dst_weights": "/workspace/athleticintuition/pipeline/weights/vitpose-plus-base-wholebody.safetensors",
        "dst_config": "/workspace/athleticintuition/pipeline/weights/vitpose-plus-base-wholebody-config.json",
        "hidden_size": 768, "num_layers": 12, "num_heads": 12, "part_features": 192,
    },
    "large": {
        "src": "/workspace/athleticintuition/ViTPose/wholebody_large.pth",
        "dst_weights": "/workspace/athleticintuition/pipeline/weights/vitpose-plus-large-wholebody.safetensors",
        "dst_config": "/workspace/athleticintuition/pipeline/weights/vitpose-plus-large-wholebody-config.json",
        "hidden_size": 1024, "num_layers": 24, "num_heads": 16, "part_features": 256,
    },
}


def convert(variant: str):
    m = MODELS[variant]
    ckpt = torch.load(m["src"], map_location="cpu")
    src = ckpt["state_dict"]
    dst = {}

    dst["backbone.embeddings.patch_embeddings.projection.weight"] = src["backbone.patch_embed.proj.weight"]
    dst["backbone.embeddings.patch_embeddings.projection.bias"] = src["backbone.patch_embed.proj.bias"]
    dst["backbone.embeddings.position_embeddings"] = src["backbone.pos_embed"]
    dst["backbone.layernorm.weight"] = src["backbone.last_norm.weight"]
    dst["backbone.layernorm.bias"] = src["backbone.last_norm.bias"]

    for i in range(m["num_layers"]):
        s = f"backbone.blocks.{i}"
        d = f"backbone.encoder.layer.{i}"

        qkv_w = src[f"{s}.attn.qkv.weight"]
        qkv_b = src[f"{s}.attn.qkv.bias"]
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
        dst[f"{d}.attention.attention.query.weight"] = q_w
        dst[f"{d}.attention.attention.query.bias"] = q_b
        dst[f"{d}.attention.attention.key.weight"] = k_w
        dst[f"{d}.attention.attention.key.bias"] = k_b
        dst[f"{d}.attention.attention.value.weight"] = v_w
        dst[f"{d}.attention.attention.value.bias"] = v_b

        dst[f"{d}.attention.output.dense.weight"] = src[f"{s}.attn.proj.weight"]
        dst[f"{d}.attention.output.dense.bias"] = src[f"{s}.attn.proj.bias"]
        dst[f"{d}.layernorm_before.weight"] = src[f"{s}.norm1.weight"]
        dst[f"{d}.layernorm_before.bias"] = src[f"{s}.norm1.bias"]
        dst[f"{d}.layernorm_after.weight"] = src[f"{s}.norm2.weight"]
        dst[f"{d}.layernorm_after.bias"] = src[f"{s}.norm2.bias"]
        dst[f"{d}.mlp.fc1.weight"] = src[f"{s}.mlp.fc1.weight"]
        dst[f"{d}.mlp.fc1.bias"] = src[f"{s}.mlp.fc1.bias"]
        dst[f"{d}.mlp.fc2.weight"] = src[f"{s}.mlp.fc2.weight"]
        dst[f"{d}.mlp.fc2.bias"] = src[f"{s}.mlp.fc2.bias"]

    dst["head.deconv1.weight"] = src["keypoint_head.deconv_layers.0.weight"]
    dst["head.batchnorm1.weight"] = src["keypoint_head.deconv_layers.1.weight"]
    dst["head.batchnorm1.bias"] = src["keypoint_head.deconv_layers.1.bias"]
    dst["head.batchnorm1.running_mean"] = src["keypoint_head.deconv_layers.1.running_mean"]
    dst["head.batchnorm1.running_var"] = src["keypoint_head.deconv_layers.1.running_var"]
    dst["head.batchnorm1.num_batches_tracked"] = src["keypoint_head.deconv_layers.1.num_batches_tracked"]
    dst["head.deconv2.weight"] = src["keypoint_head.deconv_layers.3.weight"]
    dst["head.batchnorm2.weight"] = src["keypoint_head.deconv_layers.4.weight"]
    dst["head.batchnorm2.bias"] = src["keypoint_head.deconv_layers.4.bias"]
    dst["head.batchnorm2.running_mean"] = src["keypoint_head.deconv_layers.4.running_mean"]
    dst["head.batchnorm2.running_var"] = src["keypoint_head.deconv_layers.4.running_var"]
    dst["head.batchnorm2.num_batches_tracked"] = src["keypoint_head.deconv_layers.4.num_batches_tracked"]
    dst["head.conv.weight"] = src["keypoint_head.final_layer.weight"]
    dst["head.conv.bias"] = src["keypoint_head.final_layer.bias"]

    print(f"Converted {len(dst)} tensors ({variant})")
    print(f"Head output: {dst['head.conv.weight'].shape[0]} keypoints")

    save_file(dst, m["dst_weights"])
    print(f"Saved: {m['dst_weights']}")

    id2label = {
        "0": "Nose", "1": "L_Eye", "2": "R_Eye", "3": "L_Ear", "4": "R_Ear",
        "5": "L_Shoulder", "6": "R_Shoulder", "7": "L_Elbow", "8": "R_Elbow",
        "9": "L_Wrist", "10": "R_Wrist", "11": "L_Hip", "12": "R_Hip",
        "13": "L_Knee", "14": "R_Knee", "15": "L_Ankle", "16": "R_Ankle",
        "17": "L_Big_Toe", "18": "L_Small_Toe", "19": "L_Heel",
        "20": "R_Big_Toe", "21": "R_Small_Toe", "22": "R_Heel",
    }
    for i in range(23, 91):
        id2label[str(i)] = f"face_{i - 23}"
    for i in range(91, 112):
        id2label[str(i)] = f"left_hand_{i - 91}"
    for i in range(112, 133):
        id2label[str(i)] = f"right_hand_{i - 112}"

    config = {
        "backbone_config": {
            "hidden_size": m["hidden_size"],
            "image_size": [256, 192],
            "layer_norm_eps": 1e-12,
            "mlp_ratio": 4,
            "num_attention_heads": m["num_heads"],
            "num_channels": 3,
            "num_experts": 1,
            "num_hidden_layers": m["num_layers"],
            "part_features": m["part_features"],
            "patch_size": [16, 16],
            "qkv_bias": True,
        },
        "id2label": id2label,
        "use_simple_decoder": False,
        "scale_factor": 4,
    }

    with open(m["dst_config"], "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {m['dst_config']}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    convert(variant)
