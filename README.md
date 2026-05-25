# SAM3 Ball Tracking

A SAM3-based ball-in-play tracker for soccer and tennis broadcast video.

The tracker uses a two-stage design:

1. Track every plausible ball-like object with SAM3.
2. Select the ball in play using sport-specific trajectory evidence and drift suppression.

This repo is extracted from the Athletic Intuition research pipeline as a standalone, public package.

## Setup

Install the package, including the vendored SAM3 workspace:

```bash
pip install -e .
```

SAM3 checkpoints are distributed by Meta through the gated Hugging Face repo `facebook/sam3`. This repo does not distribute model weights.

1. Request access at https://huggingface.co/facebook/sam3.
2. Create a Hugging Face access token.
3. Log in once on the machine that will run inference:

```bash
hf auth login
```

The first run downloads `facebook/sam3/sam3.pt` through Hugging Face Hub into the normal Hugging Face cache. You can prefetch it explicitly with:

```bash
huggingface-cli download facebook/sam3 sam3.pt
```

Soccer mode also loads ViTPose from the public `usyd-community/vitpose-plus-base` Hugging Face model. Tennis mode does not use pose.

## Quickstart

```bash
sam3-ball-track input.mp4 --sport soccer --output runs/soccer/annotated.mp4
sam3-ball-track input.mp4 --sport tennis --output runs/tennis/annotated.mp4
```

To put debug artifacts somewhere explicit:

```bash
sam3-ball-track input.mp4 \
  --sport soccer \
  --output runs/clip-01/selected_ball.mp4 \
  --debug-dir runs/clip-01
```

## Outputs

Each run writes the selected ball-in-play video to the exact `--output` path.

It also writes debug artifacts under `--debug-dir`. If `--debug-dir` is omitted, the debug directory defaults to the output path without its suffix. For example, `--output runs/soccer/annotated.mp4` uses `runs/soccer/annotated/` as the debug directory.

Debug files:

```text
all_candidates.mp4       all SAM3 ball-like tracks, colored by candidate
player_bboxes.pkl        per-frame SAM3 player detections used for drift regions
pose_data.pkl            per-frame ViTPose results for soccer, empty for tennis
ball_tracks.pkl          raw SAM3 ball candidate masks by object id
ball_drift_kills.pkl     drift-kill and SAM3 removal events
ball_masks.pkl           selected ball-in-play mask per frame
```

Soccer uses ViTPose keypoints to select candidates whose direction changes look like player contact. Tennis uses motion-only changepoints because racket contact is not represented by human body keypoints.
