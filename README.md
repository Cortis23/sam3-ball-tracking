# SAM3 Ball Tracking

A SAM3-based ball-in-play tracker for soccer and tennis broadcast video.

The tracker uses a two-stage design:

1. Track every plausible ball-like object with SAM3.
2. Select the ball in play using sport-specific trajectory evidence and drift suppression.

This repo is extracted from the Athletic Intuition research pipeline as a standalone, public package.

## Setup

Install the package, including the vendored SAM3 workspace:

```bash
uv sync
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
uv run huggingface-cli download facebook/sam3 sam3.pt
```

Soccer mode also loads ViTPose from the public `usyd-community/vitpose-plus-base` Hugging Face model. Tennis mode does not use pose.

## Quickstart

```bash
uv run sam3-ball-track input.mp4 --sport soccer --output-dir runs/soccer
uv run sam3-ball-track input.mp4 --sport tennis --output-dir runs/tennis
```

## Example Videos

The repo includes six short clips for smoke tests and demos:

```bash
uv run sam3-ball-track examples/videos/soccer/chelsea-arsenal/clip-3.mp4 --sport soccer --output-dir runs/chelsea-arsenal-clip-3
uv run sam3-ball-track examples/videos/soccer/portugal-spain/clip-6.mp4 --sport soccer --output-dir runs/portugal-spain-clip-6
uv run sam3-ball-track examples/videos/soccer/sportsmot-v_dw7LOz17Omg/c067.mp4 --sport soccer --output-dir runs/sportsmot-c067
uv run sam3-ball-track examples/videos/tennis/game2/Clip6.mp4 --sport tennis --output-dir runs/tennis-game2-clip6
uv run sam3-ball-track examples/videos/tennis/game3/Clip3.mp4 --sport tennis --output-dir runs/tennis-game3-clip3
uv run sam3-ball-track examples/videos/tennis/game5/Clip4.mp4 --sport tennis --output-dir runs/tennis-game5-clip4
```

## Outputs

Each run writes one directory containing both annotated videos and compressed debug artifacts.

```text
ball-in-play.mp4         selected ball-in-play track
all-balls.mp4            all SAM3 ball-like tracks, colored by candidate
player-bboxes.pkl.zst    per-frame SAM3 player detections used for drift regions
pose-data.pkl.zst        per-frame ViTPose results for soccer, empty for tennis
ball-tracks.pkl.zst      raw SAM3 ball candidate masks by object id
ball-drift-kills.pkl.zst drift-kill and SAM3 removal events
ball-in-play-masks.pkl.zst selected ball-in-play mask per frame
```

Soccer uses ViTPose keypoints to select candidates whose direction changes look like player contact. Tennis uses motion-only changepoints because racket contact is not represented by human body keypoints.
