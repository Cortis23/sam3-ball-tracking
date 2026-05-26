# SAM3 Ball Tracking

A SAM3-based ball-in-play tracker for soccer and tennis broadcast video.

The tracker uses a two-stage design:

1. Track every plausible ball-like object with SAM3.
2. Select the ball in play using sport-specific trajectory evidence and drift suppression.

This repo is extracted from the Athletic Intuition research pipeline as a standalone, public package.

## Setup

Requirements:

- NVIDIA GPU with CUDA
- Python 3.11+
- `uv`
- `ffmpeg` on `PATH`
- Hugging Face access to `facebook/sam3`; `facebook/sam3.1` is also required for `--sam-version sam3.1`

Install the package, including the vendored SAM3 workspace:

```bash
uv sync
```

SAM3 code in `sam3/` is Meta's SAM3 code and is covered by Meta's SAM License. The ball-tracking code in `src/` is MIT licensed. SAM3 checkpoints are distributed by Meta through the gated Hugging Face repos `facebook/sam3` and `facebook/sam3.1`; this repo does not distribute model weights.

1. Request access at https://huggingface.co/facebook/sam3. For SAM3.1, also request access at https://huggingface.co/facebook/sam3.1.
2. Create a Hugging Face access token.
3. Log in once on the machine that will run inference:

```bash
hf auth login
```

The first run downloads SAM checkpoints through Hugging Face Hub into the normal Hugging Face cache. SAM3.1 mode uses the SAM3.1 video checkpoint for ball tracking and still uses the SAM3 image checkpoint for player detection. You can prefetch them explicitly with:

```bash
uv run huggingface-cli download facebook/sam3 sam3.pt
uv run huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt
```

The first run can take a while because it loads large model dependencies and downloads gated weights.

## Quickstart

```bash
uv run sam3-ball-track input.mp4 --sport soccer
uv run sam3-ball-track input.mp4 --sport tennis
uv run sam3-ball-track input.mp4 --sport soccer --sam-version sam3.1
```

## Example Videos

The repo includes six short clips for smoke tests and demos:

```bash
uv run sam3-ball-track examples/videos/soccer/clip-1.mp4 --sport soccer
uv run sam3-ball-track examples/videos/soccer/clip-2.mp4 --sport soccer
uv run sam3-ball-track examples/videos/soccer/clip-3.mp4 --sport soccer
uv run sam3-ball-track examples/videos/tennis/clip-1.mp4 --sport tennis
uv run sam3-ball-track examples/videos/tennis/clip-2.mp4 --sport tennis
uv run sam3-ball-track examples/videos/tennis/clip-3.mp4 --sport tennis
```

## Outputs

Each run writes one ignored directory under `results/` with the form `results/<timestamp>-<sport>-<sam-version>-<clip-name>/`. That directory contains both annotated videos and compressed debug artifacts.

```text
ball-in-play.mp4         selected ball-in-play track
all-balls.mp4            all SAM3 ball-like tracks, colored by candidate
player-bboxes.pkl.zst    per-frame SAM3 player detections used for drift regions
player-masks.pkl.zst     per-frame SAM3 player mask detections
ball-tracks.pkl.zst      raw SAM3 ball candidate masks by object id
ball-drift-kills.pkl.zst drift-kill and SAM3 removal events
ball-in-play-masks.pkl.zst selected ball-in-play mask per frame
```

Soccer uses SAM3 player masks to select candidates whose direction changes look like player contact. Tennis uses motion-only changepoints.
