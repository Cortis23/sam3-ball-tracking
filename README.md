# SAM3 Ball Tracking

A SAM3-based ball-in-play tracker for soccer and tennis broadcast video.

The tracker uses a two-stage design:

1. Track every plausible ball-like object with SAM3.
2. Select the ball in play using sport-specific trajectory evidence and drift suppression.

This repo is extracted from the Athletic Intuition research pipeline as a standalone, public package.

## Quickstart:

```bash
sam3-ball-track input.mp4 --sport soccer --output runs/example/annotated.mp4
sam3-ball-track input.mp4 --sport tennis --output runs/example/annotated.mp4
```

Model weights are not committed. By default the package looks under `weights/`.
Set `SAM3_BALL_WEIGHTS=/path/to/weights` to use another directory.

Expected files:

```text
weights/
  sam3.pt
  vitpose-plus-base-wholebody.safetensors
  vitpose-plus-base-wholebody-config.json
```

Soccer uses ViTPose keypoints to select candidates whose direction changes look like player contact. Tennis uses motion-only changepoints because racket contact is not represented by human body keypoints.
