# SAM3 Ball Tracking

A SAM3-based ball-in-play tracker for soccer and tennis broadcast video.

The tracker uses a two-stage design:

1. Track every plausible ball-like object with SAM3.
2. Select the ball in play using sport-specific trajectory evidence and drift suppression.

This repo is extracted from the Athletic Intuition research pipeline as a standalone, public package.

## Status

Initial extraction in progress. The target CLI is:

```bash
sam3-ball-track input.mp4 --sport soccer --output runs/example/annotated.mp4
sam3-ball-track input.mp4 --sport tennis --output runs/example/annotated.mp4
```

Model weights are not committed. By default the package looks under `weights/`.
