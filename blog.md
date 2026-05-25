# Blog Plan: Ball-in-Play Tracking with SAM3

## Core Story

The blog should teach a practical pattern for tracking a small fast object in sports video with SAM3: do not try to force the model to output only the real ball. Instead, split the problem into two simpler parts:

1. Find every plausible ball-like object over time.
2. Select the ball that is actually in play.

That split is the main pedagogical point. SAM3 is good at proposing and propagating many candidate masks, but broadcast sports contain ads, crowd artifacts, referee gear, duplicate balls, ball kids, and visual clutter. The reliable tracker comes from treating SAM3 as a candidate generator and then using sport logic to choose the candidate that behaves like the match ball.

The title idea should emphasize this: not just "tracking a ball with SAM3", but "tracking the ball in play".

## Audience

The post is for engineers/researchers who can run Python and understand CV basics, but who may not know how to turn a foundation segmentation model into a robust sports tracker.

Assume they understand:

- videos as frames
- masks and bounding boxes
- basic tracking
- Python CLI tools

Do not assume they know:

- SAM3 video internals
- sports-specific ball selection logic
- why small-object tracking drifts
- why one-stage "detect the ball" approaches fail

## Opening

Start with the failure mode: asking a detector or segmenter for "ball" does not give a single clean ball track. It gives many plausible candidates.

Show the contrast:

- `all-balls.mp4`: every SAM3 candidate track
- `ball-in-play.mp4`: the selected match ball

The first visual should make the problem obvious. The all-candidates render is the hook: SAM3 finds too much, but that is useful if we handle it correctly.

## Proposed Structure

### 1. The Problem: Small Object Tracking in Broadcast Sports

Explain why soccer and tennis balls are hard:

- tiny relative to frame size
- motion blur
- occlusion by players
- camera cuts and pans
- visually similar false positives
- intermittent visibility
- detector confidence is not enough

Key claim: the hard part is not only finding balls, it is deciding which candidate is the ball in play.

### 2. The Key Abstraction: Candidate Tracking vs Ball-in-Play Selection

Introduce the two-stage architecture:

```text
video
  -> SAM3 player detection
  -> SAM3 ball candidate tracking
  -> drift suppression
  -> sport-specific candidate selection
  -> annotated videos + artifacts
```

Explain why this decomposition works:

- candidate tracking can be permissive
- selection can be conservative
- false positives are allowed to exist as long as they lose the selection step
- the selected output naturally ignores balls not in play

This section should define "ball-in-play tracker" explicitly.

### 3. Stage 1: Finding All Ball Candidates with SAM3

Explain the SAM3 usage:

- initialize video predictor
- add a text prompt: `ball`
- run per-frame text-grounded propagation
- store masks by object id and frame

Make clear that we keep masks, not just boxes, for ball candidates. Masks let us compute centroids, render precise overlays, detect drift, and measure proximity to players.

Mention the output artifact:

```text
ball-tracks.pkl.zst
```

Also explain the diagnostic video:

```text
all-balls.mp4
```

This is where readers understand that the system intentionally tracks more than one ball-like thing.

### 4. Drift: When a Ball Track Becomes a Player Track

Explain the common failure: a tiny ball track can latch onto a player, shoe, sock, or other object after occlusion/contact.

Current drift logic:

- detect players with SAM3 prompt `player`
- keep player bounding boxes
- compute ball mask centroid
- if the centroid sits inside any player box while SAM3 keep-alive has bottomed out for 10 frames, kill the track

Important nuance for the blog: this is not trying to identify the same player. It only asks whether the ball candidate has been attached to some player for long enough. That makes the algorithm simpler and more general.

Explain why 10 frames is enough: at 25-30 fps, that is roughly a third of a second, long enough to distinguish drift from a momentary overlap.

Potential future note: player masks could replace player boxes for a tighter attachment test.

### 5. Stage 2: Selecting the Ball in Play

Explain the selection philosophy:

The real ball is the candidate whose motion contains plausible play events. Static false positives or camera-panned background objects do not have the same trajectory evidence.

General selection ingredients:

- compute ball centroids from masks
- find direction changes / changepoints
- score tracks by plausible match-ball motion
- trim tracks killed by drift before selection
- output one selected mask per frame

This section should stay conceptual before splitting by sport.

### 6. Soccer Selection

Current soccer selection uses pose-gated changepoints, but explain accurately:

- compute angle changes in ball trajectory
- keep large direction changes
- gate them by proximity to confident player keypoints

Important honesty: this is not currently foot-specific. It uses any confident ViTPose keypoint as a sparse human-proximity signal. That means the blog should not oversell "kick detection" or "foot contact detection" unless we change the implementation.

Possible framing:

"For soccer, a useful signal is that meaningful ball direction changes often happen near players. We use pose keypoints as sparse player-contact anchors. A mask-based player proximity gate would be a reasonable simplification and may replace pose in a future iteration."

If we decide to remove pose before publishing, this section should become:

- detect player masks/boxes
- dilate player regions
- accept ball changepoints near player regions

Do not write the final blog prose until this decision is settled.

### 7. Tennis Selection

Explain why tennis does not use pose:

- racket contact occurs away from body keypoints
- wrist/hand keypoints are too coarse at broadcast resolution
- ball direction/speed changes are cleaner signals

Tennis uses motion-only changepoints:

- direction changes
- speed/displacement thresholds
- no pose gate

This contrast is useful pedagogically: same candidate-tracking core, different sport-specific selector.

### 8. Outputs and Reproducibility

Show the CLI:

```bash
uv run sam3-ball-track examples/videos/soccer/clip-1.mp4 --sport soccer
uv run sam3-ball-track examples/videos/tennis/clip-1.mp4 --sport tennis
```

Explain generated result directory:

```text
results/<timestamp>-<sport>-<clip-name>/
  ball-in-play.mp4
  all-balls.mp4
  player-bboxes.pkl.zst
  pose-data.pkl.zst
  ball-tracks.pkl.zst
  ball-drift-kills.pkl.zst
  ball-in-play-masks.pkl.zst
```

Mention bundled examples:

```text
examples/videos/soccer/clip-1.mp4 ... clip-3.mp4
examples/videos/tennis/clip-1.mp4 ... clip-3.mp4
```

Include SAM3 model access note:

- requires an NVIDIA CUDA GPU and `ffmpeg`
- SAM3 code in `sam3/` is Meta's code under Meta's SAM License
- the ball-tracking code in `src/` is MIT licensed
- repo does not distribute SAM3 weights
- request access to `facebook/sam3`
- authenticate with Hugging Face

### 9. What Makes This Different from a Normal Tracker

Key points:

- no single-object assumption
- false positives are first-class candidates
- drift is expected and explicitly handled
- sport logic is separate from SAM3 inference
- outputs include both raw candidates and final selection for debugging

This is the section that makes the post more than a README walkthrough.

### 10. Limitations

Be direct:

- not a benchmark claim
- selection can fail if SAM3 never detects the ball
- crowded scenes can still confuse player proximity logic
- camera cuts are not deeply modeled in the standalone version
- soccer pose gate is proximity-based, not true contact understanding
- CUDA GPU and `ffmpeg` are required
- first run downloads large torch/SAM3 dependencies and gated weights

### 11. Future Improvements

Possible improvements:

- replace soccer pose with player-mask proximity if it performs similarly
- use SAM3 player masks for drift instead of player boxes
- add camera-cut segmentation
- add confidence/score plots for candidate selection
- add frame-level debug overlays for changepoints
- benchmark runtime and accuracy across the six example clips

## Suggested Visuals

Minimum visuals:

1. `all-balls.mp4` screenshot/GIF: many candidates.
2. `ball-in-play.mp4` screenshot/GIF: selected track.
3. Architecture diagram showing candidate generation vs selection.
4. A frame with a drift-killed candidate attached to a player.
5. A trajectory plot showing changepoints for a selected candidate.

Optional visuals:

- side-by-side soccer vs tennis selection logic
- result directory tree
- mask centroid / player box drift diagram

## Terms to Use Consistently

Use:

- ball candidate
- candidate track
- ball in play
- drift suppression
- changepoint
- player proximity gate
- sport-specific selector

Avoid overclaiming:

- do not say "detects kicks" unless we implement foot-specific logic
- do not say "understands gameplay"
- do not imply SAM3 alone solves ball tracking
- do not call pose gating foot contact detection in the current implementation

## Open Decisions Before Writing

1. Do we keep soccer pose, or replace it with player-mask/player-box proximity?
2. Do we want player masks in the public implementation now?
3. Should the blog use soccer or tennis as the primary walkthrough example?
4. Do we want to include runtime notes, given the ball step may be slower than expected?
5. Should we mention the full Athletic Intuition pipeline, or keep the post fully standalone?

## Likely Final Post Thesis

SAM3 makes it easy to find plausible ball-like objects, but robust sports ball tracking comes from not asking SAM3 to choose the game ball. Track all candidates, suppress obvious drift, then use sport-specific motion and player-proximity signals to select the ball in play.
