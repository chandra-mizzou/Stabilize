# Stabilize

Python utility for measuring continuous x/y (azimuth/elevation) jitter in a
video and producing a four-quadrant Key.Net stabilization analysis video:

1. Original video
2. Key.Net feature-motion vectors overlaid on each frame
3. Raw x/y camera-shift trajectory and stabilization target versus frame number
4. Frame-to-frame translation-stabilized video

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
python3 video_jitter_stabilizer.py input.mp4 \
  --output output_jitter_analysis.mp4 \
  --stable-output stable_only.mp4 \
  --num-frames 100 \
  --csv trajectory.csv
```

Use `--display` to show the four-quadrant video while it is being processed.
Press `q` to stop the live preview.

By default, the script stabilizes and plots only the first 100 frames. Change
the frame count with `--num-frames`; use `--num-frames 0` to process the full
video.

## How stabilization works

- Kornia's `KeyNetHardNet` detects Key.Net keypoints and computes HardNet
  descriptors for every frame.
- Consecutive frames are descriptor-matched with a mutual Lowe-ratio test.
- RANSAC rejects bad matches, but the final motion estimate is translation only:
  the median x/y displacement of inlier Key.Net matches.
- These frame-to-frame x/y shifts are accumulated into a raw camera trajectory
  relative to frame 1.
- By default (`--stabilization-mode frame-to-frame`), every frame is warped by
  the negative cumulative shift. This aligns each frame back to frame 1 and does
  not use smoothing.

For pure azimuth/elevation jitter, use the default frame-to-frame mode:

```bash
python3 video_jitter_stabilizer.py input.mp4 \
  --stabilization-mode frame-to-frame \
  --num-frames 100
```

This mode removes both x and y jitter when both are present in the same frame,
because the correction is a 2D inverse translation:

```text
correction(frame n) = -sum(shift frame i-1 -> frame i), for i=2..n
```

## Why not only average the past five frames?

A causal average of only the past five frames often does **not** stabilize
azimuth/elevation jitter well. It follows the jitter with lag and may leave the
current frame close to the original shaky trajectory. For pure x/y jitter, the
better default is to align every frame back to frame 1 with cumulative inverse
translation.

The script still includes an optional smoothed-path mode for videos that contain
intentional pan/tilt that should be preserved:

```bash
python3 video_jitter_stabilizer.py input.mp4 --stabilization-mode smooth-trajectory
```

If you specifically want a five-frame centered smoother in that mode:

```bash
python3 video_jitter_stabilizer.py input.mp4 \
  --stabilization-mode smooth-trajectory \
  --smoothing-radius 2
```

For stronger stabilization, increase the radius, for example:

```bash
python3 video_jitter_stabilizer.py input.mp4 \
  --stabilization-mode smooth-trajectory \
  --smoothing-radius 30
```

To process only the first 250 frames:

```bash
python3 video_jitter_stabilizer.py input.mp4 --num-frames 250
```

Useful options:

```bash
python3 video_jitter_stabilizer.py --help
```

If Key.Net motion arrows are still too short for a low-amplitude jitter video,
increase `--flow-scale`, for example `--flow-scale 15`.

Useful tuning options:

- `--max-features`: more Key.Net features can improve matching on textured
  videos.
- `--num-frames`: number of frames to stabilize and plot from the start of the
  video; default is 100, and 0 means full video.
- `--stabilization-mode`: `frame-to-frame` aligns every frame to frame 1;
  `smooth-trajectory` preserves slow camera motion.
- `--match-ratio`: lower values make matching stricter.
- `--ransac-threshold`: higher values tolerate noisier matches.
- `--border-scale`: zooms the stabilized frame slightly to hide borders.

## Top-100 keypoint shift stabilizer

For pure azimuth/elevation jitter, `top100_keypoint_stabilizer.py` implements a
more direct frame-to-frame translation correction:

```bash
python3 top100_keypoint_stabilizer.py input.mp4 \
  --stable-output top100_stable.mp4 \
  --output top100_analysis.mp4 \
  --csv top100_shifts.csv \
  --num-frames 100 \
  --process-size 256 \
  --analysis-size 256
```

The script resizes every input frame to `256x256` by default before Key.Net
processing. The stabilized-only output is also `256x256`. The four-quadrant
analysis view is downsampled and saved as a compact `256x256` video by default
so it is only used for quick diagnostics.

Algorithm:

1. Detect the top 100 Key.Net keypoints in the previous frame and current frame.
2. Match the top-keypoint descriptors between immediate consecutive frames.
3. Compute the average matched keypoint shift `(dx, dy)`.
4. Accumulate the shift within the current reference segment.
5. Move the current frame back by the inverse cumulative shift.
6. Use black zero-padding for newly exposed image regions so the resolution is
   unchanged.
7. If at least 10 of the top 100 keypoints are no longer matched, make the
   current frame the new reference segment.

Tune these options:

- `--top-k 100`: number of top Key.Net keypoints.
- `--process-size 256`: square frame size used for keypoint extraction and
  stabilization.
- `--analysis-size 256`: final square size of the four-quadrant analysis video.
- `--reset-threshold 10`: number of changed/unmatched top keypoints before a
  reference reset.
- `--match-ratio`: stricter or looser descriptor matching.
- `--max-shift`: rejects implausibly large average shifts.
- `--progress-every 1`: print progress every frame. Increase it to reduce log
  volume.

New keypoints usually appear when the camera/scene content moves enough to
expose new texture, but that is not the only cause. Lighting changes, blur,
moving foreground objects, compression artifacts, and Key.Net score/ranking
changes can also alter the top-100 set.
