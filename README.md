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
