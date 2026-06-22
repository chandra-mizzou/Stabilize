# Stabilize

Python utility for measuring continuous x/y (azimuth/elevation) jitter in a
video and producing a four-quadrant Key.Net stabilization analysis video:

1. Original video
2. Key.Net feature-motion vectors overlaid on each frame
3. Raw and smoothed x/y camera-shift trajectory versus frame number
4. Affine trajectory-stabilized video

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
python3 video_jitter_stabilizer.py input.mp4 \
  --output output_jitter_analysis.mp4 \
  --stable-output stable_only.mp4 \
  --csv trajectory.csv
```

Use `--display` to show the four-quadrant video while it is being processed.
Press `q` to stop the live preview.

## How stabilization works

- Kornia's `KeyNetHardNet` detects Key.Net keypoints and computes HardNet
  descriptors for every frame.
- Consecutive frames are descriptor-matched with a mutual Lowe-ratio test.
- A robust partial affine transform is estimated with RANSAC, giving per-frame
  x shift, y shift, and small rotation.
- These frame-to-frame transforms are accumulated into a raw camera trajectory.
- The full trajectory is smoothed with a centered moving average
  (`--smoothing-radius`; default radius 15, so a 31-frame window).
- Each frame is warped by the difference between the smoothed trajectory and the
  raw trajectory. This removes fast jitter while preserving slow intentional
  motion.

## Why not only average the past five frames?

A causal average of only the past five frames often does **not** stabilize
offline video well. It follows the jitter with lag, cannot use future context,
and may leave the current frame close to the original shaky trajectory. A better
offline approach is:

1. Estimate the camera motion for the whole video.
2. Smooth the whole trajectory.
3. Warp each frame from the raw trajectory to the smoothed trajectory.

If you specifically want a five-frame smoother, use a centered five-frame window:

```bash
python3 video_jitter_stabilizer.py input.mp4 --smoothing-radius 2
```

For stronger stabilization, increase the radius, for example:

```bash
python3 video_jitter_stabilizer.py input.mp4 --smoothing-radius 30
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
- `--match-ratio`: lower values make matching stricter.
- `--ransac-threshold`: higher values tolerate noisier matches.
- `--border-scale`: zooms the stabilized frame slightly to hide borders.
