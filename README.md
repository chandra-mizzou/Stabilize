# Stabilize

Python utility for measuring continuous x/y (azimuth/elevation) jitter in a
video and producing a four-quadrant analysis video:

1. Original video
2. Optical-flow vectors overlaid on each frame
3. x/y average keypoint shift versus frame number, using first-frame keypoints
   as the reference
4. Jitter-corrected stabilized video

## Setup

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python video_jitter_stabilizer.py input.mp4 \
  --output output_jitter_analysis.mp4 \
  --csv shifts.csv
```

Use `--display` to show the four-quadrant video while it is being processed.
Press `q` to stop the live preview.

## How correction works

- The first frame is used to detect reference Shi-Tomasi keypoints.
- From the second frame onward, pyramidal Lucas-Kanade optical flow tracks
  keypoints common with the previous frame.
- The shift plot shows the average `(x, y)` displacement of first-frame
  reference keypoints versus frame number.
- The stabilization path uses the frame-to-frame movement of common keypoints.
  For the first few frames it corrects using the current measured movement.
  Once the rolling window is full (five frames by default), it corrects the
  current frame using the average movement over the recent window.
- When tracked points are lost because the video shifts, newly detected
  keypoints are added for future frames, so they can participate in correction
  once they become common with the previous frame.

Useful options:

```bash
python video_jitter_stabilizer.py --help
```
