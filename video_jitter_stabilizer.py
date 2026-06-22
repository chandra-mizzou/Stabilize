#!/usr/bin/env python3
"""Analyze and reduce x/y video jitter with keypoints and optical flow.

The first frame is used as the reference for the shift plot. For correction,
the script tracks keypoints common with the previous frame, adds newly detected
keypoints for future frames, and applies a rolling average of recent keypoint
motion once enough history is available.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_MAX_POINTS = 700
DEFAULT_MIN_ACTIVE_POINTS = 120
DEFAULT_ROLLING_WINDOW = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a video, plot first-frame-reference x/y keypoint shifts, "
            "overlay optical-flow vectors, and write a four-quadrant jitter "
            "analysis/stabilization video."
        )
    )
    parser.add_argument("input_video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output four-quadrant video path. Defaults to '<input>_jitter_analysis.mp4'.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV path for frame_no, shift_x, shift_y, correction_x, correction_y.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the four-quadrant video while processing. Press q to stop.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"Maximum active keypoints to keep for tracking (default: {DEFAULT_MAX_POINTS}).",
    )
    parser.add_argument(
        "--min-active-points",
        type=int,
        default=DEFAULT_MIN_ACTIVE_POINTS,
        help=(
            "Refresh keypoints when active tracks drop below this count "
            f"(default: {DEFAULT_MIN_ACTIVE_POINTS})."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help=(
            "Number of previous frame-to-frame motions to average after enough "
            f"history is available (default: {DEFAULT_ROLLING_WINDOW})."
        ),
    )
    parser.add_argument(
        "--quality-level",
        type=float,
        default=0.01,
        help="Shi-Tomasi keypoint quality level (default: 0.01).",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=8.0,
        help="Minimum distance between detected keypoints in pixels (default: 8).",
    )
    parser.add_argument(
        "--vector-stride",
        type=int,
        default=4,
        help="Draw every Nth optical-flow vector to reduce clutter (default: 4).",
    )
    parser.add_argument(
        "--plot-history",
        type=int,
        default=240,
        help="Number of recent frames shown in the shift quadrant (default: 240).",
    )
    return parser.parse_args()


def detect_keypoints(
    gray: np.ndarray,
    max_points: int,
    quality_level: float,
    min_distance: float,
    existing_points: np.ndarray | None = None,
) -> np.ndarray:
    """Detect Shi-Tomasi corners, optionally avoiding existing tracks."""
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    if existing_points is not None and len(existing_points) > 0:
        radius = max(2, int(min_distance))
        for point in existing_points.reshape(-1, 2):
            cv2.circle(mask, tuple(np.round(point).astype(int)), radius, 0, -1)

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        mask=mask,
    )
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points.astype(np.float32)


def refresh_keypoints(
    gray: np.ndarray,
    current_points: np.ndarray,
    max_points: int,
    min_active_points: int,
    quality_level: float,
    min_distance: float,
) -> np.ndarray:
    """Add newly visible keypoints for future optical-flow corrections."""
    if len(current_points) >= max_points or len(current_points) >= min_active_points:
        return current_points

    needed = max_points - len(current_points)
    new_points = detect_keypoints(
        gray,
        max_points=needed,
        quality_level=quality_level,
        min_distance=min_distance,
        existing_points=current_points,
    )
    if len(new_points) == 0:
        return current_points
    if len(current_points) == 0:
        return new_points
    return np.concatenate([current_points.astype(np.float32), new_points], axis=0)


def track_points(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Track points with pyramidal Lucas-Kanade optical flow."""
    if previous_points is None or len(previous_points) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    good = status.reshape(-1).astype(bool)
    return previous_points[good].reshape(-1, 1, 2), next_points[good].reshape(-1, 1, 2)


def track_reference_points(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    reference_initial_points: np.ndarray,
    reference_previous_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Track first-frame reference points while keeping original pairings."""
    if len(reference_previous_points) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        reference_previous_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    good = status.reshape(-1).astype(bool)
    return reference_initial_points[good].reshape(-1, 1, 2), next_points[good].reshape(-1, 1, 2)


def mean_delta(previous_points: np.ndarray, current_points: np.ndarray) -> np.ndarray:
    """Return mean dx/dy between two same-length point arrays."""
    if len(previous_points) == 0 or len(current_points) == 0:
        return np.zeros(2, dtype=np.float32)
    deltas = current_points.reshape(-1, 2) - previous_points.reshape(-1, 2)
    return np.mean(deltas, axis=0).astype(np.float32)


def moving_average(values: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.asarray(list(values), dtype=np.float32)
    if stacked.size == 0:
        return np.zeros(2, dtype=np.float32)
    return np.mean(stacked, axis=0).astype(np.float32)


def draw_flow_vectors(
    frame: np.ndarray,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Overlay optical-flow direction vectors on a frame."""
    overlay = frame.copy()
    points_prev = previous_points.reshape(-1, 2)
    points_curr = current_points.reshape(-1, 2)
    stride = max(1, stride)

    for prev, curr in zip(points_prev[::stride], points_curr[::stride]):
        p0 = tuple(np.round(prev).astype(int))
        p1 = tuple(np.round(curr).astype(int))
        cv2.arrowedLine(overlay, p0, p1, (0, 255, 255), 1, tipLength=0.35)
        cv2.circle(overlay, p1, 2, (0, 80, 255), -1)

    cv2.putText(
        overlay,
        f"Optical flow vectors: {len(points_curr)} tracked",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def draw_shift_plot(
    width: int,
    height: int,
    frame_numbers: list[int],
    shifts_x: list[float],
    shifts_y: list[float],
    history: int,
) -> np.ndarray:
    """Draw x/y shift traces in pixels for use as a video quadrant."""
    plot = np.full((height, width, 3), 245, dtype=np.uint8)
    margin_left = 58
    margin_right = 16
    margin_top = 34
    margin_bottom = 48
    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top

    cv2.rectangle(plot, (x0, y1), (x1, y0), (35, 35, 35), 1)
    cv2.putText(
        plot,
        "3. Shift wrt first-frame keypoints",
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    if not frame_numbers:
        return plot

    recent_frames = frame_numbers[-history:]
    recent_x = np.asarray(shifts_x[-history:], dtype=np.float32)
    recent_y = np.asarray(shifts_y[-history:], dtype=np.float32)
    valid = np.isfinite(recent_x) & np.isfinite(recent_y)
    if not np.any(valid):
        return plot

    max_abs = float(np.nanmax(np.abs(np.concatenate([recent_x[valid], recent_y[valid]]))))
    max_abs = max(1.0, max_abs)
    y_mid = int(round((y0 + y1) / 2.0))
    cv2.line(plot, (x0, y_mid), (x1, y_mid), (190, 190, 190), 1)

    cv2.putText(plot, f"+{max_abs:.1f}px", (6, y1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    cv2.putText(plot, "0", (30, y_mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    cv2.putText(plot, f"-{max_abs:.1f}px", (6, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)

    frame_min = recent_frames[0]
    frame_max = max(recent_frames[-1], frame_min + 1)

    def to_screen(frame_no: int, value: float) -> tuple[int, int]:
        x = x0 + int(round((frame_no - frame_min) / (frame_max - frame_min) * (x1 - x0)))
        y = y_mid - int(round(value / max_abs * ((y0 - y1) / 2.0)))
        return x, int(np.clip(y, y1, y0))

    def draw_trace(values: np.ndarray, color: tuple[int, int, int]) -> None:
        points: list[tuple[int, int]] = []
        for frame_no, value in zip(recent_frames, values):
            if np.isfinite(value):
                points.append(to_screen(frame_no, float(value)))
        for start, end in zip(points, points[1:]):
            cv2.line(plot, start, end, color, 2, cv2.LINE_AA)

    draw_trace(recent_x, (30, 80, 230))
    draw_trace(recent_y, (30, 170, 30))
    cv2.putText(plot, "x shift", (x0 + 8, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 80, 230), 2)
    cv2.putText(plot, "y shift", (x0 + 100, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 170, 30), 2)
    cv2.putText(
        plot,
        f"frames {frame_min}-{recent_frames[-1]}",
        (max(x0 + 190, x1 - 170), y0 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        1,
    )
    return plot


def translate_frame(frame: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Warp a frame by the negative cumulative shift to stabilize it."""
    dx, dy = float(shift[0]), float(shift[1])
    transform = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        transform,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def label_panel(frame: np.ndarray, title: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_four_panel(
    original: np.ndarray,
    flow: np.ndarray,
    plot: np.ndarray,
    stabilized: np.ndarray,
) -> np.ndarray:
    top = np.hstack([label_panel(original, "1. Original video"), label_panel(flow, "2. Optical flow overlay")])
    bottom = np.hstack([plot, label_panel(stabilized, "4. Jitter corrected stable video")])
    return np.vstack([top, bottom])


def write_shift_csv(
    path: Path,
    frame_numbers: list[int],
    shifts_x: list[float],
    shifts_y: list[float],
    corrections_x: list[float],
    corrections_y: list[float],
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_no", "shift_x_px", "shift_y_px", "correction_x_px", "correction_y_px"])
        writer.writerows(zip(frame_numbers, shifts_x, shifts_y, corrections_x, corrections_y))


def process_video(args: argparse.Namespace) -> None:
    input_path = args.input_video
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if args.rolling_window < 1:
        raise ValueError("--rolling-window must be at least 1")
    if args.max_points < 1:
        raise ValueError("--max-points must be at least 1")

    output_path = args.output or input_path.with_name(f"{input_path.stem}_jitter_analysis.mp4")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0

    success, first_frame = cap.read()
    if not success:
        raise RuntimeError(f"Could not read the first frame from: {input_path}")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    active_points = detect_keypoints(
        previous_gray,
        max_points=args.max_points,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
    )
    if len(active_points) == 0:
        raise RuntimeError("No keypoints detected in the first frame.")

    # A separate first-frame-reference track is kept only for the requested plot.
    reference_initial_points = active_points.copy()
    reference_previous_points = active_points.copy()

    frame_numbers = [1]
    shifts_x = [0.0]
    shifts_y = [0.0]
    corrections_x = [0.0]
    corrections_y = [0.0]
    recent_motions: deque[np.ndarray] = deque(maxlen=args.rolling_window)
    cumulative_correction = np.zeros(2, dtype=np.float32)

    first_plot = draw_shift_plot(width, height, frame_numbers, shifts_x, shifts_y, args.plot_history)
    first_panel = make_four_panel(first_frame, first_frame.copy(), first_plot, first_frame.copy())
    writer.write(first_panel)

    if args.display:
        cv2.imshow("video jitter analysis", first_panel)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            writer.release()
            return

    frame_no = 1
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        common_previous, common_current = track_points(previous_gray, gray, active_points)
        instant_motion = mean_delta(common_previous, common_current)
        recent_motions.append(instant_motion)
        if len(recent_motions) >= args.rolling_window:
            correction_step = moving_average(recent_motions)
        else:
            correction_step = instant_motion
        cumulative_correction += correction_step

        ref_initial_common, ref_current_common = track_reference_points(
            previous_gray,
            gray,
            reference_initial_points,
            reference_previous_points,
        )
        if len(ref_current_common) > 0:
            reference_initial_points = ref_initial_common
            reference_previous_points = ref_current_common
            shift = mean_delta(reference_initial_points, ref_current_common)
        else:
            shift = np.array([np.nan, np.nan], dtype=np.float32)
            reference_previous_points = np.empty((0, 1, 2), dtype=np.float32)
            reference_initial_points = np.empty((0, 1, 2), dtype=np.float32)

        stabilized = translate_frame(frame, cumulative_correction)
        flow = draw_flow_vectors(frame, common_previous, common_current, args.vector_stride)
        active_points = refresh_keypoints(
            gray,
            common_current,
            max_points=args.max_points,
            min_active_points=args.min_active_points,
            quality_level=args.quality_level,
            min_distance=args.min_distance,
        )

        frame_numbers.append(frame_no)
        shifts_x.append(float(shift[0]))
        shifts_y.append(float(shift[1]))
        corrections_x.append(float(cumulative_correction[0]))
        corrections_y.append(float(cumulative_correction[1]))

        plot = draw_shift_plot(width, height, frame_numbers, shifts_x, shifts_y, args.plot_history)
        panel = make_four_panel(frame, flow, plot, stabilized)
        writer.write(panel)

        if args.display:
            cv2.imshow("video jitter analysis", panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        previous_gray = gray

        if len(active_points) == 0:
            active_points = detect_keypoints(
                gray,
                max_points=args.max_points,
                quality_level=args.quality_level,
                min_distance=args.min_distance,
            )

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()
    if args.csv:
        write_shift_csv(args.csv, frame_numbers, shifts_x, shifts_y, corrections_x, corrections_y)

    print(f"Wrote four-quadrant analysis video: {output_path}")
    if args.csv:
        print(f"Wrote shift/correction CSV: {args.csv}")


def main() -> None:
    process_video(parse_args())


if __name__ == "__main__":
    main()
