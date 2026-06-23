#!/usr/bin/env python3
"""Stabilize x/y jitter with top-100 Key.Net keypoint shifts.

For each frame, this script extracts the top Key.Net keypoints, matches them to
the immediate previous frame, averages the matched x/y shifts, and translates
the current frame back so the keypoints align with the current reference segment.
When enough top keypoints change, the current frame becomes the new reference.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TOP_K = 100
DEFAULT_RESET_THRESHOLD = 10
DEFAULT_NUM_FRAMES = 100
DEFAULT_CROP_SIZE = 0
DEFAULT_PROCESS_SIZE = 512
DEFAULT_ANALYSIS_SIZE = 1024


@dataclass
class FeatureSet:
    points: np.ndarray
    descriptors: np.ndarray


@dataclass
class FrameResult:
    frame_no: int
    shift_x: float
    shift_y: float
    cumulative_x: float
    cumulative_y: float
    correction_x: float
    correction_y: float
    matched_count: int
    changed_count: int
    reset_reference: bool


class KeyNetTopKExtractor:
    """Extract top-K Key.Net keypoints and HardNet descriptors using Kornia."""

    def __init__(self, top_k: int, device: str) -> None:
        print("Loading Key.Net-HardNet model...", flush=True)
        try:
            import torch
            import kornia.feature as kornia_feature
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "This script requires Kornia and PyTorch. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch = torch
        self.kornia_feature = kornia_feature
        self.device = torch.device(device)
        self.model = kornia_feature.KeyNetHardNet(
            num_features=top_k,
            upright=True,
            device=self.device,
        ).to(self.device)
        self.model.eval()
        print(f"Loaded Key.Net-HardNet model on {self.device}.", flush=True)

    def extract(self, frame: np.ndarray) -> FeatureSet:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tensor = self.torch.from_numpy(gray.astype(np.float32) / 255.0)
        tensor = tensor[None, None].to(self.device)
        with self.torch.inference_mode():
            lafs, _responses, descriptors = self.model(tensor)
            centers = self.kornia_feature.get_laf_center(lafs)[0]

        points = centers.detach().cpu().numpy().astype(np.float32)
        descriptors_np = descriptors[0].detach().cpu().numpy().astype(np.float32)
        if len(points) == 0 or len(descriptors_np) == 0:
            return FeatureSet(
                points=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 128), dtype=np.float32),
            )
        return FeatureSet(points=points, descriptors=descriptors_np)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stabilize azimuth/elevation x/y jitter by averaging top-100 "
            "Key.Net keypoint shifts between immediate consecutive frames."
        )
    )
    parser.add_argument("input_video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Four-quadrant analysis video path. Defaults to '<input>_top100_analysis.mp4'.",
    )
    parser.add_argument(
        "--stable-output",
        type=Path,
        help="Optional stabilized-only video path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV path for per-frame shifts and reference resets.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames to process from the start (default: {DEFAULT_NUM_FRAMES}; 0 means full video).",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        help=(
            "Optional center-crop size before resizing. Use 0 to process the "
            f"full frame without cropping (default: {DEFAULT_CROP_SIZE})."
        ),
    )
    parser.add_argument(
        "--process-size",
        type=int,
        default=DEFAULT_PROCESS_SIZE,
        help=(
            "Resize every frame to this square size before keypoint extraction, "
            f"stabilization, and stable output (default: {DEFAULT_PROCESS_SIZE})."
        ),
    )
    parser.add_argument(
        "--analysis-size",
        type=int,
        default=DEFAULT_ANALYSIS_SIZE,
        help=(
            "Final square size of the four-quadrant analysis video "
            f"(default: {DEFAULT_ANALYSIS_SIZE})."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top Key.Net keypoints per frame (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--reset-threshold",
        type=int,
        default=DEFAULT_RESET_THRESHOLD,
        help=(
            "Reset the reference when at least this many top keypoints are not "
            f"matched to the previous set (default: {DEFAULT_RESET_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--match-ratio",
        type=float,
        default=0.85,
        help="Mutual Lowe-ratio threshold for descriptor matching (default: 0.85).",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=8,
        help="Minimum matches needed to use an average shift (default: 8).",
    )
    parser.add_argument(
        "--max-shift",
        type=float,
        default=80.0,
        help="Ignore a frame-to-frame average shift larger than this many pixels (default: 80).",
    )
    parser.add_argument(
        "--plot-history",
        type=int,
        default=240,
        help="Number of recent frames shown in the shift plot quadrant (default: 240).",
    )
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=4.0,
        help="Scale drawn keypoint-motion vectors so small shifts are visible (default: 4).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the four-quadrant output while processing. Press q to stop.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N frames (default: 1).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for Kornia Key.Net inference (default: auto).",
    )
    return parser.parse_args()


def match_top_keypoints(
    previous: FeatureSet,
    current: FeatureSet,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mutually matched previous/current keypoint coordinates."""
    if len(previous.points) < 2 or len(current.points) < 2:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(previous.descriptors, current.descriptors, k=2)
    backward = matcher.knnMatch(current.descriptors, previous.descriptors, k=2)
    ratio = float(np.clip(ratio, 0.01, 1.0))

    forward_good: dict[int, int] = {}
    for pair in forward:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            forward_good[pair[0].queryIdx] = pair[0].trainIdx

    backward_good: dict[int, int] = {}
    for pair in backward:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            backward_good[pair[0].queryIdx] = pair[0].trainIdx

    prev_points: list[np.ndarray] = []
    curr_points: list[np.ndarray] = []
    for prev_idx, curr_idx in forward_good.items():
        if backward_good.get(curr_idx) == prev_idx:
            prev_points.append(previous.points[prev_idx])
            curr_points.append(current.points[curr_idx])

    if not prev_points:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    return np.asarray(prev_points, dtype=np.float32), np.asarray(curr_points, dtype=np.float32)


def average_shift(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    min_matches: int,
    max_shift: float,
) -> tuple[np.ndarray, bool]:
    if len(previous_points) < min_matches:
        return np.zeros(2, dtype=np.float32), False
    shifts = current_points - previous_points
    mean_shift = np.mean(shifts, axis=0).astype(np.float32)
    if not np.all(np.isfinite(mean_shift)):
        return np.zeros(2, dtype=np.float32), False
    if float(np.linalg.norm(mean_shift)) > max_shift:
        return np.zeros(2, dtype=np.float32), False
    return mean_shift, True


def translate_with_zero_padding(frame: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Translate a frame while zero-padding newly exposed image regions."""
    dx, dy = float(correction[0]), float(correction[1])
    transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        transform,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def center_crop_or_pad_square(frame: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to size x size, zero-padding first if the frame is smaller."""
    height, width = frame.shape[:2]
    crop_width = min(width, size)
    crop_height = min(height, size)
    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    cropped = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]

    if crop_width == size and crop_height == size:
        return cropped

    output = np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
    paste_x = (size - crop_width) // 2
    paste_y = (size - crop_height) // 2
    output[paste_y : paste_y + crop_height, paste_x : paste_x + crop_width] = cropped
    return output


def preprocess_frame(frame: np.ndarray, crop_size: int, process_size: int) -> np.ndarray:
    """Optionally center-crop, then resize full content to processing size."""
    source = center_crop_or_pad_square(frame, crop_size) if crop_size > 0 else frame
    if source.shape[0] == process_size and source.shape[1] == process_size:
        return source
    return cv2.resize(source, (process_size, process_size), interpolation=cv2.INTER_AREA)


def make_analysis_frame(panel: np.ndarray, analysis_size: int) -> np.ndarray:
    """Resize the four-panel diagnostic view to the requested analysis size."""
    if panel.shape[0] == analysis_size and panel.shape[1] == analysis_size:
        return panel
    return cv2.resize(panel, (analysis_size, analysis_size), interpolation=cv2.INTER_AREA)


def draw_matches(
    frame: np.ndarray,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    mean_shift: np.ndarray,
    changed_count: int,
    reset_reference: bool,
    vector_scale: float,
    top_k: int,
) -> np.ndarray:
    overlay = frame.copy()
    vector_scale = max(1.0, float(vector_scale))
    if len(previous_points) > 0:
        shifts = current_points - previous_points
        for current_point, shift in zip(current_points, shifts):
            p0 = tuple(np.round(current_point).astype(int))
            p1 = tuple(np.round(current_point + shift * vector_scale).astype(int))
            cv2.arrowedLine(overlay, p0, p1, (0, 0, 0), 4, tipLength=0.35)
            cv2.arrowedLine(overlay, p0, p1, (0, 255, 255), 2, tipLength=0.35)
            cv2.circle(overlay, p0, 3, (0, 80, 255), -1)

    lines = [
        f"matched top keypoints: {len(current_points)} / {top_k}",
        f"avg shift dx={mean_shift[0]:.2f}, dy={mean_shift[1]:.2f}",
        f"changed={changed_count}, reset={'yes' if reset_reference else 'no'}",
    ]
    draw_text_lines(overlay, lines, origin_y=58)
    return overlay


def draw_text_lines(frame: np.ndarray, lines: list[str], origin_y: int) -> None:
    for index, line in enumerate(lines):
        y = origin_y + index * 24
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def draw_shift_plot(
    width: int,
    height: int,
    frame_numbers: list[int],
    shift_x: list[float],
    shift_y: list[float],
    correction_x: list[float],
    correction_y: list[float],
    history: int,
) -> np.ndarray:
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
        "3. Top-100 keypoint x/y shifts",
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
    series = [
        np.asarray(shift_x[-history:], dtype=np.float32),
        np.asarray(shift_y[-history:], dtype=np.float32),
        np.asarray(correction_x[-history:], dtype=np.float32),
        np.asarray(correction_y[-history:], dtype=np.float32),
    ]
    valid_values = np.concatenate([values[np.isfinite(values)] for values in series if np.any(np.isfinite(values))])
    if len(valid_values) == 0:
        return plot

    max_abs = max(1.0, float(np.nanmax(np.abs(valid_values))))
    y_mid = int(round((y0 + y1) / 2.0))
    cv2.line(plot, (x0, y_mid), (x1, y_mid), (190, 190, 190), 1)

    frame_min = recent_frames[0]
    frame_max = max(recent_frames[-1], frame_min + 1)

    def to_screen(frame_no: int, value: float) -> tuple[int, int]:
        x = x0 + int(round((frame_no - frame_min) / (frame_max - frame_min) * (x1 - x0)))
        y = y_mid - int(round(value / max_abs * ((y0 - y1) / 2.0)))
        return x, int(np.clip(y, y1, y0))

    def draw_trace(values: np.ndarray, color: tuple[int, int, int], dotted: bool = False) -> None:
        points = [
            to_screen(frame_no, float(value))
            for frame_no, value in zip(recent_frames, values)
            if np.isfinite(value)
        ]
        for segment_no, (start, end) in enumerate(zip(points, points[1:])):
            if not dotted or segment_no % 2 == 0:
                cv2.line(plot, start, end, color, 2, cv2.LINE_AA)

    draw_trace(series[0], (30, 80, 230))
    draw_trace(series[1], (30, 170, 30))
    draw_trace(series[2], (120, 150, 255), dotted=True)
    draw_trace(series[3], (120, 220, 120), dotted=True)

    legend_items = [
        ("shift x", (30, 80, 230), False),
        ("shift y", (30, 170, 30), False),
        ("correction x", (120, 150, 255), True),
        ("correction y", (120, 220, 120), True),
    ]
    legend_x = x0 + 8
    legend_y = y0 + 24
    for index, (label, color, dotted) in enumerate(legend_items):
        item_x = legend_x + (index % 2) * 175
        item_y = legend_y + (index // 2) * 20
        if dotted:
            cv2.line(plot, (item_x, item_y), (item_x + 16, item_y), color, 2, cv2.LINE_AA)
            cv2.line(plot, (item_x + 22, item_y), (item_x + 38, item_y), color, 2, cv2.LINE_AA)
        else:
            cv2.line(plot, (item_x, item_y), (item_x + 38, item_y), color, 2, cv2.LINE_AA)
        cv2.putText(plot, label, (item_x + 46, item_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    cv2.putText(
        plot,
        f"range +/- {max_abs:.1f} px",
        (x0 + 8, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return plot


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


def make_four_panel(original: np.ndarray, overlay: np.ndarray, plot: np.ndarray, stabilized: np.ndarray) -> np.ndarray:
    top = np.hstack([label_panel(original, "1. Original video"), label_panel(overlay, "2. Top-100 Key.Net shifts")])
    bottom = np.hstack([plot, label_panel(stabilized, "4. Zero-padded stabilized video")])
    return np.vstack([top, bottom])


def write_results_csv(path: Path, results: list[FrameResult]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_no",
                "shift_x_px",
                "shift_y_px",
                "cumulative_shift_x_px",
                "cumulative_shift_y_px",
                "correction_x_px",
                "correction_y_px",
                "matched_count",
                "changed_count",
                "reset_reference",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.frame_no,
                    result.shift_x,
                    result.shift_y,
                    result.cumulative_x,
                    result.cumulative_y,
                    result.correction_x,
                    result.correction_y,
                    result.matched_count,
                    result.changed_count,
                    int(result.reset_reference),
                ]
            )


def process_video(args: argparse.Namespace) -> None:
    if not args.input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {args.input_video}")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.reset_threshold < 0:
        raise ValueError("--reset-threshold must be non-negative")
    if args.num_frames < 0:
        raise ValueError("--num-frames must be non-negative")
    if args.crop_size < 0:
        raise ValueError("--crop-size must be non-negative")
    if 0 < args.crop_size < 16:
        raise ValueError("--crop-size must be 0 or at least 16")
    if args.process_size < 16:
        raise ValueError("--process-size must be at least 16")
    if args.analysis_size < 16:
        raise ValueError("--analysis-size must be at least 16")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    output_path = args.output or args.input_video.with_name(f"{args.input_video.stem}_top100_analysis.mp4")
    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0

    success, first_frame = cap.read()
    if not success:
        raise RuntimeError(f"Could not read first frame: {args.input_video}")

    if args.crop_size > 0:
        print(
            f"Center-cropping frames to {args.crop_size}x{args.crop_size}, "
            f"then resizing to {args.process_size}x{args.process_size} before processing.",
            flush=True,
        )
    else:
        print(
            f"Using full frames without cropping; resizing to "
            f"{args.process_size}x{args.process_size} before processing.",
            flush=True,
        )
    print(
        f"Saving analysis video at {args.analysis_size}x{args.analysis_size}.",
        flush=True,
    )
    first_frame = preprocess_frame(first_frame, args.crop_size, args.process_size)
    height, width = first_frame.shape[:2]
    panel_writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.analysis_size, args.analysis_size),
    )
    if not panel_writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    stable_writer = None
    if args.stable_output:
        stable_writer = cv2.VideoWriter(
            str(args.stable_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not stable_writer.isOpened():
            raise RuntimeError(f"Could not open stabilized writer: {args.stable_output}")

    extractor = KeyNetTopKExtractor(args.top_k, args.device)
    print("Extracting top keypoints for frame 1...", flush=True)
    previous_features = extractor.extract(first_frame)
    print(f"frame 1: extracted {len(previous_features.points)} keypoints", flush=True)
    reference_correction = np.zeros(2, dtype=np.float32)
    cumulative_shift = np.zeros(2, dtype=np.float32)

    results = [
        FrameResult(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(previous_features.points), 0, True)
    ]
    frame_numbers = [1]
    shift_x = [0.0]
    shift_y = [0.0]
    correction_x = [0.0]
    correction_y = [0.0]

    first_plot = draw_shift_plot(width, height, frame_numbers, shift_x, shift_y, correction_x, correction_y, args.plot_history)
    first_overlay = first_frame.copy()
    draw_text_lines(first_overlay, [f"reference frame, top keypoints: {len(previous_features.points)}"], origin_y=58)
    first_panel = make_four_panel(first_frame, first_overlay, first_plot, first_frame.copy())
    panel_writer.write(make_analysis_frame(first_panel, args.analysis_size))
    if stable_writer is not None:
        stable_writer.write(first_frame)

    max_frames = None if args.num_frames == 0 else args.num_frames
    frame_no = 1
    while max_frames is None or frame_no < max_frames:
        success, frame = cap.read()
        if not success:
            break
        frame_no += 1
        frame = preprocess_frame(frame, args.crop_size, args.process_size)

        if frame_no % args.progress_every == 0:
            print(f"frame {frame_no}: extracting top keypoints...", flush=True)
        current_features = extractor.extract(frame)
        matched_prev, matched_curr = match_top_keypoints(previous_features, current_features, args.match_ratio)
        mean_shift, valid_shift = average_shift(matched_prev, matched_curr, args.min_matches, args.max_shift)
        if valid_shift:
            cumulative_shift += mean_shift
        else:
            mean_shift = np.zeros(2, dtype=np.float32)

        correction = reference_correction - cumulative_shift
        stabilized = translate_with_zero_padding(frame, correction)

        matched_count = int(len(matched_curr))
        changed_count = max(0, min(args.top_k, args.top_k - matched_count))
        reset_reference = changed_count >= args.reset_threshold or not valid_shift
        if frame_no % args.progress_every == 0:
            print(
                f"frame {frame_no}: keypoints={len(current_features.points)}, "
                f"matches={matched_count}, changed={changed_count}, "
                f"shift=({mean_shift[0]:.2f}, {mean_shift[1]:.2f}), "
                f"correction=({correction[0]:.2f}, {correction[1]:.2f}), "
                f"reset={'yes' if reset_reference else 'no'}",
                flush=True,
            )

        overlay = draw_matches(
            frame,
            matched_prev,
            matched_curr,
            mean_shift,
            changed_count,
            reset_reference,
            args.vector_scale,
            args.top_k,
        )

        frame_numbers.append(frame_no)
        shift_x.append(float(mean_shift[0]))
        shift_y.append(float(mean_shift[1]))
        correction_x.append(float(correction[0]))
        correction_y.append(float(correction[1]))
        plot = draw_shift_plot(width, height, frame_numbers, shift_x, shift_y, correction_x, correction_y, args.plot_history)

        stabilized_for_panel = stabilized.copy()
        draw_text_lines(
            stabilized_for_panel,
            [
                f"correction dx={correction[0]:.2f}, dy={correction[1]:.2f}",
                "black borders are zero padding",
            ],
            origin_y=58,
        )
        panel = make_four_panel(frame, overlay, plot, stabilized_for_panel)
        analysis_frame = make_analysis_frame(panel, args.analysis_size)
        panel_writer.write(analysis_frame)
        if stable_writer is not None:
            stable_writer.write(stabilized)

        results.append(
            FrameResult(
                frame_no,
                float(mean_shift[0]),
                float(mean_shift[1]),
                float(cumulative_shift[0]),
                float(cumulative_shift[1]),
                float(correction[0]),
                float(correction[1]),
                matched_count,
                changed_count,
                reset_reference,
            )
        )

        if args.display:
            cv2.imshow("top-100 keypoint stabilizer", analysis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        previous_features = current_features
        if reset_reference:
            reference_correction = correction.copy()
            cumulative_shift = np.zeros(2, dtype=np.float32)
            print(f"frame {frame_no}: reset reference, changed top keypoints={changed_count}", flush=True)

    cap.release()
    panel_writer.release()
    if stable_writer is not None:
        stable_writer.release()
    if args.display:
        cv2.destroyAllWindows()
    if args.csv:
        write_results_csv(args.csv, results)

    print(f"Wrote {args.analysis_size}x{args.analysis_size} four-quadrant analysis video: {output_path}")
    if args.stable_output:
        print(f"Wrote stabilized-only video: {args.stable_output}")
    if args.csv:
        print(f"Wrote shift CSV: {args.csv}")


def main() -> None:
    process_video(parse_args())


if __name__ == "__main__":
    main()
