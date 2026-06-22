#!/usr/bin/env python3
"""Analyze and stabilize x/y video jitter with Key.Net features.

This script uses Kornia's KeyNetHardNet feature extractor to detect Key.Net
keypoints and HardNet descriptors. Consecutive frames are matched, a robust
partial affine camera motion is estimated with RANSAC, the full camera
trajectory is smoothed, and every frame is warped toward that smooth trajectory.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MAX_FEATURES = 1500
DEFAULT_MIN_MATCHES = 12
DEFAULT_SMOOTHING_RADIUS = 15
DEFAULT_NUM_FRAMES = 100
DEFAULT_STABILIZATION_MODE = "frame-to-frame"


@dataclass
class FeatureSet:
    points: np.ndarray
    descriptors: np.ndarray


@dataclass
class MotionEstimate:
    dx: float
    dy: float
    da: float
    matched_prev: np.ndarray
    matched_curr: np.ndarray
    inlier_mask: np.ndarray
    match_count: int
    inlier_count: int
    used_fallback: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a video, estimate Key.Net feature motion, plot x/y camera "
            "shift, overlay feature-motion vectors, and write a stabilized "
            "four-quadrant analysis video."
        )
    )
    parser.add_argument("input_video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output four-quadrant video path. Defaults to '<input>_keynet_stabilized.mp4'.",
    )
    parser.add_argument(
        "--stable-output",
        type=Path,
        help="Optional path for a stabilized-only video without quadrants.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV path for trajectory, smoothing, corrections, matches, and inliers.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the four-quadrant video while processing. Press q to stop.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=(
            "Number of frames to stabilize and plot from the start of the video "
            f"(default: {DEFAULT_NUM_FRAMES}). Use 0 to process the full video."
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help=f"Maximum Key.Net features per frame (default: {DEFAULT_MAX_FEATURES}).",
    )
    parser.add_argument(
        "--feature-max-size",
        type=int,
        default=960,
        help=(
            "Resize the longest image side before Key.Net extraction for speed, "
            "then scale keypoints back to full resolution (default: 960; use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--match-ratio",
        type=float,
        default=0.80,
        help="Lowe ratio threshold for HardNet descriptor matching (default: 0.80).",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=DEFAULT_MIN_MATCHES,
        help=f"Minimum matches/inliers before trusting affine estimation (default: {DEFAULT_MIN_MATCHES}).",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=4.0,
        help="RANSAC reprojection threshold in pixels (default: 4).",
    )
    parser.add_argument(
        "--smoothing-radius",
        type=int,
        default=DEFAULT_SMOOTHING_RADIUS,
        help=(
            "Centered trajectory smoothing radius in frames. The window is "
            "2*radius+1 frames (default: 15). Use 2 for a five-frame centered window."
        ),
    )
    parser.add_argument(
        "--stabilization-mode",
        choices=["frame-to-frame", "smooth-trajectory"],
        default=DEFAULT_STABILIZATION_MODE,
        help=(
            "Correction strategy. 'frame-to-frame' accumulates the measured x/y "
            "shift and aligns every frame back to frame 1; 'smooth-trajectory' "
            "warps to a smoothed camera path (default: frame-to-frame)."
        ),
    )
    parser.add_argument(
        "--border-scale",
        type=float,
        default=1.03,
        help="Slight zoom applied to stabilized frames to hide borders (default: 1.03).",
    )
    parser.add_argument(
        "--vector-stride",
        type=int,
        default=2,
        help="Draw every Nth matched Key.Net motion vector to reduce clutter (default: 2).",
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=4.0,
        help="Scale feature-motion vectors before drawing (default: 4).",
    )
    parser.add_argument(
        "--flow-thickness",
        type=int,
        default=2,
        help="Feature-motion vector line thickness in pixels (default: 2).",
    )
    parser.add_argument(
        "--plot-history",
        type=int,
        default=240,
        help="Number of recent frames shown in the shift quadrant (default: 240).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Kornia Key.Net inference (default: auto).",
    )
    parser.add_argument(
        "--no-status-text",
        action="store_true",
        help="Do not draw diagnostic text on overlay/stabilized panels.",
    )
    return parser.parse_args()


class KeyNetHardNetExtractor:
    """Small wrapper around Kornia's KeyNetHardNet local feature model."""

    def __init__(self, max_features: int, feature_max_size: int, device: str) -> None:
        try:
            import torch
            import kornia.feature as kornia_feature
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Key.Net support requires Kornia and PyTorch. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch = torch
        self.kornia_feature = kornia_feature
        self.device = torch.device(device)
        self.feature_max_size = max(0, feature_max_size)
        self.model = kornia_feature.KeyNetHardNet(
            num_features=max_features,
            upright=True,
            device=self.device,
        ).to(self.device)
        self.model.eval()

    def extract(self, frame: np.ndarray) -> FeatureSet:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        original_height, original_width = gray.shape[:2]
        scale = 1.0
        if self.feature_max_size > 0:
            longest_side = max(original_height, original_width)
            if longest_side > self.feature_max_size:
                scale = self.feature_max_size / float(longest_side)
                resized_width = max(1, int(round(original_width * scale)))
                resized_height = max(1, int(round(original_height * scale)))
                gray = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        tensor = self.torch.from_numpy(gray.astype(np.float32) / 255.0)
        tensor = tensor[None, None].to(self.device)

        with self.torch.inference_mode():
            lafs, _responses, descriptors = self.model(tensor)
            centers = self.kornia_feature.get_laf_center(lafs)[0]

        points = centers.detach().cpu().numpy().astype(np.float32)
        if scale != 1.0:
            points /= scale

        descriptors_np = descriptors[0].detach().cpu().numpy().astype(np.float32)
        if len(points) == 0 or len(descriptors_np) == 0:
            return FeatureSet(
                points=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 128), dtype=np.float32),
            )
        return FeatureSet(points=points, descriptors=descriptors_np)


def match_features(
    previous: FeatureSet,
    current: FeatureSet,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match HardNet descriptors with a mutual Lowe-ratio test."""
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

    matched_prev: list[np.ndarray] = []
    matched_curr: list[np.ndarray] = []
    for prev_idx, curr_idx in forward_good.items():
        if backward_good.get(curr_idx) == prev_idx:
            matched_prev.append(previous.points[prev_idx])
            matched_curr.append(current.points[curr_idx])

    if not matched_prev:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    return np.asarray(matched_prev, dtype=np.float32), np.asarray(matched_curr, dtype=np.float32)


def estimate_motion(
    matched_prev: np.ndarray,
    matched_curr: np.ndarray,
    min_matches: int,
    ransac_threshold: float,
) -> MotionEstimate:
    """Estimate x/y camera translation between adjacent frames.

    The user's jitter model is azimuth/elevation only, so the stabilizer ignores
    rotation and scale. RANSAC is used only to reject bad matches; the final
    motion is the median x/y displacement of inlier Key.Net matches.
    """
    match_count = int(len(matched_prev))
    empty_mask = np.zeros(match_count, dtype=bool)
    if match_count < min_matches:
        return MotionEstimate(0.0, 0.0, 0.0, matched_prev, matched_curr, empty_mask, match_count, 0, True)

    transform, inliers = cv2.estimateAffinePartial2D(
        matched_prev,
        matched_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    deltas = matched_curr - matched_prev
    if transform is None or inliers is None:
        delta = np.median(deltas, axis=0)
        return MotionEstimate(
            float(delta[0]),
            float(delta[1]),
            0.0,
            matched_prev,
            matched_curr,
            empty_mask,
            match_count,
            0,
            True,
        )

    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(np.sum(inlier_mask))
    if inlier_count < min_matches:
        if match_count > 0:
            delta = np.median(deltas, axis=0)
            return MotionEstimate(
                float(delta[0]),
                float(delta[1]),
                0.0,
                matched_prev,
                matched_curr,
                inlier_mask,
                match_count,
                inlier_count,
                True,
            )
        return MotionEstimate(0.0, 0.0, 0.0, matched_prev, matched_curr, inlier_mask, match_count, 0, True)

    inlier_delta = np.median(deltas[inlier_mask], axis=0)
    return MotionEstimate(
        float(inlier_delta[0]),
        float(inlier_delta[1]),
        0.0,
        matched_prev,
        matched_curr,
        inlier_mask,
        match_count,
        inlier_count,
    )


def centered_moving_average(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Smooth a trajectory with a centered moving average and edge clipping."""
    if len(trajectory) == 0:
        return trajectory.copy()
    radius = max(0, int(radius))
    if radius == 0:
        return trajectory.copy()

    smoothed = np.empty_like(trajectory, dtype=np.float32)
    for index in range(len(trajectory)):
        start = max(0, index - radius)
        end = min(len(trajectory), index + radius + 1)
        smoothed[index] = np.mean(trajectory[start:end], axis=0)
    return smoothed


def estimate_video_motion(
    input_path: Path,
    extractor: KeyNetHardNetExtractor,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[MotionEstimate], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0

    success, previous_frame = cap.read()
    if not success:
        raise RuntimeError(f"Could not read the first frame from: {input_path}")

    height, width = previous_frame.shape[:2]
    previous_features = extractor.extract(previous_frame)

    transforms = [np.zeros(3, dtype=np.float32)]
    motions = [
        MotionEstimate(
            0.0,
            0.0,
            0.0,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=bool),
            0,
            0,
            False,
        )
    ]

    frame_no = 1
    max_frames = None if args.num_frames == 0 else args.num_frames
    while max_frames is None or frame_no < max_frames:
        success, frame = cap.read()
        if not success:
            break
        frame_no += 1
        current_features = extractor.extract(frame)
        matched_prev, matched_curr = match_features(previous_features, current_features, args.match_ratio)
        motion = estimate_motion(matched_prev, matched_curr, args.min_matches, args.ransac_threshold)
        transforms.append(np.array([motion.dx, motion.dy, motion.da], dtype=np.float32))
        motions.append(motion)
        previous_features = current_features
        print(
            f"estimated frame {frame_no}: matches={motion.match_count}, "
            f"inliers={motion.inlier_count}, dx={motion.dx:.2f}, dy={motion.dy:.2f}, "
            "translation-only"
        )

    cap.release()
    print(f"estimated motion for {len(transforms)} frame(s)")
    transforms_np = np.asarray(transforms, dtype=np.float32)
    trajectory = np.cumsum(transforms_np, axis=0)
    return transforms_np, trajectory, motions, fps, (width, height)


def correction_matrix(
    correction: np.ndarray,
    width: int,
    height: int,
    border_scale: float,
) -> np.ndarray:
    """Build a correction warp around the frame center."""
    dx, dy, da = [float(value) for value in correction]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, math.degrees(da), max(1.0, float(border_scale)))
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix.astype(np.float32)


def stabilize_frame(
    frame: np.ndarray,
    correction: np.ndarray,
    border_scale: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    matrix = correction_matrix(correction, width, height, border_scale)
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_feature_vectors(
    frame: np.ndarray,
    motion: MotionEstimate,
    stride: int,
    scale: float,
    thickness: int,
    show_status: bool,
) -> np.ndarray:
    overlay = frame.copy()
    stride = max(1, int(stride))
    scale = max(1.0, float(scale))
    thickness = max(1, int(thickness))

    if len(motion.matched_curr) > 0:
        inlier_mask = motion.inlier_mask
        if len(inlier_mask) != len(motion.matched_curr):
            inlier_mask = np.ones(len(motion.matched_curr), dtype=bool)
        for prev, curr, is_inlier in zip(
            motion.matched_prev[::stride],
            motion.matched_curr[::stride],
            inlier_mask[::stride],
        ):
            delta = curr - prev
            p0 = tuple(np.round(curr).astype(int))
            p1 = tuple(np.round(curr + delta * scale).astype(int))
            color = (0, 255, 255) if is_inlier else (80, 80, 255)
            cv2.arrowedLine(overlay, p0, p1, (0, 0, 0), thickness + 2, tipLength=0.35)
            cv2.arrowedLine(overlay, p0, p1, color, thickness, tipLength=0.35)
            cv2.circle(overlay, p0, thickness + 1, (0, 80, 255), -1)

    if show_status:
        fallback = " median fallback" if motion.used_fallback else ""
        draw_text_lines(
            overlay,
            [
                f"Key.Net-HardNet matches: {motion.match_count}, inliers: {motion.inlier_count}{fallback}",
                f"translation dx={motion.dx:.2f}, dy={motion.dy:.2f}",
            ],
            origin_y=58,
        )
    return overlay


def draw_text_lines(frame: np.ndarray, lines: list[str], origin_y: int = 64) -> np.ndarray:
    for index, line in enumerate(lines):
        y = origin_y + index * 24
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_shift_plot(
    width: int,
    height: int,
    frame_numbers: list[int],
    raw_x: list[float],
    raw_y: list[float],
    target_x: list[float],
    target_y: list[float],
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
        "3. Key.Net x/y shift correction",
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
        np.asarray(raw_x[-history:], dtype=np.float32),
        np.asarray(raw_y[-history:], dtype=np.float32),
        np.asarray(target_x[-history:], dtype=np.float32),
        np.asarray(target_y[-history:], dtype=np.float32),
    ]
    valid_values = np.concatenate([values[np.isfinite(values)] for values in series if np.any(np.isfinite(values))])
    if len(valid_values) == 0:
        return plot

    max_abs = max(1.0, float(np.nanmax(np.abs(valid_values))))
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

    cv2.putText(plot, "raw x/y", (x0 + 8, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 80, 230), 2)
    cv2.putText(plot, "target x/y", (x0 + 105, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 150, 90), 2)
    cv2.putText(
        plot,
        f"frames {frame_min}-{recent_frames[-1]}",
        (max(x0 + 235, x1 - 170), y0 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        1,
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


def make_four_panel(
    original: np.ndarray,
    flow: np.ndarray,
    plot: np.ndarray,
    stabilized: np.ndarray,
) -> np.ndarray:
    top = np.hstack([label_panel(original, "1. Original video"), label_panel(flow, "2. Key.Net motion overlay")])
    bottom = np.hstack([plot, label_panel(stabilized, "4. Frame-to-frame translation stabilized video")])
    return np.vstack([top, bottom])


def write_csv(
    path: Path,
    trajectory: np.ndarray,
    target: np.ndarray,
    corrections: np.ndarray,
    motions: list[MotionEstimate],
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_no",
                "raw_shift_x_px",
                "raw_shift_y_px",
                "raw_angle_deg",
                "target_shift_x_px",
                "target_shift_y_px",
                "target_angle_deg",
                "correction_x_px",
                "correction_y_px",
                "correction_angle_deg",
                "matches",
                "inliers",
                "used_fallback",
            ]
        )
        for index, (raw, target_frame, correction, motion) in enumerate(
            zip(trajectory, target, corrections, motions),
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    float(raw[0]),
                    float(raw[1]),
                    math.degrees(float(raw[2])),
                    float(target_frame[0]),
                    float(target_frame[1]),
                    math.degrees(float(target_frame[2])),
                    float(correction[0]),
                    float(correction[1]),
                    math.degrees(float(correction[2])),
                    motion.match_count,
                    motion.inlier_count,
                    int(motion.used_fallback),
                ]
            )


def write_outputs(
    input_path: Path,
    output_path: Path,
    stable_output_path: Path | None,
    fps: float,
    frame_size: tuple[int, int],
    trajectory: np.ndarray,
    target: np.ndarray,
    corrections: np.ndarray,
    motions: list[MotionEstimate],
    args: argparse.Namespace,
) -> None:
    width, height = frame_size
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for output: {input_path}")

    panel_writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height * 2),
    )
    if not panel_writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    stable_writer = None
    if stable_output_path is not None:
        stable_writer = cv2.VideoWriter(
            str(stable_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not stable_writer.isOpened():
            raise RuntimeError(f"Could not open stabilized-only writer: {stable_output_path}")

    frame_numbers: list[int] = []
    raw_x: list[float] = []
    raw_y: list[float] = []
    target_x: list[float] = []
    target_y: list[float] = []

    frame_index = 0
    output_frame_count = min(len(corrections), args.num_frames) if args.num_frames > 0 else len(corrections)
    while frame_index < output_frame_count:
        success, frame = cap.read()
        if not success:
            break

        correction = corrections[frame_index]
        stabilized = stabilize_frame(frame, correction, args.border_scale)
        if stable_writer is not None:
            stable_writer.write(stabilized)

        motion = motions[frame_index]
        flow = draw_feature_vectors(
            frame,
            motion,
            args.vector_stride,
            args.flow_scale,
            args.flow_thickness,
            not args.no_status_text,
        )

        frame_numbers.append(frame_index + 1)
        raw_x.append(float(trajectory[frame_index, 0]))
        raw_y.append(float(trajectory[frame_index, 1]))
        target_x.append(float(target[frame_index, 0]))
        target_y.append(float(target[frame_index, 1]))
        plot = draw_shift_plot(
            width,
            height,
            frame_numbers,
            raw_x,
            raw_y,
            target_x,
            target_y,
            args.plot_history,
        )

        if not args.no_status_text:
            draw_text_lines(
                stabilized,
                [
                    f"raw dx={trajectory[frame_index, 0]:.2f}, dy={trajectory[frame_index, 1]:.2f}",
                    f"target dx={target[frame_index, 0]:.2f}, dy={target[frame_index, 1]:.2f}",
                    f"applied inverse shift dx={correction[0]:.2f}, dy={correction[1]:.2f}",
                ],
                origin_y=58,
            )

        panel = make_four_panel(frame, flow, plot, stabilized)
        panel_writer.write(panel)

        if args.display:
            cv2.imshow("video jitter analysis", panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    panel_writer.release()
    if stable_writer is not None:
        stable_writer.release()
    if args.display:
        cv2.destroyAllWindows()


def process_video(args: argparse.Namespace) -> None:
    if not args.input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {args.input_video}")
    if args.max_features < 1:
        raise ValueError("--max-features must be at least 1")
    if args.num_frames < 0:
        raise ValueError("--num-frames must be non-negative")
    if args.smoothing_radius < 0:
        raise ValueError("--smoothing-radius must be non-negative")
    if args.border_scale < 1.0:
        raise ValueError("--border-scale must be at least 1.0")

    output_path = args.output or args.input_video.with_name(f"{args.input_video.stem}_keynet_stabilized.mp4")
    extractor = KeyNetHardNetExtractor(args.max_features, args.feature_max_size, args.device)
    frame_limit = "all frames" if args.num_frames == 0 else f"first {args.num_frames} frame(s)"
    print(f"Processing {frame_limit}")

    transforms, trajectory, motions, fps, frame_size = estimate_video_motion(args.input_video, extractor, args)
    if args.stabilization_mode == "frame-to-frame":
        target = np.zeros_like(trajectory, dtype=np.float32)
        corrections = -trajectory
        corrections[:, 2] = 0.0
    else:
        target = centered_moving_average(trajectory, args.smoothing_radius)
        corrections = target - trajectory
        corrections[:, 2] = 0.0

    write_outputs(
        args.input_video,
        output_path,
        args.stable_output,
        fps,
        frame_size,
        trajectory,
        target,
        corrections,
        motions,
        args,
    )
    if args.csv:
        write_csv(args.csv, trajectory, target, corrections, motions)

    print(f"Wrote four-quadrant Key.Net stabilization video: {output_path}")
    if args.stable_output:
        print(f"Wrote stabilized-only video: {args.stable_output}")
    if args.csv:
        print(f"Wrote trajectory CSV: {args.csv}")


def main() -> None:
    process_video(parse_args())


if __name__ == "__main__":
    main()
