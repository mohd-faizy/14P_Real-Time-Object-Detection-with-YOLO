"""
YOLO Object Detection Utilities
===============================
This module provides helper functions and classes for object detection,
visualization, and performance metrics, designed to be clean, robust,
and easy to understand for learners.

Key Concepts:
- Bounding Box: Rectangular coordinates enclosing a detected object [x1, y1, x2, y2].
- Confidence Score: Probability (0.0 to 1.0) that an object is present in the box.
- Class ID & Label: Integer identifier and human-readable name of the detected class.
- FPS (Frames Per Second): Metric representing real-time processing throughput.
"""

from typing import List, Tuple, Dict, Optional, Union
import time
import cv2 as cv
import numpy as np


class FPSCalculator:
    """
    Measures and smooths real-time Frames Per Second (FPS) for video processing.

    Example:
        fps_calc = FPSCalculator()
        while capturing:
            # ... process frame ...
            current_fps = fps_calc.tick()
    """
    def __init__(self, smoothing_window: int = 30) -> None:
        """
        Args:
            smoothing_window: Number of recent frames over which to average FPS.
        """
        self.smoothing_window = smoothing_window
        self.timestamps: List[float] = []
        self.total_frames: int = 0
        self.start_time: float = time.time()

    def tick(self) -> float:
        """
        Record a frame completion and return the smoothed current FPS.
        """
        now = time.time()
        self.total_frames += 1
        self.timestamps.append(now)

        # Keep only timestamps within the smoothing window
        if len(self.timestamps) > self.smoothing_window:
            self.timestamps.pop(0)

        # Calculate FPS based on timestamps window
        if len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0]
            return len(self.timestamps) / duration if duration > 0 else 0.0
        return 0.0

    @property
    def average_fps(self) -> float:
        """Overall average FPS since initialization."""
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0


def get_color_palette(num_classes: int = 80) -> List[Tuple[int, int, int]]:
    """
    Generates a visually distinct color palette using the HSV color space.

    By spreading hues evenly around the 360-degree color wheel, classes
    get vibrant, easily distinguishable colors.

    Args:
        num_classes: Total number of unique classes to generate colors for.

    Returns:
        List of (B, G, R) color tuples formatted for OpenCV.
    """
    colors: List[Tuple[int, int, int]] = []
    for i in range(num_classes):
        # Evenly space the hue between 0 and 179 (OpenCV HSV hue range is 0-179)
        hue = int((i * 180) / max(num_classes, 1))
        # High saturation (200-255) and value (200-255) for vivid colors
        sat = 220
        val = 240
        hsv_pixel = np.uint8([[[hue, sat, val]]])
        bgr_pixel = cv.cvtColor(hsv_pixel, cv.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])))
    return colors


def draw_bounding_box(
    image: np.ndarray,
    box: Union[Tuple[int, int, int, int], List[int]],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draws an annotated bounding box with a clean label banner on the image.

    Args:
        image: The image array (BGR format).
        box: Coordinates (x1, y1, x2, y2) in pixels.
        label: Class name string (e.g., 'car', 'person').
        confidence: Prediction confidence score between 0.0 and 1.0.
        color: BGR tuple for rectangle and label background.
        line_thickness: Border thickness in pixels.

    Returns:
        Annotated image.
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # 1. Draw the object bounding rectangle
    cv.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

    # 2. Format label text with confidence percentage
    text = f"{label} {confidence:.0%}"
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Calculate text dimensions to fit background banner
    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, font_thickness)

    # Determine banner top-left corner (flip inside the box if box is at top of frame)
    banner_y1 = max(0, y1 - text_h - baseline - 4)
    banner_y2 = y1 if y1 - text_h - baseline - 4 >= 0 else y1 + text_h + baseline + 4

    # 3. Draw colored label background banner
    cv.rectangle(
        image,
        (x1, banner_y1),
        (x1 + text_w + 6, banner_y2),
        color,
        -1  # Filled rectangle
    )

    # 4. Draw white text on top of the colored banner
    text_y = banner_y2 - baseline - 2 if y1 - text_h - baseline - 4 >= 0 else banner_y2 - 2
    cv.putText(
        image,
        text,
        (x1 + 3, text_y),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv.LINE_AA
    )

    return image


def draw_fps_overlay(
    image: np.ndarray,
    fps: float,
    model_name: Optional[str] = None
) -> np.ndarray:
    """
    Draws a semi-transparent HUD banner displaying real-time FPS and model name.

    Args:
        image: Frame to draw overlay on.
        fps: Current Frames Per Second.
        model_name: Optional name of the active model.

    Returns:
        Image with overlay applied.
    """
    text = f"FPS: {fps:.1f}"
    if model_name:
        text = f"{model_name} | {text}"

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (tw, th), base = cv.getTextSize(text, font, font_scale, font_thickness)

    # Draw semi-transparent black background box in top-left corner
    overlay = image.copy()
    cv.rectangle(overlay, (10, 10), (20 + tw, 20 + th + base), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Draw bright green text
    cv.putText(
        image,
        text,
        (15, 15 + th),
        font,
        font_scale,
        (0, 255, 128),
        font_thickness,
        cv.LINE_AA
    )
    return image

