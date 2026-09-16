"""
Real-Time Object Detection with YOLO26
======================================
This script demonstrates real-time object detection on videos, images, and
live camera streams using the state-of-the-art YOLO26 model with OpenCV.

Designed to be clean, modular, and pedagogical for students and learners.

Quickstart Examples:
--------------------
1. Video Detection:
   python yolo.py --video-path data/videos/Chicago_360p.mp4

2. Static Image Detection:
   python yolo.py --image-path data/images/dog.jpg

3. Live Webcam Stream:
   python yolo.py --video-path 0 --show
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Union, List, Tuple
import cv2 as cv
import numpy as np

# Import helper functions from local utilities module
from yolo_utils import (
    FPSCalculator,
    get_color_palette,
    draw_bounding_box,
    draw_fps_overlay,
)


# Default model paths
DEFAULT_MODEL_WEIGHTS = Path("weights/yolo26n.pt")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments with sensible defaults and educational descriptions.
    """
    parser = argparse.ArgumentParser(
        description="Perform Real-Time Object Detection using YOLO26 and OpenCV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input media sources
    parser.add_argument(
        "-v",
        "--video-path",
        type=str,
        default=None,
        help="Path to input video file (e.g. 'data/videos/Chicago_360p.mp4') or '0' for live webcam.",
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=str,
        default=None,
        help="Path to input image file (e.g. 'data/images/dog.jpg').",
    )

    # Model and weights
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=str(DEFAULT_MODEL_WEIGHTS),
        help="Path to YOLO model weights (.pt file).",
    )

    # Output options
    parser.add_argument(
        "-vo",
        "--video-output-path",
        type=str,
        default="results/output.mp4",
        help="Path to save annotated output video.",
    )
    parser.add_argument(
        "-io",
        "--image-output-path",
        type=str,
        default="results/output.jpg",
        help="Path to save annotated output image.",
    )

    # Thresholds
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum detection confidence threshold (0.0 to 1.0) to filter weak predictions.",
    )
    parser.add_argument(
        "-th",
        "--iou-threshold",
        type=float,
        default=0.45,
        help="Non-Maximum Suppression (NMS) Intersection-over-Union (IoU) threshold.",
    )

    # Visualization and runtime limits
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the real-time detection window preview while processing."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before stopping (useful for quick testing on long videos)."
    )

    return parser.parse_args()



class YOLODetector:
    """
    YOLO26 Object Detector using Ultralytics and OpenCV.
    """

    def __init__(self, weights_path: str = str(DEFAULT_MODEL_WEIGHTS)) -> None:
        self.weights_path = Path(weights_path)
        self.model = None
        self.class_names: List[str] = []
        self.colors: List[Tuple[int, int, int]] = []

        self._load_model()

    @property
    def model_tag(self) -> str:
        """Returns a formatted tag string representing the active model architecture."""
        stem = self.weights_path.stem.upper()
        if "YOLO26" in stem:
            return "YOLO26"
        return stem

    def _load_model(self) -> None:
        """Loads the modern YOLO model weights."""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERROR] 'ultralytics' library is required.")
            print("Run: pip install ultralytics")
            sys.exit(1)

        if not self.weights_path.exists():
            print(f"[ERROR] Weights file not found: {self.weights_path}")
            sys.exit(1)

        print(f"[INFO] Loading modern YOLO model from: {self.weights_path}...")
        self.model = YOLO(str(self.weights_path))
        self.class_names = list(self.model.names.values())
        self.colors = get_color_palette(len(self.class_names))
        print(
            f"[INFO] Successfully loaded model with {len(self.class_names)} classes."
        )

    def detect(
        self, frame: np.ndarray, conf_thresh: float, iou_thresh: float
    ) -> Tuple[np.ndarray, int]:
        """
        Runs object detection on a single frame and returns the annotated frame and detection count.

        Args:
            frame: Input BGR image.
            conf_thresh: Minimum confidence threshold.
            iou_thresh: NMS IoU threshold.

        Returns:
            Tuple of (annotated_frame, detection_count).
        """
        annotated_frame = frame.copy()
        detection_count = 0

        # Run inference using Ultralytics YOLO26
        results = self.model.predict(
            source=frame, conf=conf_thresh, iou=iou_thresh, verbose=False
        )

        # Extract detections
        if results and len(results) > 0:
            boxes_data = results[0].boxes
            detection_count = len(boxes_data)

            for box in boxes_data:
                # Coordinates in xyxy format
                coords = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                label = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"id_{class_id}"
                )
                color = self.colors[class_id % len(self.colors)]

                draw_bounding_box(
                    annotated_frame,
                    box=coords,
                    label=label,
                    confidence=confidence,
                    color=color,
                )

        return annotated_frame, detection_count


def process_image(detector: YOLODetector, args: argparse.Namespace) -> None:
    """Handles static image detection and saves the annotated result."""
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"[ERROR] Image file not found: {img_path}")
        sys.exit(1)

    image = cv.imread(str(img_path))
    if image is None:
        print(f"[ERROR] Could not read image at: {img_path}")
        sys.exit(1)

    print(f"[INFO] Processing image: {img_path} ({image.shape[1]}x{image.shape[0]})...")
    start = time.time()
    annotated_img, count = detector.detect(image, args.confidence, args.iou_threshold)
    elapsed = time.time() - start

    print(
        f"[INFO] Found {count} object(s) in {elapsed:.3f} seconds ({1/elapsed:.1f} FPS)."
    )

    # Save output image
    output_path = Path(args.image_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), annotated_img)
    print(f"[INFO] Annotated image saved to: {output_path}")

    if args.show:
        cv.imshow("YOLO Detection Preview (Press any key to close)", annotated_img)
        cv.waitKey(0)
        cv.destroyAllWindows()


def process_video(detector: YOLODetector, args: argparse.Namespace) -> None:
    """Handles real-time video stream detection, live FPS HUD, and video writing."""
    # Support webcam (numeric string '0') or video file path
    if args.video_path.isdigit():
        video_source: Union[int, str] = int(args.video_path)
    else:
        v_path = Path(args.video_path)
        if not v_path.exists():
            print(f"[ERROR] Video file not found: {args.video_path}")
            videos_dir = Path("data/videos")
            if videos_dir.exists():
                available = [f.name for f in videos_dir.glob("*.mp4")]
                if available:
                    print(f"[INFO] Available videos in data/videos/: {', '.join(available)}")
            sys.exit(1)
        video_source = str(v_path)

    cap = cv.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {args.video_path}")
        sys.exit(1)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    is_live = total_frames <= 0
    print(
        f"[INFO] Video opened: {width}x{height} @ {fps:.1f} FPS "
        f"({total_frames if not is_live else 'Live stream'} frames)"
    )

    # Video writer setup
    output_path = Path(args.video_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(output_path), fourcc, fps, (width, height))

    fps_calc = FPSCalculator(smoothing_window=30)
    frame_idx = 0
    model_tag = detector.model_tag

    if args.show:
        print("[INFO] Live preview window enabled. Press 'q' inside the video window to stop early.")
    else:
        print("[INFO] Running in headless mode (processing directly to file).")
        print("[TIP]  To watch the detection live on your screen, add the '--show' flag!")

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            frame_idx += 1

            # 1. Run detection
            annotated_frame, count = detector.detect(
                frame, args.confidence, args.iou_threshold
            )

            # 2. Compute FPS and draw HUD overlay
            current_fps = fps_calc.tick()
            annotated_frame = draw_fps_overlay(annotated_frame, current_fps, model_tag)

            # 3. Write frame to output video
            writer.write(annotated_frame)

            # 4. Progress log every 30 frames
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                progress = (
                    f"[{frame_idx}/{total_frames}]" if not is_live else f"[{frame_idx}]"
                )
                print(
                    f"[INFO] Frame {progress} | Active FPS: {current_fps:.1f} | Detections: {count}"
                )

            # 5. Live preview window
            if args.show:
                cv.imshow(
                    "YOLO Real-Time Detection (Press 'q' to exit)", annotated_frame
                )
                if cv.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Early stopping requested by user.")
                    break

            # 6. Optional frame limit
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[INFO] Reached requested limit of {args.max_frames} frames.")
                break

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv.destroyAllWindows()

    print(
        f"[INFO] Video processing complete! Average speed: {fps_calc.average_fps:.1f} FPS."
    )
    print(f"[INFO] Annotated video saved to: {output_path}")


def main() -> None:
    """Main execution flow."""
    args = parse_arguments()

    # If no media is provided, print beginner-friendly examples
    if args.video_path is None and args.image_path is None:
        print("=" * 60)
        print("  Real-Time Object Detection with YOLO (Pedagogical Edition)")
        print("=" * 60)
        print("\nNo input source provided. Run with one of the following:\n")
        print("  1. Process a video:")
        print("     python yolo.py --video-path data/videos/Chicago_360p.mp4")
        print("\n  2. Process a static image:")
        print("     python yolo.py --image-path data/images/dog.jpg")
        print("\n  3. Run live on your webcam:")
        print("     python yolo.py --video-path 0 --show")
        print("\nFor a list of all options, run: python yolo.py --help\n")
        sys.exit(0)

    # Initialize the detector
    detector = YOLODetector(args.weights)

    # Dispatch to appropriate handler
    if args.image_path:
        process_image(detector, args)
    elif args.video_path:
        process_video(detector, args)


if __name__ == "__main__":
    main()
