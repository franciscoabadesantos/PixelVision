"""CLI example for running PixelVision on a satellite image."""

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")

sys.path.insert(0, SRC_ROOT)

import cv2
from detection.yolov8_detector import SatelliteDetector
from segmentation.hsv_segmenter import HSVSegmenter
import numpy as np


def example_detection(image_path, model_path, output_dir):
    """Example: YOLOv8 object detection"""
    print("\n--- YOLOv8 Object Detection ---")

    detector = SatelliteDetector(model_path=model_path, confidence=0.5)
    detections = detector.detect(image_path, save_result=True, output_dir=output_dir)
    
    # Print results
    print(f"Found {len(detections['boxes'])} objects:")
    for cls_name, conf in zip(detections['class_names'], detections['confidences']):
        print(f"  - {cls_name}: {conf:.2f}")
    
    if "output_path" in detections:
        print(f"Annotated image saved to: {detections['output_path']}")

    return detections


def example_hsv_segmentation(image_path, output_dir):
    """Example: HSV-based color segmentation"""
    print("\n--- HSV Color Segmentation ---")
    
    # Initialize segmenter for solar panels (red/orange)
    segmenter = HSVSegmenter(
        hsv_lower=np.array([0, 30, 30]),
        hsv_upper=np.array([180, 255, 255])
    )
    
    # Exclude vegetation (green) and water (blue)
    segmenter.add_exclude_range(
        np.array([35, 40, 40]),
        np.array([85, 255, 255])
    )
    segmenter.add_exclude_range(
        np.array([100, 50, 50]),
        np.array([140, 255, 255])
    )
    
    # Load image
    image = cv2.imread(image_path)
    
    # Segment
    mask = segmenter.segment(image, adaptive=True)
    
    # Find contours
    contours = segmenter.find_contours(mask, min_area=100)
    print(f"Found {len(contours)} segments")
    
    # Calculate properties
    for i, contour in enumerate(contours[:5]):  # Show first 5
        props = segmenter.calculate_contour_properties(contour)
        print(f"  Segment {i+1}:")
        print(f"    Area: {props['area']:.0f} pixels")
        print(f"    Solidity: {props['solidity']:.2f}")
        print(f"    Circularity: {props['circularity']:.2f}")
    
    # Draw result
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "segmentation_result.jpg")
    segmenter.draw_segmentation(image, mask, output_path=result_path)
    print(f"Segmentation overlay saved to: {result_path}")
    
    return mask, contours


def example_combined(image_path, model_path):
    """Example: Combine detection and segmentation"""
    print("\n--- Combined Detection + Segmentation ---")
    
    # Detection
    print("Running object detection...")
    detector = SatelliteDetector(model_path=model_path, confidence=0.5)
    detections = detector.detect(image_path)
    
    # Segmentation
    print("Running segmentation...")
    segmenter = HSVSegmenter(
        hsv_lower=np.array([0, 30, 30]),
        hsv_upper=np.array([180, 255, 255])
    )
    
    image = cv2.imread(image_path)
    mask = segmenter.segment(image, adaptive=True)
    
    # Calculate statistics
    detected_objects = len(detections['boxes'])
    segmented_pixels = np.count_nonzero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage = (segmented_pixels / total_pixels) * 100
    
    print(f"Objects detected: {detected_objects}")
    print(f"Segmentation coverage: {coverage:.2f}%")
    
    return detections, mask


def build_parser():
    """Build the example CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run object detection and HSV segmentation on a satellite image."
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--mode",
        choices=["detect", "segment", "all"],
        default="all",
        help="Which example workflow to run.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model weights path. Ultralytics can auto-download standard names.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated artifacts.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    try:
        if args.mode in {"detect", "all"}:
            example_detection(args.image, args.model, args.output_dir)

        if args.mode in {"segment", "all"}:
            example_hsv_segmentation(args.image, args.output_dir)

        if args.mode == "all":
            example_combined(args.image, args.model)

        print(f"\nExamples completed. Check {Path(args.output_dir)} for results.")
    except Exception as exc:
        print(f"Error: {exc}")
