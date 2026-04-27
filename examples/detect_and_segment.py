"""
Example: Detect objects and segment features in satellite images
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
from detection.yolov8_detector import SatelliteDetector
from segmentation.hsv_segmenter import HSVSegmenter
import numpy as np


def example_detection(image_path):
    """Example: YOLOv8 object detection"""
    print("\n--- YOLOv8 Object Detection ---")
    
    # Initialize detector
    detector = SatelliteDetector(
        model_path="yolov8n.pt",
        confidence=0.5
    )
    
    # Detect objects
    detections = detector.detect(image_path, save_result=True, output_dir="output")
    
    # Print results
    print(f"Found {len(detections['boxes'])} objects:")
    for cls_name, conf in zip(detections['class_names'], detections['confidences']):
        print(f"  - {cls_name}: {conf:.2f}")
    
    return detections


def example_hsv_segmentation(image_path):
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
    os_path = "output"
    os.makedirs(os_path, exist_ok=True)
    result = segmenter.draw_segmentation(image, mask, output_path=os.path.join(os_path, "segmentation_result.jpg"))
    
    return mask, contours


def example_combined(image_path):
    """Example: Combine detection and segmentation"""
    print("\n--- Combined Detection + Segmentation ---")
    
    # Detection
    print("Running object detection...")
    detector = SatelliteDetector(model_path="yolov8n.pt", confidence=0.5)
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


if __name__ == "__main__":
    # Check if sample image exists
    sample_image = "examples/sample_satellite.jpg"
    
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        print("Please provide a satellite image path")
    else:
        try:
            # Run examples
            # detections = example_detection(sample_image)
            # mask, contours = example_hsv_segmentation(sample_image)
            # detections, mask = example_combined(sample_image)
            
            print("\nExamples completed! Check output/ folder for results.")
        except Exception as e:
            print(f"Error: {e}")
