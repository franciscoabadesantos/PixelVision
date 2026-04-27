"""PixelVision - Computer vision for satellite images"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .detection.yolov8_detector import SatelliteDetector
from .segmentation.hsv_segmenter import HSVSegmenter

__all__ = ["SatelliteDetector", "HSVSegmenter"]
