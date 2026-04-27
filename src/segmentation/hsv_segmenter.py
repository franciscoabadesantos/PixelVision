"""
HSV-based Segmentation for Satellite Imagery
Color-based segmentation to detect features like solar panels, roofs, vegetation, etc.
"""

import cv2
import numpy as np
from typing import Tuple, List


class HSVSegmenter:
    """HSV color space-based image segmentation"""

    def __init__(self, hsv_lower=None, hsv_upper=None):
        """
        Initialize segmenter with HSV color ranges.
        
        Args:
            hsv_lower (np.array): Lower HSV bounds [H, S, V]
            hsv_upper (np.array): Upper HSV bounds [H, S, V]
        
        Note: OpenCV HSV ranges are H: 0-179, S: 0-255, V: 0-255
        """
        # Default bounds for red/orange detection (common for solar panels)
        self.hsv_lower = hsv_lower if hsv_lower is not None else np.array([0, 30, 30])
        self.hsv_upper = hsv_upper if hsv_upper is not None else np.array([180, 255, 255])
        self.exclude_ranges = []

    def add_exclude_range(self, hsv_lower, hsv_upper):
        """
        Add color range to exclude (e.g., vegetation, water).
        
        Args:
            hsv_lower (np.array): Lower HSV bounds
            hsv_upper (np.array): Upper HSV bounds
        """
        self.exclude_ranges.append((hsv_lower, hsv_upper))

    def calculate_hsv_stats(self, image):
        """
        Calculate median HSV values for image.
        
        Args:
            image (np.array): BGR image
        
        Returns:
            tuple: (median_h, median_s, median_v)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        median_h = np.median(hsv[:, :, 0])
        median_s = np.median(hsv[:, :, 1])
        median_v = np.median(hsv[:, :, 2])
        return median_h, median_s, median_v

    def adjust_hsv_bounds(self, image, adjustment_factor=0.1):
        """
        Adaptively adjust HSV bounds based on image statistics.
        
        Args:
            image (np.array): BGR image
            adjustment_factor (float): How much to adjust (0-1)
        
        Returns:
            tuple: (adjusted_lower, adjusted_upper)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        median_h, median_s, median_v = self.calculate_hsv_stats(image)

        # Calculate offsets from reference values
        s_offset = median_s - 100.0
        v_offset = median_v - 128.0

        # Adjust bounds
        adjusted_lower = self.hsv_lower.copy().astype(float)
        adjusted_upper = self.hsv_upper.copy().astype(float)

        adjusted_lower[1] = max(self.hsv_lower[1] + s_offset * adjustment_factor, 0)
        adjusted_upper[1] = min(self.hsv_upper[1] + s_offset * adjustment_factor, 255)

        adjusted_lower[2] = max(self.hsv_lower[2] + v_offset * adjustment_factor, 0)
        adjusted_upper[2] = min(self.hsv_upper[2] + v_offset * adjustment_factor, 255)

        return adjusted_lower.astype(np.uint8), adjusted_upper.astype(np.uint8)

    def create_mask(self, image, adaptive=True):
        """
        Create segmentation mask for image.
        
        Args:
            image (np.array): BGR image
            adaptive (bool): Use adaptive HSV bounds
        
        Returns:
            np.array: Binary mask
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Get HSV bounds
        if adaptive:
            lower, upper = self.adjust_hsv_bounds(image)
        else:
            lower, upper = self.hsv_lower, self.hsv_upper

        # Create mask for target colors
        mask = cv2.inRange(hsv, lower, upper)

        # Remove excluded color ranges
        for excl_lower, excl_upper in self.exclude_ranges:
            excl_mask = cv2.inRange(hsv, excl_lower, excl_upper)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl_mask))

        return mask

    def morphological_operations(self, mask, operation="close", kernel_size=5):
        """
        Apply morphological operations to clean mask.
        
        Args:
            mask (np.array): Binary mask
            operation (str): 'close', 'open', 'dilate', 'erode'
            kernel_size (int): Kernel size
        
        Returns:
            np.array: Processed mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        if operation == "close":
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == "open":
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "dilate":
            return cv2.dilate(mask, kernel, iterations=1)
        elif operation == "erode":
            return cv2.erode(mask, kernel, iterations=1)
        else:
            return mask

    def segment(self, image, adaptive=True, apply_morphology=True):
        """
        Perform complete segmentation on image.
        
        Args:
            image (np.array): BGR image
            adaptive (bool): Use adaptive HSV bounds
            apply_morphology (bool): Apply morphological operations
        
        Returns:
            np.array: Binary mask
        """
        mask = self.create_mask(image, adaptive=adaptive)

        if apply_morphology:
            mask = self.morphological_operations(mask, "close")
            mask = self.morphological_operations(mask, "open")

        return mask

    def find_contours(self, mask, min_area=100):
        """
        Find contours in segmentation mask.
        
        Args:
            mask (np.array): Binary mask
            min_area (int): Minimum contour area
        
        Returns:
            list: Contours above minimum area
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                significant_contours.append(contour)

        return significant_contours

    def calculate_contour_properties(self, contour):
        """
        Calculate properties of a contour.
        
        Args:
            contour (np.array): Contour
        
        Returns:
            dict: Contour properties (area, perimeter, centroid, etc.)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        moments = cv2.moments(contour)
        
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0

        # Solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        return {
            'area': area,
            'perimeter': perimeter,
            'centroid': (cx, cy),
            'solidity': solidity,
            'circularity': circularity
        }

    def draw_segmentation(self, image, mask, output_path=None, alpha=0.5):
        """
        Draw segmentation overlay on image.
        
        Args:
            image (np.array): BGR image
            mask (np.array): Binary mask
            output_path (str): Where to save result
            alpha (float): Overlay transparency
        
        Returns:
            np.array: Annotated image
        """
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]

        # Blend with original
        result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        if output_path:
            cv2.imwrite(output_path, result)

        return result
