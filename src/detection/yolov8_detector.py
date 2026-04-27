"""
YOLOv8 Object Detection on Satellite Imagery
Detects objects (solar panels, vehicles, buildings, etc.) in satellite images
"""

import cv2
import numpy as np
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("YOLOv8 requires ultralytics: pip install ultralytics")


class SatelliteDetector:
    """YOLOv8-based object detector for satellite imagery"""

    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        """
        Initialize detector with YOLOv8 model.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            confidence (float): Detection confidence threshold (0-1)
        """
        self.confidence = confidence
        self.model = YOLO(model_path)
        self.device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    def detect(self, image_path, save_result=False, output_dir=None):
        """
        Detect objects in a satellite image.
        
        Args:
            image_path (str): Path to image file
            save_result (bool): Whether to save annotated image
            output_dir (str): Directory to save results
        
        Returns:
            dict: Detection results with boxes, classes, and confidences
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run inference
        results = self.model(image, conf=self.confidence, device=self.device)

        # Process results
        detections = {
            'boxes': [],
            'classes': [],
            'confidences': [],
            'class_names': []
        }

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls]

                detections['boxes'].append([x1, y1, x2, y2])
                detections['classes'].append(cls)
                detections['confidences'].append(conf)
                detections['class_names'].append(cls_name)

        # Save annotated image if requested
        if save_result and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            annotated_image = results[0].plot()
            output_path = os.path.join(
                output_dir,
                f"detected_{Path(image_path).name}"
            )
            cv2.imwrite(output_path, annotated_image)
            detections['output_path'] = output_path

        return detections

    def detect_batch(self, image_dir, save_results=False, output_dir=None):
        """
        Detect objects in multiple images.
        
        Args:
            image_dir (str): Directory containing images
            save_results (bool): Whether to save annotated images
            output_dir (str): Directory to save results
        
        Returns:
            dict: Detection results for each image
        """
        results = {}
        image_files = list(Path(image_dir).glob("*.jpg")) + \
                     list(Path(image_dir).glob("*.png"))

        for image_path in image_files:
            print(f"Processing: {image_path.name}")
            detections = self.detect(
                str(image_path),
                save_result=save_results,
                output_dir=output_dir
            )
            results[image_path.name] = detections

        return results

    def filter_detections(self, detections, class_name):
        """
        Filter detections by class name.
        
        Args:
            detections (dict): Detection results
            class_name (str): Class to filter by
        
        Returns:
            dict: Filtered detections
        """
        filtered = {
            'boxes': [],
            'classes': [],
            'confidences': [],
            'class_names': []
        }

        for i, cn in enumerate(detections['class_names']):
            if cn == class_name:
                filtered['boxes'].append(detections['boxes'][i])
                filtered['classes'].append(detections['classes'][i])
                filtered['confidences'].append(detections['confidences'][i])
                filtered['class_names'].append(cn)

        return filtered

    def draw_detections(self, image_path, detections, output_path=None):
        """
        Draw detection boxes on image.
        
        Args:
            image_path (str): Path to image
            detections (dict): Detection results
            output_path (str): Where to save annotated image
        """
        image = cv2.imread(image_path)
        
        for box, conf, cls_name in zip(
            detections['boxes'],
            detections['confidences'],
            detections['class_names']
        ):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls_name} ({conf:.2f})"
            cv2.putText(
                image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Saved annotated image: {output_path}")
        
        return image
