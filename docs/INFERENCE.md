# Inference

## CLI Example

Run the bundled example:

```bash
python examples/detect_and_segment.py path/to/image.jpg --mode all --model yolov8n.pt
```

Modes:

- `detect`: YOLOv8 detection only
- `segment`: HSV segmentation only
- `all`: detection, segmentation, and a combined summary

## Python Example

```python
from src.detection.yolov8_detector import SatelliteDetector

detector = SatelliteDetector("yolov8n.pt", confidence=0.5)
results = detector.detect("image.jpg", save_result=True, output_dir="output")
```

Ultralytics model names such as `yolov8n.pt` can be downloaded automatically on first use.
