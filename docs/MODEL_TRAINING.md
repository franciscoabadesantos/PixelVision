# Model Training

`PixelVision` does not implement a custom training pipeline. It expects trained weights to come from Ultralytics or another external workflow.

## Typical Flow

1. Train a model with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640)
```

2. Copy the resulting weights into `models/trained/`.
3. Pass that path to `SatelliteDetector`.

## Example

```python
from src.detection.yolov8_detector import SatelliteDetector

detector = SatelliteDetector("models/trained/best.pt", confidence=0.5)
```
