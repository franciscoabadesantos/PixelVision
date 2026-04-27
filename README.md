# PixelVision

Computer vision-based object detection and feature segmentation for satellite imagery. Detect and analyze features like solar panels, buildings, roads, and vegetation using YOLOv8 and HSV-based color segmentation.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)

## Features

- **YOLOv8 Object Detection** - Real-time detection of objects in satellite images
  - Supports multiple YOLO variants (nano, small, medium, large, xlarge)
  - Pre-trained on COCO dataset or custom-trained models
  - GPU acceleration support (CUDA)

- **HSV Color-Based Segmentation** - Adaptive color space segmentation
  - Detect features by color (solar panels, roofs, vegetation, water)
  - Adaptive HSV bounds based on image statistics
  - Morphological operations for noise filtering

- **Contour Analysis** - Extract geometric properties
  - Area, perimeter, solidity, circularity calculations
  - Hierarchical contour detection

- **Batch Processing** - Process multiple images efficiently
  - Vectorized operations
  - Results export (JSON, CSV)

- **Combined Analysis** - Synergistic detection + segmentation
  - Multi-model inference
  - Complementary feature extraction

## Installation

### Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PixelVision.git
cd PixelVision
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Object Detection with YOLOv8

```python
from src.detection.yolov8_detector import SatelliteDetector

# Initialize detector
detector = SatelliteDetector(model_path="yolov8n.pt", confidence=0.5)

# Detect objects in image
detections = detector.detect("image.jpg", save_result=True, output_dir="output")

# Print results
for cls_name, conf in zip(detections['class_names'], detections['confidences']):
    print(f"{cls_name}: {conf:.2f}")
```

### Color-Based Segmentation

```python
import numpy as np
import cv2
from src.segmentation.hsv_segmenter import HSVSegmenter

# Initialize segmenter for solar panels (red/orange)
segmenter = HSVSegmenter(
    hsv_lower=np.array([0, 30, 30]),
    hsv_upper=np.array([20, 255, 255])
)

# Exclude vegetation (green)
segmenter.add_exclude_range(
    np.array([35, 40, 40]),
    np.array([85, 255, 255])
)

# Load and segment image
image = cv2.imread("image.jpg")
mask = segmenter.segment(image, adaptive=True)

# Find features
contours = segmenter.find_contours(mask, min_area=100)
print(f"Found {len(contours)} features")
```

### Batch Processing

```python
detector = SatelliteDetector(model_path="yolov8n.pt")

# Process multiple images
results = detector.detect_batch(
    image_dir="images/",
    save_results=True,
    output_dir="output/"
)
```

## Available Models

### YOLOv8 Pretrained Models

| Model | Size | Speed (GPU) | mAP50 | Parameters |
|-------|------|-----------|-------|-----------|
| YOLOv8n | 640 | 0.6ms | 37.3 | 3.2M |
| YOLOv8s | 640 | 1.2ms | 44.9 | 11.2M |
| YOLOv8m | 640 | 2.3ms | 50.2 | 25.9M |
| YOLOv8l | 640 | 2.7ms | 52.9 | 43.7M |
| YOLOv8x | 640 | 3.5ms | 53.9 | 68.2M |

### Segmentation Models Available

Pre-trained models in `models/` directory:

- **yolov8n-seg.pt** - YOLOv8 Nano instance segmentation
- **road_segmentation_model.onnx** - Road network segmentation
- **PV-Segmentation-deeplabv3.onnx** - Solar panel (PV) segmentation using DeepLabv3

## Architecture

### Core Modules

- **`src/detection/yolov8_detector.py`** - YOLOv8 detection wrapper
- **`src/segmentation/hsv_segmenter.py`** - HSV-based color segmentation

### Supported Output Formats

- Bounding boxes with confidence scores
- Segmentation masks (binary and colored)
- Contour coordinates and properties
- JSON/CSV export for downstream analysis

## Usage Examples

### Example 1: Detect Solar Panels

```python
from src.detection.yolov8_detector import SatelliteDetector
import numpy as np

detector = SatelliteDetector("yolov8m.pt", confidence=0.6)
detections = detector.detect("satellite_image.jpg")

# Filter for specific class
solar_panels = detector.filter_detections(detections, "solar_panel")
print(f"Found {len(solar_panels['boxes'])} solar panels")
```

### Example 2: Segment Red/Orange Features

```python
import cv2
import numpy as np
from src.segmentation.hsv_segmenter import HSVSegmenter

segmenter = HSVSegmenter(
    hsv_lower=np.array([140, 30, 30]),   # Red lower
    hsv_upper=np.array([180, 255, 255])  # Red upper
)

# Add second range for orange
segmenter.add_exclude_range(
    np.array([0, 30, 30]),
    np.array([20, 255, 255])
)

image = cv2.imread("rooftop.jpg")
mask = segmenter.segment(image, adaptive=True)

# Find contours
contours = segmenter.find_contours(mask, min_area=50)
print(f"Detected {len(contours)} red/orange regions")

# Analyze properties
for i, contour in enumerate(contours):
    props = segmenter.calculate_contour_properties(contour)
    print(f"Region {i}: Area={props['area']:.0f}, Solidity={props['solidity']:.2f}")
```

### Example 3: Combined Detection + Segmentation

```python
from src.detection.yolov8_detector import SatelliteDetector
from src.segmentation.hsv_segmenter import HSVSegmenter
import cv2
import numpy as np

# Detection
detector = SatelliteDetector("yolov8l.pt", confidence=0.5)
detections = detector.detect("satellite.jpg")

# Segmentation
segmenter = HSVSegmenter()
image = cv2.imread("satellite.jpg")
mask = segmenter.segment(image)

# Combined analysis
print(f"Objects detected: {len(detections['boxes'])}")
print(f"Segmentation coverage: {(np.count_nonzero(mask)/mask.size)*100:.1f}%")
```

### Example 4: Batch Processing with Export

```python
from src.detection.yolov8_detector import SatelliteDetector
import json
import os

detector = SatelliteDetector("yolov8m.pt", confidence=0.5)

results = {}
for filename in os.listdir("satellite_images/"):
    if filename.endswith(".jpg"):
        detections = detector.detect(f"satellite_images/{filename}")
        results[filename] = {
            'num_objects': len(detections['boxes']),
            'classes': detections['class_names'],
            'confidences': [float(c) for c in detections['confidences']]
        }

# Export results
with open("detection_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## HSV Color Ranges Reference

Common HSV ranges for satellite imagery analysis (OpenCV format where H: 0-179):

### Red/Orange (Solar Panels, Rooftops)
```python
lower = np.array([0, 30, 30])
upper = np.array([20, 255, 255])
# or
lower = np.array([140, 30, 30])
upper = np.array([180, 255, 255])
```

### Green (Vegetation)
```python
lower = np.array([35, 40, 40])
upper = np.array([85, 255, 255])
```

### Blue (Water)
```python
lower = np.array([100, 50, 50])
upper = np.array([140, 255, 255])
```

### Yellow (Urban Features)
```python
lower = np.array([15, 100, 100])
upper = np.array([35, 255, 255])
```

## Configuring Custom Models

### Using Your Own YOLO Weights

```python
# Train your own model
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="dataset.yaml", epochs=100, imgsz=640)

# Use custom model
detector = SatelliteDetector(model_path="runs/detect/train/weights/best.pt")
```

### Segment with Custom HSV Bounds

```python
segmenter = HSVSegmenter(
    hsv_lower=np.array([h_min, s_min, v_min]),
    hsv_upper=np.array([h_max, s_max, v_max])
)

# Interactively find bounds
import cv2
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
# ... etc for all 6 values
```

## Performance Tips

1. **GPU Acceleration** - Use CUDA-enabled GPU for 5-10x speedup
2. **Batch Size** - Process multiple images in one inference pass
3. **Model Size** - Use nano/small models for faster inference, large for accuracy
4. **Input Resolution** - Lower resolution = faster but less accurate
5. **Caching** - Cache model in GPU memory between multiple inferences

## Output Examples

### Detection Output
```json
{
  "boxes": [[100, 200, 300, 400], [150, 250, 350, 450]],
  "classes": [0, 1],
  "confidences": [0.92, 0.87],
  "class_names": ["solar_panel", "building"]
}
```

### Segmentation Properties
```python
{
  'area': 2543,
  'perimeter': 182.4,
  'centroid': (245, 312),
  'solidity': 0.89,
  'circularity': 0.76
}
```

## Troubleshooting

### Issue: Out of Memory (OOM) Error

**Solution:** Use smaller model or reduce batch size
```python
detector = SatelliteDetector("yolov8n.pt")  # Use nano instead of large
```

### Issue: Slow Inference

**Solution:** Enable GPU acceleration
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

### Issue: Poor Segmentation Results

**Solution:** Adjust HSV bounds adaptively
```python
mask = segmenter.segment(image, adaptive=True)  # Enable adaptive bounds
```

### Issue: Model Download Fails

**Solution:** Download models manually
```bash
pip install --upgrade ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

## Related Projects

- **EarthCapture** - Automated image extraction from Google Earth Pro
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

## Project Structure

```
SatelliteImagery-Detector/
├── src/
│   ├── detection/
│   │   └── yolov8_detector.py       # YOLOv8 detection wrapper
│   └── segmentation/
│       └── hsv_segmenter.py         # HSV segmentation
├── models/
│   ├── pretrained/
│   │   ├── yolov8n.pt
│   │   ├── yolov8m.pt
│   │   └── ...
│   └── trained/
│       └── model.pt
├── examples/
│   └── detect_and_segment.py        # Usage examples
├── docs/
│   ├── MODEL_TRAINING.md            # How to train custom models
│   └── INFERENCE.md                 # Inference guide
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Citation

```bibtex
@software{pixelvision,
  title={PixelVision - Satellite Imagery Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/PixelVision}
}

@article{redmon2016you,
  title={You Only Look Once: Unified, Real-Time Object Detection},
  author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  year={2016}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and research purposes only. Ensure you have appropriate permissions before analyzing satellite imagery. Respect copyright and privacy laws in your jurisdiction.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/SatelliteImagery-Detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SatelliteImagery-Detector/discussions)
- **Email**: your.email@example.com

---

**Version**: 1.0.0  
**Last Updated**: 2024
