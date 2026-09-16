# Pretrained YOLO Model Weights

This directory stores pretrained weights for the YOLO models.

## Available Models

| Model | Weights File | Architecture / Engine | Size | Speed (CPU) | Accuracy (mAP 50-95) | Recommendation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLO26 Nano** *(Default)* | `yolo26n.pt` | Ultralytics YOLO26 (NMS-free) | **~5.5 MB** | **Ultra Fast (~35+ FPS)** | **40.2%** | **Recommended for real-time CPU, webcam, and edge devices** |
| **YOLO26 Small** | `yolo26s.pt` | Ultralytics YOLO26 | ~20.0 MB | Fast (~22 FPS) | 46.8% | Balanced speed & higher accuracy |
| **YOLO26 Medium** | `yolo26m.pt` | Ultralytics YOLO26 | ~42.0 MB | Medium (~14 FPS) | 51.2% | High accuracy for production & GPU |

## Downloading Additional Ultralytics Models

When using the modern Python pipeline (`ultralytics`), any YOLO model is downloaded automatically on first run simply by specifying its name:

```python
from ultralytics import YOLO

# YOLO26 Nano (Fastest, ~5.5 MB)
model = YOLO("weights/yolo26n.pt")

# YOLO26 Small (Balanced, ~20 MB)
model = YOLO("weights/yolo26s.pt")

# YOLO26 Medium (Higher accuracy, ~42 MB)
model = YOLO("weights/yolo26m.pt")
```

