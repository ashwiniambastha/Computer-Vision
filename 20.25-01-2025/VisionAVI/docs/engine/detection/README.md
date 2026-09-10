# YOLOv5 Object Detection Wrapper Documentation (detection_model_1.py)

## Overview
The `YOLOv5` class provides a high-level wrapper around the Ultralytics YOLOv5 implementation for object detection tasks. This documentation details the internal workings and implementation specifics of the wrapper.

## Table of Contents
- [Dependencies](#dependencies)
- [Class Architecture](#class-architecture)
- [Core Methods](#core-methods)
- [Data Structures](#data-structures)
- [Usage Examples](#usage-examples)
- [Testing](#testing)

## Dependencies
```python
from ultralytics import YOLO  # Base YOLOv5 implementation
import cv2                    # OpenCV for image/video processing
import numpy as np           # Numerical computations
from typing import List, Dict, Union, Tuple  # Type hints
```

## Class Architecture

### Class: YOLOv5
A wrapper class that encapsulates YOLOv5 functionality with standardized inputs/outputs.

#### Instance Variables
- `self.model`: YOLO
  - Holds the loaded YOLOv5 model instance
  - Initialized with model weights file path

## Core Methods

### 1. Constructor
```python
def __init__(self, model_path: str):
    self.model = YOLO(model_path)
```
- **Purpose**: Initializes YOLOv5 model instance
- **Parameters**: 
  - `model_path`: Path to YOLOv5 weights file
- **Returns**: None

### 2. Process Detections
```python
def process_detections(self, results) -> List[Dict]:
```
- **Purpose**: Standardizes YOLOv5 detection results
- **Input Processing**:
  - Iterates through detection results
  - Extracts bounding boxes, confidence scores, and class information
- **Output Format**:
  ```python
  {
      "class": str,          # Class name
      "confidence": float,   # Detection confidence (0-1)
      "x_min": float,       # Left boundary
      "y_min": float,       # Top boundary
      "x_max": float,       # Right boundary
      "y_max": float,       # Bottom boundary
      "class_id": int       # Numeric class identifier
  }
  ```

### 3. Image Detection
```python
def detect_image(self, image_path: str) -> Tuple[List[Dict], Dict]:
```
- **Purpose**: Performs detection on single images
- **Process Flow**:
  1. Loads image from path
  2. Runs YOLOv5 prediction
  3. Processes detections
  4. Extracts metadata
- **Returns**:
  - Detections list
  - Metadata dictionary with resolution

### 4. Frame Detection
```python
def detect_frame(self, frame: np.ndarray) -> List[Dict]:
```
- **Purpose**: Real-time detection on video frames
- **Input**: NumPy array representing image frame
- **Process**:
  1. Performs streaming prediction
  2. Returns processed detections
- **Performance**: Optimized for real-time processing

### 5. Video Detection
```python
def detect_video(self, video_path: str) -> List[Dict]:
```
- **Purpose**: Batch detection on video files
- **Process**:
  - Loads video file
  - Performs detection on all frames
  - Aggregates results

### 6. Utility Methods

#### Get Bounding Box Coordinates
```python
def get_bbox_coordinates(self, detection: Dict) -> List[float]:
```
- **Purpose**: Extracts coordinates for visualization
- **Returns**: [x_min, y_min, x_max, y_max]

#### Save Results
```python
def save_results(self, results, output_path: str):
```
- **Purpose**: Persists detection results
- **Format**: Native YOLOv5 format

## Data Structures

### Detection Dictionary
```python
{
    "class": str,        # Object class name
    "confidence": float, # Detection confidence
    "x_min": float,      # Bounding box coordinates
    "y_min": float,
    "x_max": float,
    "y_max": float,
    "class_id": int      # Class identifier
}
```

### Metadata Dictionary
```python
{
    "resolution": str    # Format: "heightxwidth"
}
```

## Usage Examples

### Basic Image Detection
```python
detector = YOLOv5("yolov5s.pt")
detections, metadata = detector.detect_image("image.jpg")
```

### Real-time Video Processing
```python
detector = YOLOv5("yolov5s.pt")
frame = cv2.imread("frame.jpg")
detections = detector.detect_frame(frame)
```

## Testing

### Test Cases
1. Model Information Retrieval
2. Image Detection Pipeline
3. Video Processing
4. Webcam Integration
5. Result Persistence

### Running Tests
```python
if __name__ == "__main__":
    model = YOLOv5("yolov5s.pt")
    # Run specific test cases
```

## Performance Considerations

### Memory Usage
- Loads model weights once at initialization
- Streams video frames when possible
- Processes detections in batches

### Speed Optimizations
- Uses CUDA if available
- Streaming mode for real-time processing
- Batch processing for videos

## Error Handling
- Validates input paths
- Checks frame integrity
- Ensures model compatibility

## Limitations
- Requires compatible YOLOv5 weights
- Memory usage scales with video length
- Real-time performance depends on hardware