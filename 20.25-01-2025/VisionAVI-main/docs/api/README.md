# Object Detection API Documentation (database_api.py)

## Overview
The database_api.py module provides a RESTful API for object detection using YOLOv5, supporting multiple input sources and storing results in a SQLite database.

## Table of Contents
1. [API Endpoints](#api-endpoints)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Data Structures](#data-structures)
6. [Error Handling](#error-handling)

## API Endpoints

### 1. Image Detection
```http
POST /detect/image
Content-Type: multipart/form-data
```
- **Purpose**: Process uploaded images for object detection
- **Parameters**:
  - `file`: Image file (jpeg, png, jpg)
- **Response**:
```json
{
    "scene_id": int,
    "detections": List[Detection],
    "annotated_image": string
}
```

### 2. Video Detection
```http
POST /detect/video
Content-Type: multipart/form-data
```
- **Parameters**:
  - `file`: Video file (mp4, avi, mpeg)
- **Response**:
```json
{
    "scene_id": int,
    "total_detections": int
}
```

### 3. Webcam Detection
```http
POST /detect/webcam
```
- **Parameters**:
  - `duration`: int (seconds)
- **Response**:
```json
{
    "scene_id": int,
    "message": string
}
```

### 4. RTSP Stream Detection
```http
POST /detect/rtsp
```
- **Parameters**:
  - `rtsp_url`: string (must start with "rtsp://")
  - `duration`: int (seconds)
- **Response**:
```json
{
    "scene_id": int,
    "message": string
}
```

## Installation

### Dependencies
```bash
pip install fastapi uvicorn python-multipart opencv-python
```

### Required Files
- YOLOv5 model file (`yolov5s.pt`)
- Database schema (`schema.sql`)

## Configuration

### Model Settings
```python
model = YOLOv5("yolov5s.pt")  # Update path as needed
db = SceneDatabase("scenes.db")
```

### File Type Restrictions
```python
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/mpeg"]
```

## Usage Examples

### Starting the Server
```bash
uvicorn api.database_api:app --reload --port 8000
```

### API Requests

#### Image Detection
```bash
curl -X POST "http://localhost:8000/detect/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

#### Video Detection
```bash
curl -X POST "http://localhost:8000/detect/video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.mp4"
```

## Data Structures

### Detection Format
```python
Detection = {
    "class": str,          # Object class name
    "confidence": float,   # Detection confidence (0-1)
    "x_min": float,       # Bounding box coordinates
    "y_min": float,
    "x_max": float,
    "y_max": float
}
```

### Scene Metadata
```python
SceneMetadata = {
    "timestamp": str,      # ISO format datetime
    "camera_id": str,      # Source identifier
    "media_path": str,     # File/stream path
    "resolution": str      # Optional resolution
}
```

## Error Handling

### HTTP Status Codes
- 400: Bad Request (invalid file type, invalid parameters)
- 500: Internal Server Error (file saving, processing errors)

### Validation
```python
if file.content_type not in ALLOWED_IMAGE_TYPES:
    raise HTTPException(status_code=400, detail="Invalid file type")
```

## Performance Considerations

### File Handling
- Asynchronous file uploads
- Stream processing for videos
- Temporary file cleanup

### Database Operations
- Batch detection storage
- Thread-safe database connections
- Efficient query patterns

## Development
Author: Anantha Krishna B
Version: 1.0.0