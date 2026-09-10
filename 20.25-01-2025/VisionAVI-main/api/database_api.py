"""
Object Detection API using FastAPI and YOLOv5

This module provides a REST API for object detection using YOLOv5 model.
It supports multiple input sources including images, videos, webcam feeds,
and RTSP streams. All detections are stored in a SQLite database for
further analysis.

Endpoints:
    POST /detect/image:
        Process uploaded image files for object detection
    POST /detect/video:
        Process uploaded video files for object detection
    POST /detect/webcam:
        Capture and process webcam feed for specified duration
    POST /detect/rtsp:
        Process RTSP stream for specified duration

Dependencies:
    - FastAPI
    - OpenCV (cv2)
    - YOLOv5 (custom implementation)
    - SQLite database

Environment Setup:
    1. Install required packages:
        pip install fastapi uvicorn python-multipart opencv-python
    2. Ensure YOLOv5 model file (yolov5s.pt) is present
    3. Initialize database using schema.sql

Usage Examples:
    # Start the API server
    $ uvicorn api.database_api:app --reload

    # Test endpoints using curl:
    $ curl -X POST "http://localhost:8000/detect/image" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test.jpg"
    $ curl -X POST "http://localhost:8000/detect/video" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test.mp4"
    $ curl -X POST "http://localhost:8000/detect/webcam?duration=10"
    $ curl -X POST "http://localhost:8000/detect/rtsp?rtsp_url=rtsp://example.com/stream&duration=30"

Configuration:
    - Model Path: Update YOLOv5 model path in initialization
    - Database: Update database path in SceneDatabase initialization
    - Upload Directory: Files are saved to 'uploads/' directory

Author: Anantha Krishna B
Version: 1.0.0
"""

# Import the required libraries
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
import numpy as np
from datetime import datetime
import uuid
import os
from pathlib import Path

# Import your custom modules
from core.backend.detection.detection_model_1 import YOLOv5
from database.schema import SceneDatabase

app = FastAPI()

# Initialize YOLO model
model = YOLOv5("yolov5s.pt")  # Update with your model path
db = SceneDatabase("scenes.db")

ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/mpeg"]


# Helper function to save uploaded file
async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to the server.

    Args:
        upload_file (UploadFile): The file uploaded by the user.

    Returns:
        str: The file path where the uploaded file is saved.
    """
    file_location = f"uploads/{upload_file.filename}"
    os.makedirs("uploads", exist_ok=True)

    with open(file_location, "wb+") as file_object:
        file_object.write(await upload_file.read())
    return file_location


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Process image with content type validation."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Must be one of: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    try:
        file_path = await save_upload_file(file)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        detections, metadata = model.detect_image(file_path)

        # Debug print
        print("Received detections:", detections[0] if detections else "No detections")

        scene_metadata = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": "image_upload",
            "media_path": file_path,
            "resolution": metadata["resolution"],
        }

        scene_id = db.add_scene(scene_metadata)
        db.add_detections(scene_id, detections)  # No need to format detections anymore

        # Draw detections on image for visualization
        img = cv2.imread(file_path)
        for det in detections:
            x1, y1, x2, y2 = (
                int(det["x_min"]),
                int(det["y_min"]),
                int(det["x_max"]),
                int(det["y_max"]),
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Save annotated image
        output_path = f"uploads/annotated_{os.path.basename(file_path)}"
        cv2.imwrite(output_path, img)

        return JSONResponse(
            {
                "scene_id": scene_id,
                "detections": detections,
                "annotated_image": output_path,
            }
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in detect_image: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    file_path = await save_upload_file(file)

    # Use the new detect_video method
    detections = model.detect_video(file_path)

    scene_metadata = {
        "timestamp": datetime.now().isoformat(),
        "camera_id": "video_upload",
        "media_path": file_path,
    }

    scene_id = db.add_scene(scene_metadata)
    db.add_detections(scene_id, detections)

    return JSONResponse({"scene_id": scene_id, "total_detections": len(detections)})


@app.post("/detect/webcam")
async def start_webcam_detection(
    duration: int = Query(..., gt=0, description="Duration in seconds")
):
    """Start webcam detection with duration validation."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open webcam")

        scene_metadata = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": "webcam",
            "media_path": f"webcam_stream_{uuid.uuid4()}",
        }

        scene_id = db.add_scene(scene_metadata)

        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < duration:
            success, frame = cap.read()
            if not success:
                break

            # Use the new detect_frame method
            frame_detections = model.detect_frame(frame)
            if frame_detections:
                db.add_detections(scene_id, frame_detections)
                for det in frame_detections:
                    x1, y1, x2, y2 = (
                        det["x_min"],
                        det["y_min"],
                        det["x_max"],
                        det["y_max"],
                    )
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                    label = f"{det['class']}: {det['confidence']:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        cap.release()

        return JSONResponse(
            {
                "scene_id": scene_id,
                "message": f"Webcam detection completed for {duration} seconds",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/rtsp")
async def start_rtsp_detection(
    rtsp_url: str = Query(..., regex="^rtsp://.*"), duration: int = Query(..., gt=0)
):
    """Process RTSP stream with URL validation."""
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400, detail="Failed to connect to RTSP stream"
            )

        scene_metadata = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": "rtsp_stream",
            "media_path": rtsp_url,
        }

        scene_id = db.add_scene(scene_metadata)

        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < duration:
            success, frame = cap.read()
            if not success:
                break

            # Use the model's detect_frame method instead of direct predict
            frame_detections = model.detect_frame(frame)

            if frame_detections:
                db.add_detections(scene_id, frame_detections)
                # Optionally add visualization like in webcam endpoint
                for det in frame_detections:
                    x1, y1, x2, y2 = (
                        det["x_min"],
                        det["y_min"],
                        det["x_max"],
                        det["y_max"],
                    )
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )

        cap.release()

        return JSONResponse(
            {
                "scene_id": scene_id,
                "message": f"RTSP stream detection completed for {duration} seconds",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
