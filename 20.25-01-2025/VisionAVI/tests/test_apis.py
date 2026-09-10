# Python module: test_apis.py
import pytest
from fastapi.testclient import TestClient


import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.database_api import app
import os
import cv2
import numpy as np

client = TestClient(app)

# Test data setup
TEST_DATA_DIR = Path("tests/data")
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_test_image():
    """Create a dummy test image for testing."""
    img_path = TEST_DATA_DIR / "test.jpg"
    if not img_path.exists():
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.imwrite(str(img_path), img)
    return img_path


def create_test_video():
    """Create a dummy test video for testing."""
    video_path = TEST_DATA_DIR / "test.mp4"
    if not video_path.exists():
        out = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (100, 100)
        )
        for _ in range(30):  # 1 second video
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.rectangle(frame, (30, 30), (70, 70), (255, 255, 255), -1)
            out.write(frame)
        out.release()
    return video_path


@pytest.fixture(scope="module")
def test_files():
    """Create test files and clean up after tests."""
    img_path = create_test_image()
    video_path = create_test_video()
    yield {"image": img_path, "video": video_path}


def test_detect_image(test_files):
    """Test image detection endpoint."""
    img_path = test_files["image"]
    with open(img_path, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        response = client.post("/detect/image", files=files)

    assert response.status_code == 200
    json_response = response.json()
    assert "scene_id" in json_response
    assert "detections" in json_response
    assert isinstance(json_response["detections"], list)


def test_detect_video(test_files):
    """Test video detection endpoint."""
    video_path = test_files["video"]
    with open(video_path, "rb") as f:
        files = {"file": ("test.mp4", f, "video/mp4")}
        response = client.post("/detect/video", files=files)

    assert response.status_code == 200
    json_response = response.json()
    assert "scene_id" in json_response
    assert "total_detections" in json_response
    assert isinstance(json_response["total_detections"], int)


def test_webcam_detection_invalid_duration():
    """Test webcam detection with invalid duration."""
    response = client.post("/detect/webcam?duration=-1")
    assert response.status_code == 422  # FastAPI validation error
    assert "greater than" in response.json()["detail"][0]["msg"]


def test_webcam_detection():
    """Test webcam detection endpoint."""
    response = client.post("/detect/webcam?duration=1")
    assert response.status_code == 200
    json_response = response.json()
    assert "scene_id" in json_response
    assert "message" in json_response
    assert "completed" in json_response["message"]


def test_rtsp_detection():
    """Test RTSP stream detection endpoint."""
    test_url = "rtsp://example.com/test"
    response = client.post(f"/detect/rtsp?rtsp_url={test_url}&duration=1")
    assert response.status_code == 200
    json_response = response.json()
    assert "scene_id" in json_response
    assert "message" in json_response
    assert "completed" in json_response["message"]


def test_invalid_image_upload():
    """Test uploading invalid image file."""
    files = {"file": ("test.txt", b"invalid content", "text/plain")}
    response = client.post("/detect/image", files=files)
    assert response.status_code == 400  # Bad request
    assert "Invalid file type" in response.json()["detail"]


def test_missing_file():
    """Test API response when no file is uploaded."""
    response = client.post("/detect/image")
    assert response.status_code == 422  # Validation error


def test_invalid_rtsp_url():
    """Test RTSP endpoint with invalid URL."""
    response = client.post("/detect/rtsp?rtsp_url=invalid_url&duration=1")
    assert response.status_code == 422  # Invalid URL format
    assert "regex" in response.json()["detail"][0]["msg"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
