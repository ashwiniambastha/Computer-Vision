# Documentation for `test_apis.py`

This document provides a detailed explanation of the `test_apis.py` module, which is designed to test various endpoints of a FastAPI application. The module uses `pytest` as the testing framework and `TestClient` from FastAPI for making HTTP requests to the API.

## Overview

The `test_apis.py` module includes tests for the following API endpoints:

1. **Image Detection Endpoint** (`/detect/image`)
2. **Video Detection Endpoint** (`/detect/video`)
3. **Webcam Detection Endpoint** (`/detect/webcam`)
4. **RTSP Stream Detection Endpoint** (`/detect/rtsp`)
5. **Invalid File Uploads**
6. **Missing File Uploads**
7. **Invalid RTSP URL Handling**

The tests ensure that the API behaves as expected under various scenarios, including valid and invalid inputs.

---

## Prerequisites

Before running the tests, ensure the following:

1. **Python Environment**: Python 3.8 or higher is installed.
2. **Dependencies**: Install the required dependencies using `pip install -r requirements.txt`. The module requires:
   - `pytest`
   - `fastapi`
   - `opencv-python`
   - `numpy`
3. **Test Data Directory**: The script automatically creates a `tests/data` directory for storing test files (images and videos).

---

## Test Setup

### Test Data Creation

The module includes helper functions to create dummy test files:

- **`create_test_image()`**: Generates a 100x100 black image with a white rectangle in the center and saves it as `test.jpg`.
- **`create_test_video()`**: Generates a 1-second video (30 frames) with a similar black background and white rectangle, saved as `test.mp4`.

These files are stored in the `tests/data` directory.

### Pytest Fixture

A `pytest` fixture named `test_files` is used to create the test files before the tests run and clean them up afterward.

---

## Test Cases

### 1. Image Detection Endpoint

- **Function**: `test_detect_image`
- **Description**: Tests the `/detect/image` endpoint by uploading a valid image file.
- **Assertions**:
  - Status code is `200`.
  - Response contains `scene_id` and `detections` fields.
  - `detections` is a list.

---

### 2. Video Detection Endpoint

- **Function**: `test_detect_video`
- **Description**: Tests the `/detect/video` endpoint by uploading a valid video file.
- **Assertions**:
  - Status code is `200`.
  - Response contains `scene_id` and `total_detections` fields.
  - `total_detections` is an integer.

---

### 3. Webcam Detection Endpoint

#### a. Invalid Duration
- **Function**: `test_webcam_detection_invalid_duration`
- **Description**: Tests the `/detect/webcam` endpoint with an invalid duration (`-1`).
- **Assertions**:
  - Status code is `422`.
  - Error message indicates the duration must be greater than 0.

#### b. Valid Duration
- **Function**: `test_webcam_detection`
- **Description**: Tests the `/detect/webcam` endpoint with a valid duration (`1` second).
- **Assertions**:
  - Status code is `200`.
  - Response contains `scene_id` and a `message` indicating completion.

---

### 4. RTSP Stream Detection Endpoint

#### a. Valid RTSP URL
- **Function**: `test_rtsp_detection`
- **Description**: Tests the `/detect/rtsp` endpoint with a valid RTSP URL.
- **Assertions**:
  - Status code is `200`.
  - Response contains `scene_id` and a `message` indicating completion.

#### b. Invalid RTSP URL
- **Function**: `test_invalid_rtsp_url`
- **Description**: Tests the `/detect/rtsp` endpoint with an invalid RTSP URL.
- **Assertions**:
  - Status code is `422`.
  - Error message indicates an invalid URL format.

---

### 5. Invalid File Uploads

- **Function**: `test_invalid_image_upload`
- **Description**: Tests the `/detect/image` endpoint by uploading an invalid file (e.g., a text file).
- **Assertions**:
  - Status code is `400`.
  - Error message indicates an invalid file type.

---

### 6. Missing File Uploads

- **Function**: `test_missing_file`
- **Description**: Tests the `/detect/image` endpoint without uploading any file.
- **Assertions**:
  - Status code is `422`.
  - Error message indicates a validation error.

---

## Running the Tests

To execute the tests, follow these steps:

1. Open a terminal and navigate to the directory containing `test_apis.py`.
2. Run the tests using the following command:

   ```bash
   pytest -v test_apis.py