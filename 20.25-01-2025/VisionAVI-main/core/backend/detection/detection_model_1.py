# Python module: detection_model_1.py

# Import the required libraries
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Union, Tuple


# Class for YOLOv5 API
class YOLOv5:
    """A wrapper class for YOLOv5 object detection model.

    This class provides a high-level interface for object detection using YOLOv5.
    It supports detection on images, videos, and real-time frames, with standardized
    output formats and metadata collection.

    Attributes:
        model: An instance of YOLO model loaded from the specified weights file.

    Methods:
        process_detections(results):
            Processes raw YOLOv5 results into a standardized format.

        detect_image(image_path):
            Performs object detection on a single image file.

        detect_frame(frame):
            Performs object detection on a single video frame.

        detect_video(video_path):
            Performs object detection on a video file.

        info():
            Returns model architecture and parameter information.

        save_results(results, output_path):
            Saves detection results to specified path.

    Example:
        ```python
        # Initialize detector with model weights
        detector = YOLOv5("yolov5s.pt")

        # Perform detection on an image
        detections, metadata = detector.detect_image("image.jpg")

        # Process video file
        video_detections = detector.detect_video("video.mp4")

        # Get model information
        model_info = detector.info()
        ```

    Note:
        This implementation uses the Ultralytics YOLOv5 package and requires
        the model weights to be compatible with YOLOv5 architecture.
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def process_detections(self, results) -> List[Dict]:
        """Process detection results into a standardized format."""
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "class_id": class_id,
                    }
                )
        return detections

    def detect_image(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Perform detection on a single image.

        Returns:
            Tuple[List[Dict], Dict]: (detections, metadata)
        """
        results = self.model.predict(source=image_path)
        detections = self.process_detections(results)
        metadata = {
            "resolution": f"{results[0].orig_shape[0]}x{results[0].orig_shape[1]}"
        }
        return detections, metadata

    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform detection on a single frame.
        """
        results = self.model.predict(source=frame, stream=True)
        return self.process_detections(results)

    def detect_video(self, video_path: str) -> List[Dict]:
        """
        Perform detection on a video file.
        """
        results = self.model.predict(source=video_path)
        return self.process_detections(results)

    def info(self):
        """
        Get model information.
        """
        return self.model.info()

    def save_results(self, results, output_path: str):
        """
        Save detection results to file.
        """
        results.save(output_path)

    def get_bbox_coordinates(self, detection: Dict) -> List[float]:
        """
        Helper method to get bbox coordinates in array format for visualization.

        Args:
            detection (Dict): Detection dictionary with x_min, y_min, x_max, y_max

        Returns:
            List[float]: [x_min, y_min, x_max, y_max]
        """
        return [
            detection["x_min"],
            detection["y_min"],
            detection["x_max"],
            detection["y_max"],
        ]


####################################################################### TESTING CODE #######################################################################################


if __name__ == "__main__":
    """
    Test code for YOLOv5 class functionality.
    Uncomment sections to test different features.
    """

    # Initialize the model
    model = YOLOv5("yolov5s.pt")

    # Test 1: Get model information
    print("\n=== Model Information ===")
    model_info = model.info()
    print(model_info)

    # Test 2: Image detection
    """
    print("\n=== Image Detection Test ===")
    image_path = "path/to/test/image.jpg"
    detections, metadata = model.detect_image(image_path)
    print(f"Image Resolution: {metadata['resolution']}")
    print(f"Number of detections: {len(detections)}")
    for det in detections:
        print(f"Detected {det['class']} with confidence {det['confidence']:.2f}")
        print(f"Bounding box: {det['bbox']}")
    """

    # Test 3: Video detection
    """
    print("\n=== Video Detection Test ===")
    video_path = "path/to/test/video.mp4"
    video_detections = model.detect_video(video_path)
    print(f"Total detections in video: {len(video_detections)}")
    """

    # Test 4: Webcam frame detection
    """
    print("\n=== Webcam Frame Detection Test ===")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame_detections = model.detect_frame(frame)
        print(f"Detections in frame: {len(frame_detections)}")
        for det in frame_detections:
            print(f"Detected {det['class']} with confidence {det['confidence']:.2f}")
    cap.release()
    """

    # Test 5: Save detection results
    """
    print("\n=== Save Results Test ===")
    test_image = "path/to/test/image.jpg"
    results = model.model.predict(source=test_image)
    model.save_results(results, "test_output")
    print("Results saved to 'test_output'")
    """

    print("\nTests completed!")
