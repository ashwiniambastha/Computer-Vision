# Python module: schema.py

# Import the required libraries
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict, Optional


class SceneDatabase:
    """
    A class to manage a SQLite database for storing scenes, detections, and annotations.
    """

    _local = threading.local()

    def __init__(self, db_path: str = "scenes.db"):
        """
        Initialize the SceneDatabase instance and create the required tables.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to "scenes.db".
        """
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_conn()
        conn.execute(
            """CREATE TABLE IF NOT EXISTS scenes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            resolution TEXT,
            camera_id TEXT NOT NULL,
            media_path TEXT NOT NULL
        )"""
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id INTEGER NOT NULL,
                class TEXT NOT NULL,
                confidence REAL NOT NULL,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                FOREIGN KEY (scene_id) REFERENCES scenes (id)
            )
        """
        )
        conn.commit()

    def add_scene(self, metadata: Dict) -> int:
        """
        Add a new scene to the database.

        Args:
            metadata (Dict): A dictionary containing scene metadata, including:
                - timestamp (str): Timestamp of the scene.
                - latitude (float, optional): Latitude of the scene location.
                - longitude (float, optional): Longitude of the scene location.
                - resolution (str, optional): Resolution of the scene.
                - camera_id (str): ID of the camera capturing the scene.
                - media_path (str): Path to the media file.

        Returns:
            int: The ID of the newly inserted scene.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        sql = """INSERT INTO scenes (
            timestamp, latitude, longitude, resolution, 
            camera_id, media_path
        ) VALUES (?, ?, ?, ?, ?, ?)"""
        values = (
            metadata["timestamp"],
            metadata.get("latitude"),
            metadata.get("longitude"),
            metadata.get("resolution"),
            metadata["camera_id"],
            metadata["media_path"],
        )
        cursor.execute(sql, values)
        conn.commit()
        return cursor.lastrowid

    def add_detections(self, scene_id: int, detections: List[Dict]):
        """
        Add multiple detections for a specific scene.

        Args:
            scene_id (int): The ID of the scene to associate the detections with.
            detections (List[Dict]): A list of dictionaries, each containing:
                - class (str): The class label of the detection.
                - confidence (float): Confidence score of the detection.
                - bbox (List[float]): Bounding box coordinates [x_min, y_min, x_max, y_max].
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        sql = """INSERT INTO detections (
            scene_id, class, confidence,
            x_min, y_min, x_max, y_max
        ) VALUES (?, ?, ?, ?, ?, ?, ?)"""
        values = [
            (
                scene_id,
                d["class"],
                d["confidence"],
                d["x_min"],  # Using individual coordinates
                d["y_min"],
                d["x_max"],
                d["y_max"],
            )
            for d in detections
        ]
        cursor.executemany(sql, values)
        conn.commit()

    def add_annotation(self, scene_id: int, annotation: Dict):
        """
        Add an annotation for a specific scene.

        Args:
            scene_id (int): The ID of the scene to associate the annotation with.
            annotation (Dict): A dictionary containing annotation details, including:
                - label_type (str): Type of the label (e.g., "manual", "auto").
                - description (str, optional): Description of the annotation.
                - class_label (str, optional): Class label of the annotation.
                - x_min, y_min, x_max, y_max (float, optional): Bounding box coordinates.
                - annotated_by (str): Name or ID of the annotator.
        """
        sql = """INSERT INTO annotations (
            scene_id, label_type, description,
            class_label, x_min, y_min, x_max, y_max, annotated_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        values = (
            scene_id,
            annotation["label_type"],
            annotation.get("description"),
            annotation.get("class_label"),
            annotation.get("x_min"),
            annotation.get("y_min"),
            annotation.get("x_max"),
            annotation.get("y_max"),
            annotation.get("annotated_by"),
        )
        with self._get_conn():
            self._get_conn().execute(sql, values)

    def get_scenes_by_time_range(self, start: datetime, end: datetime) -> List[Dict]:
        """
        Retrieve scenes within a specific time range.

        Args:
            start (datetime): Start of the time range.
            end (datetime): End of the time range.

        Returns:
            List[Dict]: A list of scenes within the specified time range.
        """
        sql = """SELECT * FROM scenes 
                WHERE timestamp BETWEEN ? AND ?"""
        return (
            self._get_conn()
            .execute(sql, (start.isoformat(), end.isoformat()))
            .fetchall()
        )

    def get_detections_by_class(
        self, class_label: str, confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve detections for a specific class label with a confidence threshold.

        Args:
            class_label (str): The class label to filter detections.
            confidence_threshold (float): Minimum confidence score. Defaults to 0.5.

        Returns:
            List[Dict]: A list of detections matching the criteria.
        """
        sql = """SELECT * FROM detections 
                WHERE class_label = ? AND confidence >= ?"""
        return (
            self._get_conn()
            .execute(sql, (class_label, confidence_threshold))
            .fetchall()
        )

    def assign_to_dataset(self, scene_ids: List[int], dataset_type: str):
        """
        Assign scenes to a specific dataset type (e.g., "train", "test").

        Args:
            scene_ids (List[int]): List of scene IDs to assign.
            dataset_type (str): The dataset type to assign the scenes to.
        """
        sql = """INSERT INTO datasets (scene_id, dataset_type) 
                 VALUES (?, ?)"""
        values = [(sid, dataset_type) for sid in scene_ids]
        with self._get_conn():
            self._get_conn().executemany(sql, values)

    def get_training_data(self) -> List[Dict]:
        """
        Retrieve all scenes and descriptions assigned to the "train" dataset.

        Returns:
            List[Dict]: A list of training data scenes with descriptions.
        """
        sql = """SELECT s.*, d.description 
                 FROM scenes s
                 JOIN datasets ds ON s.scene_id = ds.scene_id
                 LEFT JOIN scene_descriptions d ON s.scene_id = d.scene_id
                 WHERE ds.dataset_type = 'train' """
        return self._get_conn().execute(sql).fetchall()

    def add_scene_description(
        self, scene_id: int, description: str, confidence: float, model_version: str
    ):
        """
        Add a description for a specific scene.

        Args:
            scene_id (int): The ID of the scene to describe.
            description (str): The description text.
            confidence (float): Confidence score of the description.
            model_version (str): Version of the model generating the description.
        """
        sql = """INSERT INTO scene_descriptions 
                 (scene_id, description, confidence, model_version)
                 VALUES (?, ?, ?, ?)"""
        with self._get_conn():
            self._get_conn().execute(
                sql, (scene_id, description, confidence, model_version)
            )

    def incremental_update(self, new_scenes: List[Dict]):
        """
        Perform an incremental update by adding new scenes, detections, and descriptions.

        Args:
            new_scenes (List[Dict]): A list of dictionaries, each containing:
                - metadata (Dict): Scene metadata.
                - detections (List[Dict], optional): List of detections.
                - description (str, optional): Scene description.
                - confidence (float, optional): Confidence score of the description.
                - model_version (str, optional): Model version for the description.
        """
        with self._get_conn():
            for scene in new_scenes:
                scene_id = self.add_scene(scene["metadata"])
                if "detections" in scene:
                    self.add_detections(scene_id, scene["detections"])
                if "description" in scene:
                    self.add_scene_description(
                        scene_id,
                        scene["description"],
                        scene["confidence"],
                        scene["model_version"],
                    )

    def close(self):
        """
        Close the database connection.
        """
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def get_scene(self, scene_id: int) -> Optional[Dict]:
        """Read a single scene by ID"""
        sql = """SELECT * FROM scenes WHERE id = ?"""
        cursor = self._get_conn().execute(sql, (scene_id,))
        row = cursor.fetchone()
        return self._row_to_dict(row, cursor.description) if row else None

    def update_scene(self, scene_id: int, updates: Dict):
        """Update scene metadata"""
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        sql = f"""UPDATE scenes SET {set_clause} WHERE id = ?"""
        values = list(updates.values()) + [scene_id]

        with self._get_conn() as conn:
            conn.execute(sql, values)

    def _row_to_dict(self, row, description):
        """Convert SQLite row to dictionary"""
        if row is None:
            return None
        return {description[i][0]: value for i, value in enumerate(row)}
