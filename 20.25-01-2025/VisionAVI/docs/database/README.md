# Database System Documentation
A comprehensive guide to the object detection database system implementation.

## Table of Contents
1. [Overview](#overview)
2. [Database Schema](#database-schema)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Performance Considerations](#performance-considerations)

## Overview
The database system provides a thread-safe SQLite implementation for storing and managing:
- Scene metadata and images/videos
- Object detections with bounding boxes
- Manual and automated annotations
- Dataset assignments
- Scene descriptions

## Database Schema

### Tables Structure

#### 1. Scenes Table
```sql
CREATE TABLE scenes (
    scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    latitude REAL,
    longitude REAL,
    resolution TEXT,
    camera_id TEXT NOT NULL,
    media_path TEXT NOT NULL,
    processed BOOLEAN DEFAULT 0
);
```

#### 2. Detections Table
```sql
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id INTEGER NOT NULL,
    class_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    x_max REAL NOT NULL,
    y_max REAL NOT NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(scene_id) ON DELETE CASCADE
);
```

### Performance Optimizations
- Indexed timestamps for scene queries
- Indexed class labels for detection filtering
- Indexed annotation types for quick filtering

## Core Components

### SceneDatabase Class
Thread-safe database manager implementing:
- Connection pooling
- Transaction management
- CRUD operations
- Batch processing

#### Key Methods
1. Scene Management:
```python
def add_scene(metadata: Dict) -> int
def get_scene(scene_id: int) -> Optional[Dict]
def update_scene(scene_id: int, updates: Dict)
```

2. Detection Operations:
```python
def add_detections(scene_id: int, detections: List[Dict])
def get_detections_by_class(class_label: str, confidence_threshold: float = 0.5)
```

## Implementation Details

### Thread Safety
```python
class SceneDatabase:
    _local = threading.local()
    
    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn
```

### Data Types

#### Scene Metadata
```python
metadata = {
    "timestamp": str,      # ISO format
    "latitude": float,     # Optional
    "longitude": float,    # Optional
    "resolution": str,     # "WxH"
    "camera_id": str,      
    "media_path": str
}
```

#### Detection Format
```python
detection = {
    "class": str,         # Object class
    "confidence": float,  # 0.0-1.0
    "x_min": float,      # Normalized coordinates
    "y_min": float,
    "x_max": float,
    "y_max": float
}
```

## Usage Examples

### 1. Adding a New Scene
```python
db = SceneDatabase()
scene_id = db.add_scene({
    "timestamp": datetime.now().isoformat(),
    "camera_id": "cam_01",
    "media_path": "/path/to/image.jpg",
    "resolution": "1920x1080"
})
```

### 2. Recording Detections
```python
detections = [{
    "class": "person",
    "confidence": 0.95,
    "x_min": 0.1,
    "y_min": 0.2,
    "x_max": 0.3,
    "y_max": 0.4
}]
db.add_detections(scene_id, detections)
```

## Performance Considerations

### 1. Connection Management
- Thread-local connections prevent concurrency issues
- Connections are reused within threads
- Automatic cleanup on thread termination

### 2. Query Optimization
```sql
CREATE INDEX idx_scenes_timestamp ON scenes(timestamp);
CREATE INDEX idx_detections_class ON detections(class_label);
CREATE INDEX idx_annotations_type ON annotations(label_type);
```

### 3. Batch Operations
- Uses `executemany` for multiple inserts
- Transaction management for atomic operations
- Optimized for bulk data processing

## Error Handling

### 1. Database Errors
```python
try:
    cursor.execute(sql, values)
except sqlite3.Error as e:
    logger.error(f"Database error: {str(e)}")
    raise DatabaseError(str(e))
```

### 2. Data Validation
- Type checking before insertion
- Constraint enforcement
- Coordinate validation

## Dependencies
- sqlite3
- threading
- datetime
- typing (List, Dict, Optional)