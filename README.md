# Poaching Detection with Kafka, YOLOv8, and GNN  

## Overview  
This project implements a real-time poaching detection system by integrating:  
- **YOLOv8** → for fast, specialized object detection (vehicles, campfires, water bodies).  
- **Graph Neural Networks (GNNs)** → for contextual reasoning, modeling relationships between detected objects.  
- **Apache Kafka** → for building an event-driven streaming pipeline, enabling scalable, fault-tolerant data ingestion and processing.  

The system analyzes aerial/satellite images and outputs a poaching risk score, with results streamed to a live dashboard.  

---

## System Architecture  

    ┌───────────┐
    │ Producer  │  (image ingestion → Kafka topic: image-topic)
    └─────┬─────┘
          │
          ▼
    ┌─────────────┐
    │ YOLO Service│  (consumes images → detects objects → produces JSON to detection-topic)
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │ GNN Service │  (consumes detections → builds graph → predicts risk → pushes to Redis)
    └─────┬───────┘
          │
          ▼
    ┌──────────────┐
    │ Dashboard    │  (visualizes detections & risk score in real time)
    └──────────────┘

---

## Components  

### Producer  
- Reads images from a directory (`testimages/`).  
- Publishes raw image bytes into **Kafka topic: `image-topic`**.  

### YOLO Service  
- Subscribes to `image-topic`.  
- Runs YOLOv8 detection:  
  - **Vehicles & Campfires**: tiling + detection.  
  - **Water**: tiling + blue-ratio validation + custom IoU-based NMS.  
- Publishes structured JSON detections into **`detection-topic`**.  

**Detection JSON Example:**  
```json
{
  "image_id": "frame_001.jpg",
  "detections": [
    {"class": "vehicle", "conf": 0.91, "x": 0.45, "y": 0.62, "area": 0.03},
    {"class": "water", "conf": 0.88, "x": 0.72, "y": 0.35, "blue_score": 0.21}
  ]
}
