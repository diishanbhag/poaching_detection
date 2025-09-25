# Poaching Detection with Kafka, YOLOv8, and GNN  

## Overview  
This project implements a real-time poaching detection system by integrating:  
- **YOLOv8** → for fast, specialized object detection (vehicles, campfires, water bodies).  
- **Graph Neural Networks (GNNs)** → for contextual reasoning, modeling relationships between detected objects.  
- **Apache Kafka** → for building an event-driven streaming pipeline, enabling scalable, fault-tolerant data ingestion and processing.  

The system analyzes aerial/satellite images and outputs a poaching risk score, with results streamed to a live dashboard.  

<img width="1600" height="731" alt="image" src="https://github.com/user-attachments/assets/6c795bc2-1814-475c-83de-9076189f54ce" />
<img width="1600" height="480" alt="image" src="https://github.com/user-attachments/assets/8ccf5e0c-1a71-4391-a172-830dff5b0e11" />

<img width="1600" height="769" alt="image" src="https://github.com/user-attachments/assets/d3e00d94-4c0e-479d-8521-3bd30d4807f3" />



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
```
# GNN Service and Dashboard

## GNN Service
- Consumes detections from **`detection-topic`**.  
- Builds a **graph representation**:  
  - **Nodes** = detected objects (`vehicle`, `campfire`, `water`).  
  - **Node features** = `[class_onehot, area, x, y]`.  
  - **Edges** = based on proximity, weights = **Inverse Distance Weighting (IDW)**.  
- Processes the graph through a **2-layer Graph Convolutional Network (GCN)**.  
- Outputs a **poaching risk probability**.  
- Stores results in **Redis** for the dashboard.  

---

## Dashboard
- Subscribes to Redis.  
- Visualizes detections and risk scores.  
- Provides real-time monitoring for operators.  



---

## Running the Pipeline

### 1. Clone the Repository
```bash
git clone https://github.com/diishanbhag/poaching_detection.git
cd poaching_detection
```
### 2. Start kafka and redis
```bash
# Start Zookeeper & Kafka
bin/zookeeper-server-start.sh config/zookeeper.properties
bin/kafka-server-start.sh config/server.properties
# Start Redis
redis-server
```
### 3. Start the Services
```bash
# Terminal 1: Run Producer
python producer.py
# Terminal 2: Run YOLO Service
python yolo_service.py --weights vehicle.pt wildfire.pt water.pt

# Terminal 3: Run GNN Service
python gnn_service.py

# Terminal 4: Run Dashboard
python dashboard.py
# Terminal 1: Run Producer
python producer.py

# Terminal 2: Run YOLO Service
python yolo_service.py --weights vehicle.pt wildfire.pt water.pt

# Terminal 3: Run GNN Service
python gnn_service.py

# Terminal 4: Run Dashboard
python dashboard.py
```


