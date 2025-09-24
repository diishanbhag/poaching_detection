# Filename: gnn_service.py (UPDATED)

import json
from kafka import KafkaConsumer
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys
import redis
import pickle
import datetime

# --- Kafka Configuration ---
KAFKA_BROKER = 'localhost:9092'
DETECTION_TOPIC = 'detection-topic'

# --- Redis Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# --- GNN Model Definition ---
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = x.mean(dim=0)
        x = self.fc(x)
        return torch.sigmoid(x)

# --- Graph Building Logic (aligned with your scripts) ---
def build_graph(detections, image_size):
    if not detections: return None
    # UPDATED class map to match your exact script classes
    class_map = {"vehicle": 0, "wildfire": 1, "water": 2}
    nodes, positions = [], []
    
    for det in detections:
        if det['class'] not in class_map:
            continue
            
        class_vec = [0] * len(class_map)
        class_vec[class_map[det['class']]] = 1
        
        bbox = det['bbox']
        w, h = image_size
        centroid_x = ((bbox[0] + bbox[2]) / 2) / w
        centroid_y = ((bbox[1] + bbox[3]) / 2) / h
        area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)
        
        nodes.append(class_vec + [centroid_x, centroid_y, area])
        positions.append([centroid_x, centroid_y])

    if not nodes: return None
        
    dist_matrix = euclidean_distances(positions)
    adj_matrix = dist_matrix < 0.5 
    np.fill_diagonal(adj_matrix, 0)
    
    edge_index = torch.tensor(np.array(np.where(adj_matrix)), dtype=torch.long)
    x = torch.tensor(nodes, dtype=torch.float)

    if x.shape[0] > 0 and edge_index.shape[1] == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

# --- Setup Connections ---
try:
    consumer = KafkaConsumer(
        DETECTION_TOPIC, bootstrap_servers=KAFKA_BROKER, auto_offset_reset='earliest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8')), api_version=(0, 10, 1)
    )
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    r.ping()
    print("Successfully connected to Redis.")
except Exception as e:
    print(f"FATAL: Could not connect to Kafka or Redis. Error: {e}")
    sys.exit(1)

# --- GNN Model Initialization ---
input_dim = 3 + 3 
gnn_model = SimpleGNN(input_dim)
gnn_model.eval()

if __name__ == "__main__":
    print("--- GNN Service with Improved Logic ---")
    print("Waiting for detections from Kafka...")
    for message in consumer:
        kafka_data = message.value
        image_id = kafka_data.get("image_id", "Unknown")
        detections = kafka_data.get("detections", [])
        image_size = kafka_data.get("image_size")
        
        print(f"\n-> Received detections for image: {image_id}")
        if not image_size:
             print(f"   Skipping {image_id} due to missing image size data.")
             continue
        
        graph_data = build_graph(detections, image_size)
        confidence = 0.0
        if graph_data is not None and graph_data.x.shape[0] > 0:
            with torch.no_grad():
                confidence = gnn_model(graph_data).item()
        
        print(f"   ✅ FINAL RESULT for {image_id} -> Poaching Confidence: {confidence:.4f}")
        
        redis_payload = {
            "image_id": image_id, "confidence": confidence, "detections": detections,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        r.set("latest_result", pickle.dumps(redis_payload))
        r.rpush("all_results", pickle.dumps(redis_payload))
        print(f"   Result for {image_id} stored in Redis.")
