# Filename: yolo_service.py (UPDATED)

import json
from kafka import KafkaConsumer, KafkaProducer
import sys
import os
import uuid
from PIL import Image

# --- Import the new, advanced detection logic ---
from detection_logic import run_vehicle_detection, run_wildfire_detection, run_water_detection

# --- Kafka Configuration ---
KAFKA_BROKER = 'localhost:9092'
IMAGE_TOPIC = 'image-topic'
DETECTION_TOPIC = 'detection-topic'

# --- Directory for temporarily saving images for processing ---
TEMP_DIR = "temp_processing"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- Kafka Consumer/Producer Setup ---
try:
    consumer = KafkaConsumer(
        IMAGE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='earliest',
        value_deserializer=lambda v: v,
        api_version=(0, 10, 1)
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        api_version=(0, 10, 1)
    )
except Exception as e:
    print(f"FATAL: Could not connect to Kafka broker. Error: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("--- YOLO Service with User-Provided Advanced Logic ---")
    print("Waiting for images from Kafka...")
    for message in consumer:
        image_key = message.key.decode('utf-8')
        image_bytes = message.value
        print(f"\n-> Received image: {image_key}")

        # Create a unique temporary file path
        temp_image_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")

        try:
            # Save the image bytes to the temporary file
            with open(temp_image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Run all detection functions from your exact logic
            print("   Running advanced detection logic...")
            vehicle_dets = run_vehicle_detection(temp_image_path)
            wildfire_dets = run_wildfire_detection(temp_image_path)
            water_dets = run_water_detection(temp_image_path)

            all_detections = vehicle_dets + wildfire_dets + water_dets
            print(f"   Found {len(all_detections)} total objects.")

            # Get image size
            with Image.open(temp_image_path) as img:
                image_size = img.size

            output_data = {
                "image_id": image_key,
                "detections": all_detections,
                "image_size": image_size
            }

            producer.send(DETECTION_TOPIC, value=output_data)
            producer.flush()
            print(f"   Sent detections for {image_key} to topic '{DETECTION_TOPIC}'")

        except Exception as e:
            print(f"   Error processing image {image_key}: {e}")
        finally:
            # Clean up by removing the temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
