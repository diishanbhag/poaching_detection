import time
import os
import sys
from kafka import KafkaProducer

# --- Kafka Configuration ---
KAFKA_BROKER = 'localhost:9092'
IMAGE_TOPIC = 'image-topic'

# --- Folder containing the images to process ---
IMAGE_DIR = 'testimages'

# --- Time to wait between sending each image (in seconds) ---
DELAY_BETWEEN_IMAGES = 2

# --- List of valid image file extensions ---
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

if __name__ == "__main__":
    print("--- Batch Producer Service ---")

    # 1. Check if the image directory exists
    if not os.path.isdir(IMAGE_DIR):
        print(f"FATAL: Directory '{IMAGE_DIR}' not found. Please create it and add your images.")
        sys.exit(1)

    # 2. Get a list of all image files in the directory
    image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    if not image_files:
        print(f"No images found in '{IMAGE_DIR}'. Exiting.")
        sys.exit(0)

    print(f"Found {len(image_files)} images to process.")

    # 3. Connect to Kafka
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: v, # We send raw bytes
            api_version=(0, 10, 1)
        )
    except Exception as e:
        print(f"FATAL: Failed to connect to Kafka broker at {KAFKA_BROKER}. Please ensure Kafka is running. Error: {e}")
        sys.exit(1)

    # 4. Loop through each image and send it to Kafka
    for i, filename in enumerate(image_files):
        image_path = os.path.join(IMAGE_DIR, filename)
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Send the image bytes to Kafka, using the filename as the key
            producer.send(IMAGE_TOPIC, key=filename.encode('utf-8'), value=image_bytes)
            producer.flush()
            print(f"({i+1}/{len(image_files)}) Sent {filename} to Kafka.")

            # Wait before sending the next image
            time.sleep(DELAY_BETWEEN_IMAGES)

        except Exception as e:
            print(f"   Error processing {filename}: {e}")

    # 5. Clean up and exit
    print("\nAll images have been sent to the pipeline.")
    producer.close()
    print("Producer has stopped.")
