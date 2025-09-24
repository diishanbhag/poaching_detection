# Filename: detection_logic.py (CORRECTED)

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import sys

# --- Load all models once to be efficient ---
try:
    vehicle_model = YOLO("vehicles.pt")
    wildfire_model = YOLO("wildfire.pt")
    water_model = YOLO("water.pt")
    print("Detection models loaded successfully in detection_logic.py")
except Exception as e:
    print(f"FATAL: Could not load models in detection_logic.py. Make sure .pt files are present. Error: {e}")
    sys.exit(1)

# --- Helper functions for Water Detection (EXACTLY from your water_yolo.py) ---
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def nms(dets, iou_thresh=0.35):
    if not dets: return []
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best["bbox"], d["bbox"]) < iou_thresh]
    return keep

def blue_ratio(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    blue_pixels = cv2.countNonZero(mask)
    return blue_pixels / total_pixels if total_pixels > 0 else 0

# --- Main Detection Functions (CORRECTED to use .tolist() for stability) ---

def run_vehicle_detection(image_path, grid_size=7, overlap=50, conf=0.25):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    tile_w, tile_h = w // grid_size, h // grid_size
    detections = []
    for row in range(grid_size):
        for col in range(grid_size):
            left, upper = max(col * tile_w - overlap, 0), max(row * tile_h - overlap, 0)
            right, lower = min((col + 1) * tile_w + overlap, w), min((row + 1) * tile_h + overlap, h)
            tile = image.crop((left, upper, right, lower))
            results = vehicle_model(tile, conf=conf, verbose=False)
            for result in results:
                for box in result.boxes:
                    conf_score = float(box.conf[0])
                    # --- FIXED ---
                    # Using .tolist() which is more stable than .cpu().numpy()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # --- END FIX ---
                    detections.append({
                        "class": "vehicle", "conf": conf_score,
                        "bbox": [int(x1 + left), int(y1 + upper), int(x2 + left), int(y2 + upper)]
                    })
    return detections

def run_wildfire_detection(image_path, grid_size=7, overlap=50, conf=0.25):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    tile_w, tile_h = w // grid_size, h // grid_size
    detections = []
    for row in range(grid_size):
        for col in range(grid_size):
            left, upper = max(col * tile_w - overlap, 0), max(row * tile_h - overlap, 0)
            right, lower = min((col + 1) * tile_w + overlap, w), min((row + 1) * tile_h + overlap, h)
            tile = image.crop((left, upper, right, lower))
            results = wildfire_model(tile, conf=conf, verbose=False)
            for result in results:
                for box in result.boxes:
                    conf_score = float(box.conf[0])
                    # --- FIXED ---
                    # Using .tolist() which is more stable than .cpu().numpy()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # --- END FIX ---
                    detections.append({
                        "class": "wildfire", "conf": conf_score,
                        "bbox": [int(x1 + left), int(y1 + upper), int(x2 + left), int(y2 + upper)]
                    })
    return detections

def run_water_detection(image_path, grid_size=7, yolo_conf=0.6, blue_thresh=0.15, iou_thresh=0.35):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return []
    H, W = img_bgr.shape[:2]
    tile_w, tile_h = W // grid_size, H // grid_size
    dets = []
    for r in range(grid_size):
        for c in range(grid_size):
            left, top = c * tile_w, r * tile_h
            right, bottom = (c + 1) * tile_w, (r + 1) * tile_h
            tile = img_bgr[top:bottom, left:right]
            if tile.size == 0: continue
            results = water_model(tile[..., ::-1], conf=yolo_conf, verbose=False)
            for res in results:
                for box in res.boxes:
                    conf = float(box.conf[0])
                    # --- FIXED ---
                    # Using .tolist() which is more stable than .cpu().numpy()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # --- END FIX ---
                    abs_bbox = [int(x1+left), int(y1+top), int(x2+left), int(y2+top)]
                    crop = img_bgr[abs_bbox[1]:abs_bbox[3], abs_bbox[0]:abs_bbox[2]]
                    if crop.size > 0 and blue_ratio(crop) >= blue_thresh:
                        dets.append({"class": "water", "conf": conf, "bbox": abs_bbox})
    return nms(dets, iou_thresh)
