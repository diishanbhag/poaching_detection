# Filename: dashboard.py (UPDATED with interactive threshold)

import streamlit as st
import redis
import pickle
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw
import time
import os
from collections import Counter

# --- Page Configuration ---
st.set_page_config(
    page_title="Poaching Detection Dashboard",
    page_icon="🐅",
    layout="wide",
)

# --- Redis Connection ---
@st.cache_resource
def get_redis_connection():
    return redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

try:
    r = get_redis_connection()
    r.ping()
except redis.exceptions.ConnectionError as e:
    st.error(f"Could not connect to Redis. Please ensure the Redis server is running. Error: {e}")
    st.stop()


# --- Helper Functions ---
def draw_bounding_boxes(image_path, detections):
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        class_colors = {"vehicle": "cyan", "campfire": "red", "water": "blue"}
        for det in detections:
            bbox = det["bbox"]
            label = f"{det['class']} ({det['conf']:.2f})"
            color = class_colors.get(det["class"], "yellow")
            draw.rectangle(bbox, outline=color, width=4)
            text_bg = (bbox[0], bbox[1] - 15, bbox[0] + len(label) * 8, bbox[1])
            draw.rectangle(text_bg, fill="black")
            draw.text((bbox[0] + 2, bbox[1] - 15), label, fill=color)
        return image
    except FileNotFoundError:
        return None

def load_data_from_redis(key):
    data = r.get(key)
    if data:
        return pickle.loads(data)
    return None

def load_list_from_redis(key):
    data_list = r.lrange(key, 0, -1)
    return [pickle.loads(data) for data in data_list]

# --- Sidebar and Page Navigation ---
st.sidebar.title("🐅 Poaching Intelligence")
page = st.sidebar.radio("Navigation", ["Live Dashboard", "High Alerts History", "Analytics & Visuals"])

# --- NEW: Interactive Threshold Slider ---
st.sidebar.header("Alert Configuration")
alert_threshold = st.sidebar.slider(
    "Set High Alert Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75, # Default value
    step=0.05
)
st.sidebar.info(f"Alerts will be shown for confidence scores above **{alert_threshold:.2%}**.")

# =====================================================================================
#                                  LIVE DASHBOARD PAGE
# =====================================================================================
if page == "Live Dashboard":
    st.title("Live Poaching Detection Feed")
    placeholder = st.empty()
    while True:
        latest_data = load_data_from_redis("latest_result")
        with placeholder.container():
            if latest_data:
                image_id, confidence, detections, ts = latest_data.values()
                timestamp = pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')

                st.header(f"Processing")
                st.write(f"Last update: {timestamp}")

                col1, col2 = st.columns([2, 1])
                with col1:
                    image_path = os.path.join("testimages", image_id)
                    processed_image = draw_bounding_boxes(image_path, detections)
                    if processed_image:
                        st.image(processed_image, caption="Processed Image with Detections", use_column_width=True)
                    else:
                        st.warning(f"Image '{image_id}' not found in 'testimages' folder.")
                with col2:
                    st.subheader("Poaching Confidence Score")
                    st.metric(label="Confidence Level", value=f"{confidence:.2%}")
                    st.progress(confidence)
                    st.subheader("Object Detection Summary")
                    detection_counts = Counter([d['class'] for d in detections])
                    st.markdown(f"**Wildfires/Campfires:** `{detection_counts.get('campfire', 0)}`")
                    st.markdown(f"**Vehicles:** `{detection_counts.get('vehicle', 0)}`")
                    st.markdown(f"**Water Bodies:** `{detection_counts.get('water', 0)}`")
            else:
                st.info("Waiting for the first image to be processed by the pipeline...")
        time.sleep(2)

# =====================================================================================
#                               HIGH ALERTS HISTORY PAGE
# =====================================================================================
elif page == "High Alerts History":
    st.title("🚨 High Alerts History")
    st.write(f"Showing events where the poaching confidence score exceeded the selected threshold of **{alert_threshold:.2%}**.")
    
    all_results = load_list_from_redis("all_results")
    # NEW: Filter results based on the slider value
    high_alerts = [alert for alert in all_results if alert['confidence'] > alert_threshold]
    
    if not high_alerts:
        st.info("No alerts found for the selected threshold.")
    else:
        high_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        for alert in high_alerts:
            timestamp = pd.to_datetime(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            with st.expander(f"**Alert on `{alert['image_id']}` at {timestamp}** (Confidence: {alert['confidence']:.2%})"):
                col1, col2 = st.columns(2)
                image_path = os.path.join("testimages", alert['image_id'])
                processed_image = draw_bounding_boxes(image_path, alert['detections'])

                if processed_image:
                    col1.image(processed_image, use_column_width=True)
                else:
                    col1.warning(f"Image not found.")
                
                detection_counts = Counter([d['class'] for d in alert['detections']])
                col2.write("#### Detection Details")
                col2.json({
                    "Confidence Score": f"{alert['confidence']:.4f}",
                    "Detected Objects": dict(detection_counts)
                })

# =====================================================================================
#                               ANALYTICS & VISUALS PAGE
# =====================================================================================
elif page == "Analytics & Visuals":
    st.title("📊 Analytics & Visualizations")
    st.write(f"Showing analytics for events with a confidence score above **{alert_threshold:.2%}**.")

    all_results = load_list_from_redis("all_results")
    # NEW: Filter results based on the slider value
    high_alerts = [alert for alert in all_results if alert['confidence'] > alert_threshold]

    if not high_alerts:
        st.info("No high alert data available to generate visualizations for the selected threshold.")
    else:
        df = pd.DataFrame(high_alerts)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        st.subheader("High Alert Confidence Scores Over Time")
        fig_time = px.line(df, x='timestamp', y='confidence', title='Confidence Score of Alerts', markers=True,
                           labels={"timestamp": "Time of Alert", "confidence": "Confidence Score"})
        fig_time.update_traces(line_color='red')
        st.plotly_chart(fig_time, use_container_width=True)
        
        st.subheader("Distribution of Detected Objects in High Alerts")
        all_detections = [d['class'] for alert in high_alerts for d in alert['detections']]
        detection_counts = Counter(all_detections)
        
        if not detection_counts:
            st.warning("No objects were detected in the recorded high alerts.")
        else:
            pie_df = pd.DataFrame(detection_counts.items(), columns=['Object Type', 'Count'])
            fig_pie = px.pie(pie_df, names='Object Type', values='Count', title='Object Types in High Alerts',
                             color_discrete_map={'campfire': 'red', 'vehicle': 'cyan', 'water': 'blue'})
            st.plotly_chart(fig_pie, use_container_width=True)
