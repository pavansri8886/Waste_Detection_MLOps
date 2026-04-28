import os
from datetime import datetime
from typing import List

import folium
import requests
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="Waste Detection App", layout="wide")
st.title("Waste Detection — Streamlit Interface")

st.markdown("This app uses the Waste Detection API to select a model, upload an image, and display the prediction and detection history.")


def fetch_models() -> List[dict]:
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.sidebar.error(f"Unable to reach the API at {API_URL}: {exc}")
        return []


def fetch_history() -> List[dict]:
    try:
        response = requests.get(f"{API_URL}/history", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Unable to load history from {API_URL}: {exc}")
        return []


def parse_timestamp(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


models = fetch_models()
model_names = [model["name"] for model in models] if models else []

history = fetch_history()

sources = sorted({row["source"] for row in history}) if history else []
source_filter = st.sidebar.multiselect("Sources", options=sources, default=sources)

model_options = sorted({row["model_name"] for row in history}) if history else model_names
model_filter = st.sidebar.multiselect("Models", options=model_options, default=model_options)

if history:
    timestamps = [parse_timestamp(row["timestamp"]) for row in history if parse_timestamp(row["timestamp"])]
    min_date = min(timestamps).date() if timestamps else datetime.utcnow().date()
    max_date = max(timestamps).date() if timestamps else datetime.utcnow().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
else:
    date_range = None

st.sidebar.header("Available models")
for model in models:
    st.sidebar.write(f"**{model['name']}** — version {model['version']} registered at {model['registered_at']}")

with st.form("predict_form"):
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    if model_names:
        model_name = st.selectbox("Select model", model_names)
    else:
        st.warning("No model definitions are available. Check the API or register a model first.")
        model_name = None
    latitude = st.text_input("Latitude", "48.8566")
    longitude = st.text_input("Longitude", "2.3522")
    submit = st.form_submit_button("Send prediction")

if submit:
    if not uploaded_file:
        st.warning("Please upload an image before submitting.")
    elif not model_name:
        st.warning("Please select a model.")
    else:
        try:
            latitude_value = float(latitude)
            longitude_value = float(longitude)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {
                "latitude": latitude_value,
                "longitude": longitude_value,
                "model_name": model_name,
            }
            result = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
            if result.status_code == 200:
                st.success("Prediction received successfully.")
                st.json(result.json())
            else:
                st.error(f"API error {result.status_code}: {result.text}")
        except ValueError:
            st.error("Latitude and longitude must be numeric.")


if history:
    st.subheader("Historical detections")

    filtered_history = []
    start_date, end_date = date_range if date_range else (None, None)
    for row in history:
        row_date = parse_timestamp(row["timestamp"]) if row.get("timestamp") else None
        if source_filter and row["source"] not in source_filter:
            continue
        if model_filter and row["model_name"] not in model_filter:
            continue
        if row_date and start_date and end_date and not (start_date <= row_date.date() <= end_date):
            continue
        filtered_history.append(row)

    st.write(f"Showing {len(filtered_history)} historical detections.")

    if filtered_history:
        map_center = [filtered_history[0]["latitude"], filtered_history[0]["longitude"]]
    else:
        map_center = [48.8566, 2.3522]

    folium_map = folium.Map(location=map_center, zoom_start=6)
    cluster = MarkerCluster().add_to(folium_map)

    for row in filtered_history:
        color = "orange" if row["source"] == "drone_patrol" else "red"
        popup_text = (
            f"<b>Source:</b> {row['source']}<br>"
            f"<b>Model:</b> {row['model_name']}<br>"
            f"<b>Confidence:</b> {row['confiance']}<br>"
            f"<b>Time:</b> {row['timestamp']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=300),
        ).add_to(cluster)

    st_folium(folium_map, width=700, height=520)
else:
    st.info("No historical detections are available yet.")
