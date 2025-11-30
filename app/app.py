# src/app_ui.py
import streamlit as st
import requests
from PIL import Image
import pandas as pd
import sqlite3
import os

# ------------------ API & Paths ------------------
BASE_URL = "http://localhost:8000"
PREDICT_URL = f"{BASE_URL}/predict"
UPLOAD_URL = f"{BASE_URL}/upload_training_image"
RETRAIN_URL = f"{BASE_URL}/retrain"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
DB_PATH = os.path.join(BASE_DIR, "training_images.db")

st.title("Fruits & Vegetables Classifier Application")

# ------------------ TAB LAYOUT ------------------
tabs = st.tabs([
    "🔮 Predict",
    "📤 Upload Training Data",
    "📊 Data Visualization",
    "⚡ Retrain Model"
])

# ==========================================================
# 1️⃣ PREDICTION TAB
# ==========================================================
with tabs[0]:
    st.header("Upload an Image for Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            with st.spinner("Sending to API..."):
                try:
                    resp = requests.post(PREDICT_URL, files=files, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Prediction: **{data['label']}** — Confidence: {data['confidence']:.2f}")
                    else:
                        st.error(f"API Error: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Error contacting API: {e}")


# ==========================================================
# 2️⃣ UPLOAD TRAINING DATA TAB
# ==========================================================
with tabs[1]:
    st.header("Upload New Training Images for Retraining")

    upload_label = st.text_input("Label / class name (e.g., apple)")
    upload_file = st.file_uploader("Choose an image to add", type=["jpg","jpeg","png"])

    if st.button("Upload to Training Set"):
        if not upload_label:
            st.warning("Please enter a label.")
        elif not upload_file:
            st.warning("Please choose an image file.")
        else:
            files = {"file": (upload_file.name, upload_file.getvalue(), upload_file.type)}
            data = {"label": upload_label}

            with st.spinner("Uploading..."):
                try:
                    resp = requests.post(UPLOAD_URL, files=files, data=data, timeout=60)
                    if resp.status_code == 200:
                        st.success("Saved to training set: " + resp.json().get("path",""))
                    else:
                        st.error("Upload failed: " + resp.text)
                except Exception as e:
                    st.error("Error contacting API: " + str(e))


# ==========================================================
# 3️⃣ DATA VISUALIZATION TAB
# ==========================================================
with tabs[2]:
    st.header("Training Data Visualization")

    # --- Class distribution ---
    st.subheader("📌 Class Distribution (Image Count per Label)")
    if os.path.exists(TRAIN_DIR):
        class_counts = {}
        for label in os.listdir(TRAIN_DIR):
            label_path = os.path.join(TRAIN_DIR, label)
            if os.path.isdir(label_path):
                class_counts[label] = len(os.listdir(label_path))
        if class_counts:
            df_counts = pd.DataFrame(list(class_counts.items()), columns=["Label","Image Count"])
            st.bar_chart(df_counts.set_index("Label"))
            st.dataframe(df_counts)
        else:
            st.warning("No training data found.")
    else:
        st.error("Training directory not found.")

    st.markdown("---")

    # --- Upload history from DB ---
    st.subheader("📁 Uploaded Images Log (From Database)")
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df_uploads = pd.read_sql_query("SELECT * FROM uploads", conn)
            conn.close()
            st.dataframe(df_uploads)
        except Exception as e:
            st.error(f"Error loading database: {e}")
    else:
        st.warning("Upload database not found.")


# ==========================================================
# 4️⃣ RETRAIN MODEL TAB
# ==========================================================
with tabs[3]:
    st.header("⚡ Retrain Model")

    st.write("Click the button below to retrain the model using the current training dataset.")
    if st.button("Retrain Model"):
        with st.spinner("Retraining model... this may take a few minutes"):
            try:
                resp = requests.post(RETRAIN_URL, timeout=300)  # longer timeout
                if resp.status_code == 200:
                    metrics = resp.json().get("metrics", {})
                    st.success("Model retrained successfully!")
                    st.write("### Training Metrics")
                    st.write(f"Training Accuracy: {metrics.get('train_acc',0):.4f}")
                    st.write(f"Training Loss: {metrics.get('train_loss',0):.4f}")
                    st.write(f"Validation Accuracy: {metrics.get('val_acc',0):.4f}")
                    st.write(f"Validation Loss: {metrics.get('val_loss',0):.4f}")
                else:
                    st.error(f"Retrain failed: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Error contacting API: {e}")
