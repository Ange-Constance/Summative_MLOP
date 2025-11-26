import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_BASE = 'http://127.0.0.1:8000'

PREDICT_URL = f'{API_BASE}/predict'
UPLOAD_URL = f'{API_BASE}/upload_training_image'
RETRAIN_URL = f'{API_BASE}/retrain_model'
UPTIME_URL = f'{API_BASE}/model_uptime'
METRICS_URL = f'{API_BASE}/get_training_metrics'
ACC_PLOT_URL = f'{API_BASE}/get_training_accuracy_plot'
LOSS_PLOT_URL = f'{API_BASE}/get_training_loss_plot'

st.title("Fruits & Vegetables Classifier Application")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Data Visualization", "Model Retraining", "Model Uptime"])


with tab1:
    st.header("Upload an image for prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"], key="predict_upload")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded image', use_column_width=True)
        if st.button("Predict", key="predict_btn"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            with st.spinner("Sending to API..."):
                try:
                    resp = requests.post(PREDICT_URL, files=files, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Prediction: **{data['label']}** — Confidence: {data['confidence']:.2f}")
                    else:
                        st.error(f"API error: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Error contacting API: {e}")

    st.header("Upload new training images")
    upload_label = st.text_input("Label / class name", value="", key="label_input")
    upload_file = st.file_uploader("Choose an image to add to training set", type=["jpg","jpeg","png"], key="train_upload")
    if st.button("Upload to training set", key="upload_btn"):
        if not upload_label:
            st.warning("Please enter a label")
        elif not upload_file:
            st.warning("Please choose a file")
        else:
            files = {'file': (upload_file.name, upload_file.getvalue(), upload_file.type)}
            data = {'label': upload_label}
            with st.spinner("Uploading to training set..."):
                try:
                    resp = requests.post(UPLOAD_URL, files=files, data=data, timeout=60)
                    if resp.status_code == 200:
                        st.success("Saved to training set: " + resp.json().get('path', ''))
                    else:
                        st.error("Upload failed: " + resp.text)
                except Exception as e:
                    st.error("Error contacting API: " + str(e))


with tab2:
    st.header("Data Visualization")


    try:
        resp = requests.get(METRICS_URL, timeout=10)
        if resp.status_code == 200:
            metrics = resp.json()
            st.subheader("Test Metrics")
            st.text(metrics.get("metrics", "No metrics available."))
        else:
            st.info("Metrics not available yet.")
    except Exception as e:
        st.info("Error fetching metrics: " + str(e))


    try:
        resp = requests.get(ACC_PLOT_URL, timeout=10)
        if resp.status_code == 200:
            acc_img = Image.open(BytesIO(resp.content))
            st.subheader("Training Accuracy")
            st.image(acc_img)
        else:
            st.info("No training accuracy plot found.")
    except Exception as e:
        st.info("Error fetching accuracy plot: " + str(e))

    try:
        resp = requests.get(LOSS_PLOT_URL, timeout=10)
        if resp.status_code == 200:
            loss_img = Image.open(BytesIO(resp.content))
            st.subheader("Training Loss")
            st.image(loss_img)
        else:
            st.info("No training loss plot found.")
    except Exception as e:
        st.info("Error fetching loss plot: " + str(e))

with tab3:
    st.header("Model Retraining")
    if st.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            try:
                resp = requests.post(RETRAIN_URL, timeout=10)
                if resp.status_code == 200:
                    st.success(resp.json().get('message', 'Training started!'))
                else:
                    st.error(f"Retrain failed: {resp.text}")
            except Exception as e:
                st.error(f"Error contacting API: {e}")


with tab4:
    st.header("Model Uptime")
    if st.button("Check Model Uptime"):
        with st.spinner("Checking model uptime..."):
            try:
                resp = requests.get(UPTIME_URL, timeout=10)
                if resp.status_code == 200:
                    status = resp.json()
                    st.success(f"Model Status: {status.get('status','Unknown')}")
                    st.write(status)
                else:
                    st.error(f"Status check failed: {resp.text}")
            except Exception as e:
                st.error(f"Error contacting API: {e}")
