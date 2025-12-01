# Fruits & Veggies Classification API

## Deployment Links

- **Video Demo:** (https://www.bugufi.link/qvLhEs)
- **API (Render):** [https://api-zwdd.onrender.com/docs](https://api-zwdd.onrender.com/docs)
- **Streamlit App (UI):** [https://summative-mlop-4xmz.onrender.com/](https://summative-mlop-4xmz.onrender.com/)

---

## Project Objective

Build an end-to-end machine learning pipeline to classify images of fruits and vegetables, with:

- A FastAPI backend for predictions and image uploads
- A Streamlit frontend for user interaction
- Automated model retraining and evaluation
- SQLite database for upload tracking

---

## Dataset Used

- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition/code/data) (stored in the `data/` folder)
- **Classes:** 36 fruits and vegetables (see `data/train/` for full list)
- **Structure:**
  - `data/train/<class>/` — Training images
  - `data/validation/<class>/` — Validation images
  - `data/test/<class>/` — Test images

---

## Features

- Upload new training images via API
- Predict fruit/vegetable from image
- Track uploads in SQLite database
- Model retraining script
- Streamlit UI for easy use
- Dockerized for easy deployment

---

## Project Structure

```
Summative_MLOP/
├── app/                  # Streamlit frontend
│   └── app.py
├── data/                 # Dataset (train/validation/test)
├── models/               # Saved models and class indices
├── notebook/             # Jupyter notebooks for EDA & prototyping
├── src/                  # Core backend code
│   ├── api_upload.py     # FastAPI app (image upload & predict)
│   ├── model.py          # Model architecture
│   ├── prediction.py     # Prediction logic
│   ├── preprocessing.py  # Preprocessing utilities
│   └── retrain.py        # Retraining script
├── Dockerfile            # Docker build for API
├── docker-compose.yml    # Multi-service orchestration
├── requirements.txt      # Python dependencies
└── README.MD             # Project documentation
```

---

##  Approach Used

- **Image Preprocessing:** Resize, normalize, augment
- **Model:** TensorFlow/Keras CNN
- **Evaluation:** Accuracy, confusion matrix, per-class metrics
- **Retraining:** Script to add new data and retrain model
- **API:** FastAPI for upload/predict endpoints
- **Frontend:** Streamlit for user-friendly interface
- **Database:** SQLite for upload tracking

---

##  Project Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd Summative_MLOP
   ```
2. **Install Python 3.11+** (recommended)
3. **Create and activate a virtual environment:**
   ```sh
   python3.11 -m venv streamlit-env
   source streamlit-env/bin/activate
   ```
4. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
5. **(Optional) Build with Docker:**
   ```sh
   docker-compose up --build
   ```

---

##  How to Run the App

- **API (FastAPI):**

  ```sh
  uvicorn src.api_upload:app --host 0.0.0.0 --port 8000
  ```

  Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

- **Streamlit Frontend:**
  ```sh
  streamlit run app/app.py
  ```
  Visit [http://localhost:8501](http://localhost:8501)

---

##  Usage

- **Upload Training Image:**
  - Endpoint: `POST /upload_training_image`
  - Params: `file` (image), `label` (class name)
- **Predict:**
  - Endpoint: `POST /predict`
  - Params: `file` (image)
- **Retrain Model:**
  - Run: `python src/retrain.py`

---

## Database Implementation

- **SQLite** database (`training_images.db`) tracks all uploaded images:
  - `id`, `filename`, `label`, `saved_path`, `uploaded_at`
- Table auto-created on API startup

---

##  Model Evaluation

- **Metrics:** Accuracy, confusion matrix, per-class accuracy
- **Evaluation scripts:** See `notebook/fruits-veggies-pipeline.ipynb`
- **Retrain:** Add new images, run retrain script, evaluate, and update model

---

##  Deployment

- **Docker:**
  - Build and run all services: `docker-compose up --build`
- **Render:** API deployed at [https://api-zwdd.onrender.com](https://api-zwdd.onrender.com)

---

##  API Endpoints

- `POST /upload_training_image` — Upload new training image
- `POST /predict` — Predict fruit/vegetable from image

---


##  Author

Ange Constance Nimuhire
