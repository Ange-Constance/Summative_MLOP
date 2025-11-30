# src/api_upload.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
from PIL import Image
import sqlite3
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else '.'
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(BASE_DIR, 'training_images.db')

MODEL_PATH = os.path.join(MODELS_DIR, 'fruits_veggies_model.keras')
CLASS_JSON = os.path.join(MODELS_DIR, 'class_indices.json')

# ------------------ FASTAPI ------------------
app = FastAPI(title="Fruits & Veggies API")

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8501",   # Streamlit
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------
# Load Model at Startup
# -----------------------------------------------
model = None
class_indices = {}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON, 'r') as f:
        class_indices = json.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print("Model NOT loaded:", e)


# -----------------------------------------------
# Initialize SQLite DB
# -----------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label TEXT,
            saved_path TEXT,
            uploaded_at REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ==========================================================
# -------------   UPLOAD TRAINING IMAGE  --------------------
# ==========================================================
@app.post("/upload_training_image")
async def upload_training_image(file: UploadFile = File(...), label: str = Form(...)):
    label = label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label required")

    label_dir = os.path.join(TRAIN_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    contents = await file.read()

    # Validate image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Save image
    timestamp = int(time.time() * 1000)
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    save_name = f"{label}_{timestamp}{ext}"
    save_path = os.path.join(label_dir, save_name)

    with open(save_path, "wb") as f:
        f.write(contents)

    # Log into DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO uploads (filename, label, saved_path, uploaded_at) VALUES (?,?,?,?)",
        (save_name, label, save_path, time.time())
    )
    conn.commit()
    conn.close()

    return {"status": "saved", "path": save_path}


# ==========================================================
# ---------------------  PREDICT  ---------------------------
# ==========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])

    inv_map = {v: k for k, v in class_indices.items()}
    label = inv_map.get(idx, "unknown")

    return {
        "label": label,
        "confidence": float(np.max(preds))
    }


# ==========================================================
# ---------------------  RETRAIN  ---------------------------
# ==========================================================
def retrain_model():
    global model, class_indices

    # Build data generators
    datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        subset="validation"
    )

    # Update class indices
    class_indices = train_gen.class_indices

    # Simple CNN model (you already used this in your training file)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPool2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(train_gen.class_indices), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5
    )

    # Save model
    model.save(MODEL_PATH)

    # Save class map
    with open(CLASS_JSON, "w") as f:
        json.dump(class_indices, f)

    return {
        "train_acc": float(history.history["accuracy"][-1]),
        "train_loss": float(history.history["loss"][-1]),
        "val_acc": float(history.history["val_accuracy"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
    }


@app.post("/retrain")
async def retrain():
    try:
        metrics = retrain_model()
        return {
            "status": "Model retrained successfully",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# ---------------------  MAIN  ------------------------------
# ==========================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
