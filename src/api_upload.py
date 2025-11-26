from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os, io, time, json, sqlite3, datetime, threading
from PIL import Image
import numpy as np
import tensorflow as tf


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else '.'
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

DB_PATH = os.path.join(BASE_DIR, 'training_images.db')
MODEL_PATH = os.path.join(MODELS_DIR, 'classifier_model.keras')
CLASS_JSON = os.path.join(MODELS_DIR, 'class_indices.json')
METRICS_PATH = os.path.join(MODELS_DIR, "test_metrics.txt")
ACC_PLOT = os.path.join(MODELS_DIR, "training_accuracy.png")
LOSS_PLOT = os.path.join(MODELS_DIR, "training_loss.png")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)


app = FastAPI(title="Fruits & Vegetables API")
origins = ["http://localhost:8501", "https://api-zwdd.onrender.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


model = None
class_indices = {}
is_training = False
server_start_time = datetime.datetime.now()

def load_model_files():
    global model, class_indices
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_JSON, "r") as f:
            class_indices = json.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print("Model not loaded:", e)

load_model_files()


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


@app.post('/upload_training_image')
async def upload_training_image(file: UploadFile = File(...), label: str = Form(...)):
    label = label.strip()
    if not label:
        raise HTTPException(status_code=400, detail='Label required')

    label_dir = os.path.join(TRAIN_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image file')

    timestamp = int(time.time() * 1000)
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    save_name = f'{label}_{timestamp}{ext}'
    save_path = os.path.join(label_dir, save_name)
    with open(save_path, 'wb') as f:
        f.write(contents)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'INSERT INTO uploads (filename, label, saved_path, uploaded_at) VALUES (?,?,?,?)',
        (save_name, label, save_path, time.time())
    )
    conn.commit()
    conn.close()

    return JSONResponse({'status': 'saved', 'path': save_path})


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    inv_map = {v:k for k,v in class_indices.items()}
    label = inv_map.get(idx, 'unknown')

    return {'label': label, 'confidence': float(np.max(preds))}


def train_model_background():
    global is_training
    is_training = True
    print("Background retraining started...")

    try:
        datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=32,
                                                class_mode='categorical', subset='training')
        val_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=32,
                                              class_mode='categorical', subset='validation')
        num_classes = len(train_gen.class_indices)

        with open(CLASS_JSON, 'w') as f:
            json.dump(train_gen.class_indices, f)

        cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224,224,3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = cnn.fit(train_gen, validation_data=val_gen, epochs=5)

        cnn.save(MODEL_PATH)

        with open(METRICS_PATH, 'w') as f:
            f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")

        # Accuracy plot
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Training Accuracy')
        plt.legend()
        plt.savefig(ACC_PLOT)
        plt.close()

        # Loss plot
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(LOSS_PLOT)
        plt.close()

        load_model_files()
        print("Retraining finished!")

    except Exception as e:
        print("Training error:", e)
    finally:
        is_training = False


@app.post('/retrain_model')
async def retrain_model():
    global is_training
    if is_training:
        return {"status":"training", "message":"Training already running"}
    thread = threading.Thread(target=train_model_background)
    thread.start()
    return {"status":"started", "message":"Model retraining started in background."}


@app.get('/model_uptime')
async def model_uptime():
    uptime = (datetime.datetime.now() - server_start_time).total_seconds()
    return {'status':'loaded' if model else 'not loaded', 'uptime_seconds':uptime, 'is_training':is_training, 'model_path':MODEL_PATH}


@app.get("/get_training_metrics")
async def get_training_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = f.read()
        return {"metrics": metrics}
    return {"metrics": None}

@app.get("/get_training_accuracy_plot")
async def get_training_accuracy_plot():
    if os.path.exists(ACC_PLOT):
        return FileResponse(ACC_PLOT, media_type="image/png")
    return {"error": "Accuracy plot not found"}

@app.get("/get_training_loss_plot")
async def get_training_loss_plot():
    if os.path.exists(LOSS_PLOT):
        return FileResponse(LOSS_PLOT, media_type="image/png")
    return {"error": "Loss plot not found"}


if __name__ == "__main__":
    uvicorn.run("api_upload:app", host="0.0.0.0", port=8000, reload=True)
