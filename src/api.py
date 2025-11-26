from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import io
from PIL import Image
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'fruits_veggies_model.keras')
CLASS_JSON = os.path.join(BASE_DIR, '..', 'models', 'class_indices.json')

app = FastAPI()
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_JSON, 'r') as f:
    class_indices = json.load(f)

start_time = time.time()

@app.get('/uptime')
def uptime():
    return {'uptime_seconds': time.time() - start_time}

def preprocess_pil(img: Image.Image, target_size=(224,224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    x = preprocess_pil(img)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    inv_map = {v:k for k,v in class_indices.items()}
    label = inv_map[idx]
    confidence = float(np.max(preds))
    return JSONResponse({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
