import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_image(model, img_path, class_indices):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    class_label = {v:k for k,v in class_indices.items()}[class_idx]
    confidence = preds[0][class_idx]
    return class_label, confidence