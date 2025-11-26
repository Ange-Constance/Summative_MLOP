from preprocessing import create_generators
from model import create_model
import os
import tensorflow as tf


def retrain_model(train_dir, val_dir, model_path, epochs=5):
    # Load model
    model = tf.keras.models.load_model(model_path)


    # Create generators
    train_gen, val_gen, _ = create_generators(train_dir, val_dir, val_dir)


    # Fine-tune
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)


    # Save updated model
    model.save(model_path)
    print('Model retrained and saved.')