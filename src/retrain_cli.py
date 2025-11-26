# src/retrain_cli.py
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import create_generators
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else '.'
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'fruits_veggies_model.keras')

os.makedirs(MODELS_DIR, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # generators
    train_gen, val_gen, _ = create_generators(TRAIN_DIR, VAL_DIR, VAL_DIR)
    num_classes = len(train_gen.class_indices)

    # load or create model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print('Loaded existing model for fine-tuning')
    else:
        from model import create_model
        model = create_model(num_classes)
        print('Created new model')

    # callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ckpt_path = MODEL_PATH + '.ckpt'
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

    # train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[early_stop, checkpoint]
    )