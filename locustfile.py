from locust import HttpUser, task, between
import os
import random


PREDICT_IMAGES_DIR = "data/test"

TRAIN_IMAGES_DIR = "data/train"


TRAIN_LABELS = ["apple", "banana", "carrot", "tomato"]

class MLUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0.5, 2)

    def get_random_image(self, base_dir):
        """Return a random image path from the folder and subfolders"""
        all_images = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_images.append(os.path.join(root, f))
        if not all_images:
            return None
        return random.choice(all_images)

    @task(3)  
    def predict(self):
        img_path = self.get_random_image(PREDICT_IMAGES_DIR)
        if not img_path:
            return

        filename = os.path.basename(img_path)
        with open(img_path, "rb") as f:
            files = {"file": (filename, f, "image/jpg")}
            self.client.post("/predict", files=files, timeout=30)

    @task(1) 
    def upload_training_image(self):
        img_path = self.get_random_image(TRAIN_IMAGES_DIR)
        if not img_path:
            return

        label = random.choice(TRAIN_LABELS)
        filename = os.path.basename(img_path)
        with open(img_path, "rb") as f:
            files = {"file": (filename, f, "image/jpg")}
            data = {"label": label}
            self.client.post("/upload_training_image", files=files, data=data, timeout=60)
