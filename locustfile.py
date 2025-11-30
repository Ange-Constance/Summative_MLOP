from locust import HttpUser, task, between
import base64

class MLUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predict(self):
        with open('sample_images/apple_01.jpg', 'rb') as f:
            files = {'file': ('apple_01.jpg', f, 'image/jpeg')}
            self.client.post('/predict', files=files, timeout=30)
