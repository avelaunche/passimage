import sys
import hashlib
import string
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel

class User:

    model = YOLO("yolov8n.pt")

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base")
    dino.eval()

    def __init__(self) -> None:
        rng = np.random.default_rng()
        self.seed = int(rng.integers(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max, dtype=np.int64, endpoint=True))
        self.passwords = {}
        self.password_length = 10

    def detect_and_crop(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        results = self.model(img)[0]

        if len(results.boxes) == 0:
            h, w = img.shape[:2]
            return img[h//4:3*h//4, w//4:3*w//4]

        box = results.boxes[0].xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        return img[y1:y2, x1:x2]

    def embed(self, image_bgr: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.dino(**inputs)

        emb = outputs.last_hidden_state[:, 0]
        emb = emb / emb.norm()
        return emb

    def embed_to_ascii(self, emb: torch.Tensor, length: int) -> str:
        emb_bytes = emb.cpu().float().numpy().tobytes()
        seed_bytes = self.seed.to_bytes(8, byteorder="little", signed=True)
        digest = hashlib.sha512(seed_bytes + emb_bytes).digest()
        rng_seed = int(np.frombuffer(digest[:8], dtype=np.uint64)[0])
        rng = np.random.default_rng(rng_seed)
        chars = string.ascii_letters + string.digits + string.punctuation
        indices = rng.integers(0, len(chars), size=length)
        return "".join(chars[i] for i in indices)

    def pipeline(self, path: str) -> torch.Tensor:
        crop = self.detect_and_crop(path)
        return self.embed(crop)

    def add_password(self, application: str, path: str) -> str:
        if application in self.passwords:
            return "A password already exists for this application."
        else:
            embed = self.pipeline(path)
            self.passwords[application] = embed
            return "Your new " + application + " password is " + self.embed_to_ascii(embed, self.password_length)

    def check_password(self, application: str, path: str) -> str:
        embed = self.pipeline(path)
        comparison = self.passwords[application]
        if F.cosine_similarity(embed, comparison).item() < 0.5:
            return "Password recovery failed. Please try again."
        else:
            return "Your " + application + " password is " + self.embed_to_ascii(comparison, self.password_length)