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

    def __init__(self, email: str = "", password_hash: str = "", password_salt: str = "") -> None:
        rng = np.random.default_rng()
        self.seed = int(rng.integers(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max, dtype=np.int64, endpoint=True))
        self.passwords = {}
        self.password_length = 10
        self.email = email
        self.password_hash = password_hash
        self.password_salt = password_salt

    def detect_and_crop(self, img_bgr: np.ndarray) -> np.ndarray:
        results = self.model(img_bgr)[0]

        if len(results.boxes) == 0:
            h, w = img_bgr.shape[:2]
            return img_bgr[h//4:3*h//4, w//4:3*w//4]

        box = results.boxes[0].xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        return img_bgr[y1:y2, x1:x2]

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

    def pipeline(self, img_bgr: np.ndarray) -> torch.Tensor:
        crop = self.detect_and_crop(img_bgr)
        return self.embed(crop)

    def add_password(self, application: str, img_bgr: np.ndarray) -> str:
        if application in self.passwords:
            return "A password already exists for this application."
        embed = self.pipeline(img_bgr)
        self.passwords[application] = embed
        return self.embed_to_ascii(embed, self.password_length)

    def check_password(self, application: str, img_bgr: np.ndarray) -> str:
        embed = self.pipeline(img_bgr)
        if application not in self.passwords:
            return None
        comparison = self.passwords[application]
        if F.cosine_similarity(embed, comparison).item() < 0.5:
            return None
        return self.embed_to_ascii(comparison, self.password_length)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "password_length": self.password_length,
            "email": self.email,
            "password_hash": self.password_hash,
            "password_salt": self.password_salt,
            "passwords": {
                app: emb.cpu().float().numpy().tolist()
                for app, emb in self.passwords.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        user = object.__new__(cls)
        user.seed = data["seed"]
        user.password_length = data.get("password_length", 10)
        user.email = data.get("email", "")
        user.password_hash = data.get("password_hash", "")
        user.password_salt = data.get("password_salt", "")
        user.passwords = {
            app: torch.tensor(emb)
            for app, emb in data["passwords"].items()
        }
        return user
