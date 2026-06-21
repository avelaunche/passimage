import base64
import hashlib
import io
import json
import os
import secrets
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from user import User

DATA_DIR = os.environ.get("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace * with your Lovable domain before going live
    allow_methods=["POST"],
    allow_headers=["*"],
)


# --- persistence helpers ---

def user_path(user_id: str) -> str:
    return os.path.join(DATA_DIR, f"{user_id}.json")

def load_user(user_id: str) -> User:
    path = user_path(user_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="User not found")
    with open(path) as f:
        return User.from_dict(json.load(f))

def save_user(user_id: str, user: User) -> None:
    with open(user_path(user_id), "w") as f:
        json.dump(user.to_dict(), f)

def index_path() -> str:
    return os.path.join(DATA_DIR, "_index.json")

def load_index() -> dict:
    path = index_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def save_index(index: dict) -> None:
    with open(index_path(), "w") as f:
        json.dump(index, f)


# --- auth helpers ---

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000).hex()


# --- image decoding ---

def decode_image(base64_image: str) -> np.ndarray:
    """Decode a base64 data URL or raw base64 string into a BGR numpy array."""
    raw = base64_image.split(",", 1)[-1]
    image_bytes = base64.b64decode(raw)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# --- request models ---

class NewUserRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UserRequest(BaseModel):
    user_id: str

class PasswordRequest(BaseModel):
    user_id: str
    application: str
    image: str  # base64 data URL


# --- endpoints ---

@app.post("/new_user")
def new_user(req: NewUserRequest):
    index = load_index()
    if req.email in index:
        raise HTTPException(status_code=409, detail="Email already registered")
    user_id = str(uuid.uuid4())
    salt = secrets.token_hex(16)
    password_hash = hash_password(req.password, salt)
    save_user(user_id, User(email=req.email, password_hash=password_hash, password_salt=salt))
    index[req.email] = user_id
    save_index(index)
    return {"user_id": user_id}

@app.post("/login")
def login(req: LoginRequest):
    index = load_index()
    user_id = index.get(req.email)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user = load_user(user_id)
    expected = hash_password(req.password, user.password_salt)
    if not secrets.compare_digest(expected, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user_id}

@app.post("/logout")
def logout():
    return {"detail": "Logged out"}

@app.post("/get_applications")
def get_applications(req: UserRequest):
    user = load_user(req.user_id)
    return {"applications": list(user.passwords.keys())}

@app.post("/add_password")
def add_password(req: PasswordRequest):
    user = load_user(req.user_id)
    img_bgr = decode_image(req.image)
    password = user.add_password(req.application, img_bgr)
    save_user(req.user_id, user)
    if password.startswith("A password already"):
        raise HTTPException(status_code=409, detail=password)
    return {"password": password}

@app.post("/check_password")
def check_password(req: PasswordRequest):
    user = load_user(req.user_id)
    img_bgr = decode_image(req.image)
    password = user.check_password(req.application, img_bgr)
    if password is None:
        raise HTTPException(status_code=401, detail="Password recovery failed. Please try again.")
    return {"password": password}
