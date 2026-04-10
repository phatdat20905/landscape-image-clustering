# configs/config.py
import os
from dotenv import load_dotenv

# load file .env
load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────
PEXELS_API_KEY   = "C6uJCFQRDQQM5J7G2URmvGC7tDY8uiS7qaopBl9trAOesYd4RgBNHa0T"
UNSPLASH_KEY   = os.getenv("UNSPLASH_KEY")  

# ── MinIO ─────────────────────────────────────────────────────────
MINIO_ENDPOINT   = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET     = "landscape-data"
MINIO_SECURE     = False

# ── MongoDB ──────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB  = os.getenv("MONGO_DB")

MONGO_COLLECTIONS = {
    "raw": "images_raw",
    "clean": "images_clean",
    "integrated": "images_integrated",
    "transformed": "images_transformed",
    "features": "image_features",
    "clusters": "clusters"
}

# ── Crawler chung ─────────────────────────────────────────────────
KEYWORDS = [
    "mountain", "forest", "sea", "desert", "snow"
]

MIN_SIZE = 200      # pixel chiều nhỏ nhất
DELAY    = 1.0      # giây nghỉ giữa request
