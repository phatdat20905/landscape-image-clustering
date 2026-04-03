# configs/config.py
# ================================================================
#  Cấu hình tập trung toàn dự án
# ================================================================

# ── API Keys ─────────────────────────────────────────────────────
PEXELS_API_KEY   = "C6uJCFQRDQQM5J7G2URmvGC7tDY8uiS7qaopBl9trAOesYd4RgBNHa0T"
UNSPLASH_KEY     = "ym_KSqtZZP7qdzyWH3tngJNGJeXICCEm8hH6mug4ZJY"  

# ── MinIO ─────────────────────────────────────────────────────────
MINIO_ENDPOINT   = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET     = "landscape-data"
MINIO_SECURE     = False

# ── MongoDB ───────────────────────────────────────────────────────
MONGO_URI = "mongodb+srv://phatdat:CB6Y08iZtj6YSynu@cluster0.bkalcm4.mongodb.net/landscape_db?retryWrites=true&w=majority"
MONGO_DB  = "landscape_db"
MONGO_COLLECTIONS = {
    "raw": "images_raw",
    "clean": "images_clean",
    "features": "image_features",
    "clusters": "clusters"
}

# ── Crawler chung ─────────────────────────────────────────────────
KEYWORDS = [
    "mountain", "forest", "sea", "desert", "snow"
]

MIN_SIZE = 200      # pixel chiều nhỏ nhất
DELAY    = 1.0      # giây nghỉ giữa request
