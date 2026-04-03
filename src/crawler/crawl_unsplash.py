# ================================================================
#  Crawler Unsplash API → MinIO + MongoDB (images_raw)
# ================================================================

import requests
import time
from datetime import datetime
from io import BytesIO
import numpy as np
import cv2
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.config import (
    UNSPLASH_KEY,
    KEYWORDS,
    MIN_SIZE,
    DELAY,
    MINIO_BUCKET
)

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient

# ================================================================
BASE_URL = "https://api.unsplash.com/search/photos"
HEADERS  = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}
PER_PAGE = 30
TARGET   = 5000
# ================================================================


# ================================================================
# Validate ảnh (clean sơ bộ)
# ================================================================
def is_valid_image(img_bytes: bytes) -> bool:
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        return h >= MIN_SIZE and w >= MIN_SIZE
    except:
        return False


# ================================================================
# MAIN CRAWL
# ================================================================
def crawl_unsplash(minio: MinioClient, mongo: MongoDBClient, target=TARGET):

    # 👉 lấy collection RAW
    col = mongo.get_col("raw")

    # 👉 tạo index (chỉ chạy 1 lần)
    mongo.create_indexes(col)

    # 👉 resume theo source
    count = col.count_documents({"source": "unsplash"})
    print(f"[Unsplash] Đã có {count} ảnh | Target: {target}")

    for keyword in KEYWORDS:
        if count >= target:
            break

        print(f"\n[Keyword] {keyword}")
        page = 1

        while count < target:
            params = {
                "query": keyword,
                "page": page,
                "per_page": PER_PAGE,
                "orientation": "landscape"
            }

            try:
                res = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
            except Exception as e:
                print("Request error:", e)
                break

            if res.status_code == 401:
                print("❌ API key sai")
                return

            if res.status_code == 403:
                print("⚠️ Rate limit → sleep 60s")
                time.sleep(60)
                continue

            if res.status_code != 200:
                print("API error:", res.status_code)
                break

            photos = res.json().get("results", [])
            if not photos:
                break

            for photo in photos:
                if count >= target:
                    break

                img_url = (
                    photo["urls"].get("raw")
                    or photo["urls"].get("full")
                    or photo["urls"].get("regular")
                )

                if not img_url:
                    continue

                # download
                try:
                    img_res = requests.get(img_url, timeout=20)
                    img_bytes = img_res.content
                except:
                    continue

                # clean sơ bộ
                if not is_valid_image(img_bytes):
                    continue

                filename    = f"unsplash_{count:06d}.jpg"
                object_name = f"raw/images/{filename}"

                # upload MinIO
                ok = minio.put_object(
                    MINIO_BUCKET,
                    object_name,
                    data=BytesIO(img_bytes),
                    length=len(img_bytes),
                    content_type="image/jpeg"
                )

                if not ok:
                    continue

                # metadata
                metadata = {
                    "filename": filename,
                    "object_name": object_name,
                    "source": "unsplash",
                    "url": photo["urls"].get("full"),
                    "description": photo.get("description") or keyword,
                    "keyword": keyword,
                    "width": photo.get("width", 0),
                    "height": photo.get("height", 0),
                    "crawled_at": datetime.utcnow().strftime("%Y-%m-%d")
                }

                try:
                    col.insert_one(metadata)
                except:
                    continue

                count += 1
                print(f"  → {count}/{target}", end="\r")

            page += 1
            time.sleep(DELAY)

    print(f"\n✅ DONE UNSPLASH: {count} images")


# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  CRAWL UNSPLASH")
    print("=" * 50)

    minio = MinioClient()
    mongo = MongoDBClient()

    crawl_unsplash(minio, mongo, target=TARGET)