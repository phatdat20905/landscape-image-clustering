# ================================================================
#  Crawler Pexels API → MinIO + MongoDB (images_raw)
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
    PEXELS_API_KEY,
    KEYWORDS,
    MIN_SIZE,
    DELAY,
    MINIO_BUCKET
)

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient

# ================================================================
BASE_URL = "https://api.pexels.com/v1/search"
HEADERS  = {"Authorization": PEXELS_API_KEY}
PER_PAGE = 80
TARGET   = 5000
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


def crawl_pexels(minio: MinioClient, mongo: MongoDBClient, target=TARGET):

    col = mongo.get_col("raw")
    mongo.create_indexes(col)

    count = col.count_documents({"source": "pexels"})
    print(f"[Pexels] Đã có {count} ảnh | Target: {target}")

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

            if res.status_code == 429:
                print("⚠️ Rate limit → sleep 60s")
                time.sleep(60)
                continue

            if res.status_code != 200:
                print("API error:", res.status_code)
                break

            photos = res.json().get("photos", [])
            if not photos:
                break

            for photo in photos:
                if count >= target:
                    break

                src = photo.get("src", {})
                img_url = (
                    src.get("original")
                    or src.get("large2x")
                    or src.get("large")
                )

                if not img_url:
                    continue

                try:
                    img_res = requests.get(img_url, timeout=20)
                    img_bytes = img_res.content
                except:
                    continue

                if not is_valid_image(img_bytes):
                    continue

                filename    = f"pexels_{count:06d}.jpg"
                object_name = f"raw/images/{filename}"

                ok = minio.put_object(
                    MINIO_BUCKET,
                    object_name,
                    data=BytesIO(img_bytes),
                    length=len(img_bytes),
                    content_type="image/jpeg"
                )

                if not ok:
                    continue

                metadata = {
                    "filename": filename,
                    "object_name": object_name,
                    "source": "pexels",
                    "url": src.get("original") or src.get("large2x"),
                    "description": photo.get("alt") or keyword,
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

    print(f"\n✅ DONE PEXELS: {count} images")


if __name__ == "__main__":
    print("=" * 50)
    print("  CRAWL PEXELS")
    print("=" * 50)

    minio = MinioClient()
    mongo = MongoDBClient()

    crawl_pexels(minio, mongo)