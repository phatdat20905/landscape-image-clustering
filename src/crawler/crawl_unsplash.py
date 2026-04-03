# ================================================================
#  Crawler Unsplash API → MinIO + MongoDB (images_raw)
# ================================================================

import requests
import time
import uuid
import threading
from datetime import datetime, timezone
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.config import (
    UNSPLASH_KEY,
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
MAX_WORKERS = 8  # concurrent download/upload workers
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

KEYWORDS = [
    "forest", "sea", "desert", "snow"
]
# ================================================================


# ================================================================
# Validate ảnh (clean sơ bộ)
# ================================================================
def is_valid_image(img_bytes: bytes) -> tuple:
    """Validate ảnh, trả về (is_valid, width, height)"""
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False, 0, 0
        h, w = img.shape[:2]
        max_dim = max(w, h)
        is_ok = max_dim >= MIN_SIZE
        return is_ok, w, h
    except:
        return False, 0, 0


def download_image_with_retry(session: requests.Session, url: str, retries=RETRY_ATTEMPTS) -> bytes:
    """Download ảnh từ URL với retry + exponential backoff"""
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff)
                backoff *= RETRY_BACKOFF
    return b""


class CounterManager:
    """Thread-safe counter cho filename"""
    def __init__(self, initial=0):
        self.count = initial
        self.lock = threading.Lock()
    
    def get_next(self):
        with self.lock:
            self.count += 1
            return self.count


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

    # Tạo session chung (reuse TCP connections)
    session = requests.Session()
    session.headers.update(HEADERS)
    
    counter_mgr = CounterManager(count)

    for keyword in KEYWORDS:
        if count >= target:
            break

        print(f"\n[Keyword] {keyword}")
        page = 1
        keyword_total = 0

        while count < target:
            params = {
                "query": keyword,
                "page": page,
                "per_page": PER_PAGE,
                "orientation": "landscape"
            }

            try:
                res = session.get(BASE_URL, params=params, timeout=15)
            except Exception as e:
                print(f"  Request error: {e}")
                break

            if res.status_code == 401:
                print("  ❌ API key sai")
                return

            if res.status_code == 403:
                print("  ⚠️ Rate limit → sleep 60s")
                time.sleep(60)
                continue

            if res.status_code != 200:
                print(f"  API error: {res.status_code}")
                break

            data = res.json()
            total_results = data.get("total", 0)
            if keyword_total == 0 and total_results > 0:
                print(f"  [Info] Total available: {total_results}")
            
            photos = data.get("results", [])
            if not photos:
                print(f"  [Info] No more photos on page {page}")
                break

            # Concurrent processing của photos
            def process_photo(photo):
                nonlocal count
                
                img_url = (
                    photo["urls"].get("raw")
                    or photo["urls"].get("full")
                    or photo["urls"].get("regular")
                )

                if not img_url:
                    return None

                # Skip nếu URL đã tồn tại trong DB (tránh trùng lặp)
                if col.find_one({"url": img_url, "source": "unsplash"}):
                    return None

                img_bytes = download_image_with_retry(session, img_url)
                if not img_bytes:
                    return None

                is_valid, w, h = is_valid_image(img_bytes)
                if not is_valid:
                    return None

                # Reserve a thread-safe id for the filename
                next_id = counter_mgr.get_next()
                filename = f"unsplash_{next_id:06d}.jpg"
                object_name = f"raw/images/{filename}"

                ok = minio.put_object(
                    MINIO_BUCKET,
                    object_name,
                    data=BytesIO(img_bytes),
                    length=len(img_bytes),
                    content_type="image/jpeg"
                )

                if not ok:
                    return None

                metadata = {
                    "filename": filename,
                    "object_name": object_name,
                    "source": "unsplash",
                    "url": img_url,
                    "description": photo.get("description") or keyword,
                    "keyword": keyword,
                    "width": int(w),
                    "height": int(h),
                    "crawled_at": datetime.now(timezone.utc).isoformat()
                }

                try:
                    col.insert_one(metadata)
                    return 1
                except:
                    return None

            # Chạy concurrent với ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_photo, p) for p in photos]
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        if res:
                            count += 1
                            keyword_total += 1
                            if count % 50 == 0:
                                print(f"  → {count}/{target} (page {page})")
                    except Exception as e:
                        pass

            if count >= target:
                break

            page += 1
            time.sleep(DELAY)

        print(f"  [Summary] {keyword}: +{keyword_total} images (total: {count}/{target})")

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