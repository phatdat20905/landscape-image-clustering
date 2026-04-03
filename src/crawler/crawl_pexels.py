# ================================================================
#  Crawler Pexels API → MinIO + MongoDB (images_raw)
#  Optimized: Session reuse, concurrent downloads, URL dedup, threading
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
MAX_WORKERS = 8  # concurrent download/upload workers
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0
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


def crawl_pexels(minio: MinioClient, mongo: MongoDBClient, target=TARGET):

    col = mongo.get_col("raw")
    mongo.create_indexes(col)

    count = col.count_documents({"source": "pexels"})
    print(f"[Pexels] Đã có {count} ảnh | Target: {target}")

    # Tạo session chung (reuse TCP connections)
    session = requests.Session()
    session.headers.update({"Authorization": PEXELS_API_KEY, "User-Agent": "Mozilla/5.0"})
    
    counter_mgr = CounterManager(count)
    total_found = 0
    
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

            if res.status_code == 429:
                print("  ⚠️ Rate limit → sleep 60s")
                time.sleep(60)
                continue

            if res.status_code != 200:
                print(f"  API error: {res.status_code}")
                break

            data = res.json()
            total_results = data.get("total_results", 0)
            if keyword_total == 0 and total_results > 0:
                print(f"  [Info] Total available: {total_results}")
            
            photos = data.get("photos", [])
            if not photos:
                print(f"  [Info] No more photos on page {page}")
                break

            # Concurrent processing của photos
            def process_photo(photo):
                nonlocal count
                
                src = photo.get("src", {})
                # Ưu tiên large thay vì original (nhanh hơn, tiết kiệm bandwidth)
                img_url = src.get("large") or src.get("large2x") or src.get("original")
                
                if not img_url:
                    return None

                # Skip nếu URL đã tồn tại trong DB (tránh trùng lặp)
                if col.find_one({"url": img_url, "source": "pexels"}):
                    return None

                img_bytes = download_image_with_retry(session, img_url)
                if not img_bytes:
                    return None

                is_valid, w, h = is_valid_image(img_bytes)
                if not is_valid:
                    return None

                
                filename = f"pexels_{count:06d}.jpg"
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
                    "source": "pexels",
                    "url": img_url,
                    "description": photo.get("alt") or keyword,
                    "keyword": keyword,
                    "width": photo.get("width", 0),
                    "height": photo.get("height", 0),
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

    print(f"\n✅ DONE PEXELS: {count} images")


if __name__ == "__main__":
    print("=" * 50)
    print("  CRAWL PEXELS (Optimized)")
    print("=" * 50)

    minio = MinioClient()
    mongo = MongoDBClient()

    crawl_pexels(minio, mongo)