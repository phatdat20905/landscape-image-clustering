# src/preprocessing/step1_cleaning.py
# ================================================================
#  STEP 1 – DATA CLEANING  (Fixed deduplication)
#  Workflow: Raw Data → [Cleaning] → Integration → Transformation → Encoding
#
#  Kỹ thuật:
#    1. Kiểm tra đọc được (cv2.imdecode)
#    2. Kiểm tra kích thước tối thiểu
#    3. Kiểm tra đơn sắc
#    4. Gaussian Blur (noise reduction)
#    5. Deduplication – ưu tiên theo thứ tự:
#         a) URL exact match  → chính xác nhất, không false positive
#         b) pHash (hash_size=8, threshold=2) → chỉ bắt ảnh gần như
#            giống hệt nhau về pixel, ngưỡng rất chặt
#
#  Tại sao dùng URL làm dedup chính:
#    - pHash thu nhỏ ảnh về (hash_size)² pixels rồi so sánh
#      → hai ảnh khác nhau nhưng cùng tone màu / cùng kích thước
#      gốc có thể cho hash giống nhau (false positive)
#    - URL là định danh duy nhất từ API, không bao giờ false positive
#    - pHash chỉ là lớp phụ để bắt ảnh tải nhiều lần từ URL khác nhau
#      nhưng byte giống hệt nhau
#
#  Input : MinIO raw/images/  +  MongoDB images_raw
#  Output: MongoDB images_clean
#  Chạy: python src/preprocessing/step1_cleaning.py
# ================================================================

import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
from urllib.parse import urlparse
from PIL import Image
import imagehash
from datetime import datetime

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET

# ── Config ──────────────────────────────────────────────────────
MIN_WIDTH       = 200
MIN_HEIGHT      = 200
MONO_STD_THRESH = 8.0    # std grayscale < ngưỡng → đơn sắc

GAUSSIAN_KSIZE  = 3

# pHash config – ngưỡng RẤT CHẶT để tránh false positive
# hash_size=8  → ảnh thu nhỏ về 8×8 = 64 bits
# hash_size=16 → ảnh thu nhỏ về 16×16 = 256 bits (dễ collision hơn)
# threshold=2  → chỉ chấp nhận sai khác tối đa 2/64 bits (~3%)
#               → chỉ bắt được ảnh gần như pixel-perfect giống nhau
PHASH_SIZE      = 8
PHASH_THRESHOLD = 2      # cực kỳ chặt, tránh false positive


# ================================================================
#  HELPERS
# ================================================================
def bytes_to_cv2(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def check_size(img_cv) -> tuple:
    if img_cv is None:
        return False, 0, 0
    h, w = img_cv.shape[:2]
    return (w >= MIN_WIDTH and h >= MIN_HEIGHT), w, h


def check_monotone(img_cv) -> bool:
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) >= MONO_STD_THRESH


def apply_gaussian_blur(img_cv):
    # kept for compatibility; we will NOT use blur for pHash calculation
    return cv2.GaussianBlur(img_cv, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)


def compute_phash(img_cv) -> str:
    """
    Tính pHash với hash_size=8 (64 bits).
    Nhỏ hơn → ít chiều hơn → ngưỡng so sánh phải chặt hơn.
    """
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return str(imagehash.phash(pil, hash_size=PHASH_SIZE))


def normalize_url(url: str) -> str:
    """
    Chuẩn hoá URL để so sánh chính xác:
    - Bỏ query params của Unsplash (?ixid=...) vì cùng ảnh
      có thể được tải với params khác nhau
    - Giữ photo ID (phần path chính)
    """
    # Keep the full URL (including query) for exact matching.
    # We still canonicalize the netloc to lower-case and remove URL fragment.
    if not url:
        return ""
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url)
        scheme = p.scheme or 'http'
        netloc = p.netloc.lower()
        path = p.path.rstrip('/')
        # keep params and query to preserve identifying parts (e.g., Google tbn id)
        query = p.query
        return urlunparse((scheme, netloc, path, p.params, query, ''))
    except Exception:
        # fallback to raw string trimmed
        return url.strip()


def is_url_duplicate(url: str, seen_urls: set) -> bool:
    """
    Kiểm tra URL trùng lặp (sau chuẩn hoá).
    Đây là phương pháp chính xác nhất – không có false positive.
    """
    if not url:
        return False
    norm = normalize_url(url)
    return norm in seen_urls


def get_url_base(url: str) -> str:
    """Return the canonical URL base used for grouping images that come
    from the same resource (scheme + netloc + path, no query or fragment).
    Example: https://images.unsplash.com/photo-123?ixid=... ->
             https://images.unsplash.com/photo-123
    """
    if not url:
        return ""
    try:
        p = urlparse(url)
        scheme = (p.scheme or 'http').lower()
        netloc = p.netloc.lower()
        path = p.path.rstrip('/')
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return url.split('?')[0].split('#')[0].rstrip('/')


def is_phash_duplicate(phash_str: str, seen_hashes: dict) -> tuple:
    """
    Kiểm tra pHash trùng lặp với ngưỡng rất chặt (threshold=2).
    Chỉ bắt ảnh gần như pixel-perfect giống nhau.
    Trả về (is_dup, dup_filename).
    """
    try:
        h = imagehash.hex_to_hash(phash_str)
        for s, fn in seen_hashes.items():
            dist = h - imagehash.hex_to_hash(s)
            if dist <= PHASH_THRESHOLD:
                return True, fn
    except Exception:
        pass
    return False, None


def _insert_reject(col, doc: dict, reason: str, phash: str = ""):
    entry = {
        **{k: v for k, v in doc.items() if k != "_id"},
        "cleaned":       False,
        "reject_reason": reason,
        "phash":         phash,
        "cleaned_at":    datetime.now().strftime("%Y-%m-%d"),
    }
    try:
        col.insert_one(entry)
    except Exception:
        pass


# ================================================================
#  MAIN
# ================================================================
def run_cleaning(minio: MinioClient, mongo: MongoDBClient):
    col_raw   = mongo.get_col("raw")
    col_clean = mongo.get_col("clean")
    mongo.create_indexes(col_clean)

    total_raw = col_raw.count_documents({})
    already   = col_clean.count_documents({})
    print(f"[Cleaning] Raw: {total_raw:,} | Đã xử lý trước: {already:,}")
    print(f"  Dedup strategy: URL (primary) + pHash (secondary, only when URL base matches)")

    # ── Load trạng thái đã clean (để resume) ────────────────────
    done_files: set[str] = {
        d["filename"] for d in
        col_clean.find({}, {"filename": 1, "_id": 0})
    }

    # ── Load seen URLs từ các ảnh đã clean=True ─────────────────
    # Đây là tập dedup chính – chính xác, không false positive
    seen_urls: set[str] = set()
    for d in col_clean.find({"cleaned": True, "url": {"$ne": None}},
                             {"url": 1, "_id": 0}):
        norm = normalize_url(d.get("url", ""))
        if norm:
            seen_urls.add(norm)

    # ── Load seen pHashes grouped by URL base from cleaned images ──
    # We only compare pHash with images that share the same URL base
    # (i.e. same scheme+host+path, different query params are ignored here)
    seen_hashes_by_base: dict[str, dict[str, str]] = {}
    for d in col_clean.find({"cleaned": True, "phash": {"$exists": True, "$ne": ""}, "url": {"$exists": True, "$ne": None}}, {"phash": 1, "filename": 1, "url": 1, "_id": 0}):
        base = get_url_base(d.get("url", ""))
        if not base:
            continue
        seen_hashes_by_base.setdefault(base, {})[d["phash"]] = d["filename"]

    stats = {
        "ok": 0,
        "reject_corrupt":  0,
        "reject_small":    0,
        "reject_monotone": 0,
        "reject_missing_url": 0,
        "reject_url_dup":  0,   # trùng URL (chính xác)
        "reject_hash_dup": 0,   # trùng pHash (backup)
    }

    for i, doc in enumerate(col_raw.find({}, {"_id": 0}), 1):
        fname = doc["filename"]
        if fname in done_files:
            continue

        obj = doc.get("object_name", f"raw/images/{fname}")
        url = doc.get("url", "") or ""

        # ── Nếu URL bị thiếu → từ chối ngay (policy: drop missing-url)
        if not url:
            stats["reject_missing_url"] += 1
            _insert_reject(col_clean, doc, "missing_url")
            continue

        # ── Dedup bước 1: kiểm tra URL TRƯỚC khi tải ảnh ────────
        # Tiết kiệm bandwidth – không cần tải ảnh nếu URL đã thấy
        if is_url_duplicate(url, seen_urls):
            stats["reject_url_dup"] += 1
            _insert_reject(col_clean, doc, f"duplicate_url")
            continue

        # ── Tải ảnh từ MinIO ────────────────────────────────────
        try:
            resp = minio.client.get_object(MINIO_BUCKET, obj)
            raw  = resp.read(); resp.close()
        except Exception as e:
            stats["reject_corrupt"] += 1
            _insert_reject(col_clean, doc, "download_failed")
            continue

        # ── 1. Kiểm tra đọc được ────────────────────────────────
        img_cv = bytes_to_cv2(raw)
        if img_cv is None:
            stats["reject_corrupt"] += 1
            _insert_reject(col_clean, doc, "corrupt_image")
            continue

        # ── 2. Kiểm tra kích thước ──────────────────────────────
        ok, w, h = check_size(img_cv)
        if not ok:
            stats["reject_small"] += 1
            _insert_reject(col_clean, doc, f"too_small_{w}x{h}")
            continue

        # ── 3. Kiểm tra đơn sắc ─────────────────────────────────
        if not check_monotone(img_cv):
            stats["reject_monotone"] += 1
            _insert_reject(col_clean, doc, "monotone")
            continue

        # ── 4. Gaussian Blur (noise reduction) ──────────────────
        img_denoised = apply_gaussian_blur(img_cv)

        # ── 5. Dedup bước 2: pHash nhưng ONLY when URL base matches
        # Compute URL base (scheme+host+path) and only compare pHash with
        # phashes of previously-seen images that have the same base.
        base = get_url_base(url)
        # Always compute pHash on the raw image (no blur). We only compare
        # against previously-seen phashes for the same URL base, but we must
        # also store the phash for the current image so later records in the
        # same run can be detected as duplicates.
        ph = compute_phash(img_cv)
        if base and base in seen_hashes_by_base:
            is_dup, dup_fn = is_phash_duplicate(ph, seen_hashes_by_base.get(base, {}))
            if is_dup:
                stats["reject_hash_dup"] += 1
                _insert_reject(col_clean, doc, f"duplicate_phash_of_{dup_fn}", ph)
                continue

        # ── Ảnh hợp lệ → cập nhật seen sets ─────────────────────
        norm_url = normalize_url(url)
        if norm_url:
            seen_urls.add(norm_url)
        # store phash under the base group so future images (in this run
        # or subsequent runs) will be compared against it
        seen_hashes_by_base.setdefault(base, {})[ph] = fname

        # ── Lưu vào images_clean ────────────────────────────────
        entry = {
            **{k: v for k, v in doc.items() if k != "_id"},
            "phash":         ph,
            "cleaned":       True,
            "reject_reason": None,
            "cleaned_at":    datetime.now().strftime("%Y-%m-%d"),
        }
        try:
            col_clean.insert_one(entry)
            stats["ok"] += 1
        except Exception:
            pass

        if i % 200 == 0:
            total_rej = sum(v for k, v in stats.items() if k.startswith("reject"))
            print(f"  [{i}/{total_raw}] OK={stats['ok']} "
                  f"URLdup={stats['reject_url_dup']} "
                  f"HashDup={stats['reject_hash_dup']} "
                  f"Other={stats['reject_corrupt']+stats['reject_small']+stats['reject_monotone']}",
                  end="\r")

    total_processed = sum(stats.values())
    total_rej = sum(v for k, v in stats.items() if k.startswith("reject"))

    print(f"""
[Cleaning] HOÀN THÀNH
  ┌──────────────────────────────────────────┐
  │  Tổng xử lý    : {total_processed:>6,}                  │
  │  Hợp lệ        : {stats['ok']:>6,}  ✓               │
  │  Lỗi/hỏng      : {stats['reject_corrupt']:>6,}                  │
  │  Kích thước nhỏ: {stats['reject_small']:>6,}                  │
  │  Đơn sắc       : {stats['reject_monotone']:>6,}                  │
    │  Trùng URL     : {stats['reject_url_dup']:>6,}  (primary dedup)  │
    │  Trùng pHash   : {stats['reject_hash_dup']:>6,}  (backup dedup)  │
    │  Missing URL    : {stats['reject_missing_url']:>6,}  (dropped)     │
  └──────────────────────────────────────────┘
  Pass rate: {stats['ok']/total_processed*100:.1f}% (nếu total>0)
""")
    return stats


# ================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 1 – DATA CLEANING  ")
    print("=" * 55)
    minio = MinioClient()
    mongo = MongoDBClient()
    run_cleaning(minio, mongo)