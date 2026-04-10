# src/preprocessing/step2_integration.py
# ================================================================
#  STEP 2 – DATA INTEGRATION
#  Workflow: Cleaning → [Integration] → Transformation → Encoding
#
#  Mục tiêu:
#    - Kết hợp dữ liệu từ 3 nguồn thành tập thống nhất
#    - Chuẩn hoá schema metadata (source, keyword, description)
#    - Kiểm tra file tồn tại trên MinIO raw
#    - Tạo metadata chuẩn (schema unified) từ các nguồn khác nhau
#
#  Lưu ý quan trọng:
#    Step 2 CHỈ cập nhật MongoDB, KHÔNG upload ảnh lên MinIO.
#    Ảnh vẫn lấy từ raw/images/ ở Step 3.
#    Step 2 chỉ chuẩn bị metadata (schema unified, không lưu đường dẫn MinIO).
#
#  Input : MongoDB images_clean (cleaned=True)
#          MinIO raw/images/ (kiểm tra tồn tại)
#  Output: MongoDB images_integrated:
#            _id (ref to images_clean._id hoặc filename)
#            source, keyword, description, width_raw, height_raw
#            source_url, source_domain (metadata chuẩn)
#            integrated=True, integrated_at
#
#  Chạy: python src/preprocessing/step2_integration.py
# ================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime
from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET
from pymongo import ASCENDING

# ── Config ──────────────────────────────────────────────────────
# Không cần PREPROC_PREFIX – Step 3 sẽ tính toán path upload

# ================================================================
#  SCHEMA CHUẨN HÓA
# ================================================================
SCHEMA_FIELDS = [
    "filename", "object_name", "label", "width", "height",
    "cleaned", "reject_reason", "cleaned_at",
]

def normalize_meta(doc: dict) -> dict:
    """Chuẩn hoá metadata về schema thống nhất từ 3 nguồn."""
    out = {f: doc.get(f) for f in SCHEMA_FIELDS}

    # Normalize label: prefer explicit `label`, fall back to `keyword` if present
    label = (doc.get("label") or doc.get("keyword") or "unknown")
    out["label"] = (label or "unknown").lower().strip()

    # Width/height: accept either width/height or width_raw/height_raw from cleaned doc
    def to_int(v):
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    out["width"] = to_int(doc.get("width") or doc.get("width_raw") or doc.get("w") )
    out["height"] = to_int(doc.get("height") or doc.get("height_raw") or doc.get("h") )

    return out


# ================================================================
#  MAIN
# ================================================================
def run_integration(minio: MinioClient, mongo: MongoDBClient):
    col_clean       = mongo.get_col("clean")
    col_integrated  = mongo.get_col("integrated")
    # Ensure deterministic upserts: unique index on filename
    try:
        col_integrated.create_index([("filename", ASCENDING)], unique=True)
    except Exception:
        # ignore index creation errors (index may already exist)
        pass
    
    query     = {"cleaned": True, "integrated": {"$ne": True}}
    total     = col_clean.count_documents(query)
    print(f"[Integration] Ảnh cần integrate: {total:,}")
    print(f"  Action: chuẩn hoá metadata → lưu vào images_integrated")
    print(f"  Upload ảnh: KHÔNG (vẫn trong raw/, step3 sẽ xử lý)")

    # ── Lấy danh sách files tồn tại trên MinIO raw ──────────────
    print(f"  Đang quét MinIO raw/images/ ...")
    raw_objects = set(minio.list_objects(prefix="raw/images/"))
    print(f"  Objects trong raw/images/: {len(raw_objects):,}")

    stats = {"ok": 0, "no_file": 0}

    for i, doc in enumerate(col_clean.find(query, {"_id": 0}), 1):
        fname   = doc.get("filename", "")
        obj     = doc.get("object_name", f"raw/images/{fname}")

        # ── Kiểm tra file tồn tại trên MinIO raw ────────────────
        if obj not in raw_objects and f"raw/images/{fname}" not in raw_objects:
            stats["no_file"] += 1
            continue

        # ── Chuẩn hoá metadata ──────────────────────────────────
        unified = normalize_meta(doc)

        # ── Ghi vào collection images_integrated (trimmed schema) ───
        # Keep only the minimal fields needed for downstream steps.
        trimmed_doc = {
            "filename":      unified.get("filename", fname),
            "object_name":   unified.get("object_name", obj),
            "label":         unified.get("label", "unknown"),
            "width":         unified.get("width", 0),
            "height":        unified.get("height", 0),
            "integrated":    True,
            "integrated_at": datetime.now().strftime("%Y-%m-%d"),
        }

        try:
            col_integrated.update_one({"filename": fname}, {"$set": trimmed_doc}, upsert=True)
            stats["ok"] += 1
        except Exception as e:
            print(f"  [!] Lỗi lưu integrated cho {fname}: {e}")
            stats["error"] = stats.get("error", 0) + 1

        if i % 500 == 0:
            print(f"  {i}/{total} | OK={stats['ok']}", end="\r")

    print(f"""
[Integration] HOÀN THÀNH
  OK (metadata chuẩn hoá) : {stats['ok']:,}
  Không có file MinIO     : {stats['no_file']:,}

  Dữ liệu chuẩn hoá lưu tại: MongoDB images_integrated
  Ảnh gốc vẫn trong: raw/images/
  Step 3 sẽ xử lý.
""")

    # Print counts per label (helpful summary)
    labels = col_integrated.distinct("label")
    for lbl in sorted([l for l in labels if l]):
        cnt = col_integrated.count_documents({"label": lbl, "integrated": True})
        print(f"    label={lbl}  ←  {cnt:,} ảnh")

    print()
    return stats


# ================================================================
if __name__ == "__main__":
    print("=" * 58)
    print("  STEP 2 – DATA INTEGRATION")
    print("  (Metadata only – no MinIO upload)")
    print("=" * 58)
    minio = MinioClient()
    mongo = MongoDBClient()
    run_integration(minio, mongo)