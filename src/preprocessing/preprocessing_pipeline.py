# src/preprocessing/preprocessing_pipeline.py
# ================================================================
#  PREPROCESSING PIPELINE – Orchestrator
#  Raw Data → Cleaning → Integration → Transformation → Encoding → Ready Data
#
#  Chạy riêng từng bước:
#    python src/preprocessing/step1_cleaning.py
#    python src/preprocessing/step2_integration.py
#    python src/preprocessing/step3_transformation.py
#    python src/preprocessing/step4_encoding.py
#
#  MinIO layout sau preprocessing:
#    landscape-data/
#      raw/images/          ← ảnh thô từ crawler
#      preprocessed/images/
#        mountain/          ← ảnh 640×640 PNG + augmented
#        forest/
#        sea/
#        desert/
#        snow/
# ================================================================

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.storage.minio_client    import MinioClient
from src.storage.mongodb_client  import MongoDBClient
from src.preprocessing.step1_cleaning       import run_cleaning
from src.preprocessing.step2_integration    import run_integration
from src.preprocessing.step3_transformation import run_transformation
from src.preprocessing.step4_encoding       import run_encoding
from configs.config import KEYWORDS


def main():
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("  Raw → Cleaning → Integration → Transformation → Encoding")
    print("=" * 60)

    minio = MinioClient()
    mongo = MongoDBClient()

    steps = [
        ("1. Cleaning",       run_cleaning),
        ("2. Integration",    run_integration),
        ("3. Transformation", run_transformation),
        ("4. Encoding",       run_encoding),
    ]
    results = {}

    for name, fn in steps:
        print(f"\n{'─'*55}")
        print(f"  BƯỚC {name}")
        print(f"{'─'*55}")
        t0    = time.time()
        stats = fn(minio, mongo)
        elapsed = time.time() - t0
        results[name] = {"stats": stats, "time": round(elapsed, 1)}
        print(f"  Thời gian: {elapsed:.1f}s")

    # ── Tổng kết ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  KẾT QUẢ PREPROCESSING")
    print(f"{'='*60}")
    for name, r in results.items():
        ok = r["stats"].get("ok", "?")
        print(f"  {name:<28} OK={ok:>6}  ({r['time']}s)")

    col_clean    = mongo.get_col("clean")
    col_features = mongo.get_col("features")

    print(f"\n  MongoDB images_clean   : {col_clean.count_documents({}):,}")
    print(f"  MongoDB image_features : {col_features.count_documents({}):,}")
    print(f"\n  MinIO preprocessed/images/:")
    for kw in KEYWORDS:
        cnt = len(minio.list_objects(prefix=f"preprocessed/images/{kw}/"))
        print(f"    {kw:<12} {cnt:,} ảnh")

    print(f"\n  Tiếp theo: notebooks/eda_preprocessing.ipynb")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
