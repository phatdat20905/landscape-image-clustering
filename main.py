# main.py
# ================================================================
#  Data Collection Pipeline – Orchestrator
#  Chạy tất cả crawler liên tiếp: Pexels → Unsplash → Bing → Google
#
#  Hoặc chạy từng crawler riêng lẻ:
#    python src/crawler/crawl_pexels.py
#    python src/crawler/crawl_unsplash.py
#    python src/crawler/crawl_google.py
#    python src/crawler/crawl_google.py --show     # hiện Chrome
#    python src/crawler/crawl_google.py --target 500
#
#  Cài đặt:
#    pip install requests opencv-python pymongo minio numpy \
#                selenium webdriver-manager
#
#  Khởi động dịch vụ:
#    docker run -d -p 9000:9000 -p 9001:9001 \
#        -e MINIO_ROOT_USER=minioadmin \
#        -e MINIO_ROOT_PASSWORD=minioadmin \
#        minio/minio server /data --console-address ':9001'
#    docker run -d -p 27017:27017 mongo
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient

from src.crawler.crawl_pexels   import crawl_pexels,   TARGET as PEXELS_TARGET
from src.crawler.crawl_unsplash import crawl_unsplash, TARGET as UNSPLASH_TARGET
from src.crawler.crawl_google   import crawl_google,   TARGET as GOOGLE_TARGET


def main():
    print("=" * 60)
    print("  DATA COLLECTION PIPELINE – Image Clustering Project")
    print("=" * 60)

    minio = MinioClient()
    mongo = MongoDBClient()

    total_target = PEXELS_TARGET + UNSPLASH_TARGET + GOOGLE_TARGET
    existing     = mongo.count()

    print(f"\n  Hiện có : {existing:,} ảnh trong MongoDB")
    print(f"  Mục tiêu: Pexels={PEXELS_TARGET:,} | Unsplash={UNSPLASH_TARGET:,} "
          f" | Google={GOOGLE_TARGET:,}")
    print(f"  Tổng    : {total_target:,} ảnh\n")

    # ── BƯỚC 1: Pexels ────────────────────────────────────────
    print("\n" + "─" * 52)
    print("  BƯỚC 1/3 – PEXELS")
    print("─" * 52)
    crawl_pexels(minio, mongo, target=PEXELS_TARGET)

    # ── BƯỚC 2: Unsplash ──────────────────────────────────────
    print("\n" + "─" * 52)
    print("  BƯỚC 2/3 – UNSPLASH")
    print("─" * 52)
    crawl_unsplash(minio, mongo, target=UNSPLASH_TARGET)

    # ── BƯỚC 3 Google (Selenium) ─────────────────────────────
    print("\n" + "─" * 52)
    print("  BƯỚC 3/3 – GOOGLE  (Selenium headless)")
    print("─" * 52)
    crawl_google(minio, mongo, target=GOOGLE_TARGET, headless=True)

    # ── Tổng kết ──────────────────────────────────────────────
    total  = mongo.count()
    by_src = mongo.count_by_source()
    by_kw  = mongo.count_by_keyword()

    print(f"\n{'='*60}")
    print(f"  HOÀN THÀNH DATA COLLECTION")
    print(f"{'='*60}")
    print(f"  Tổng ảnh trong MongoDB : {total:,}\n")

    print("  Theo nguồn:")
    for src, cnt in sorted(by_src.items()):
        pct = cnt / total * 100 if total else 0
        print(f"    {src:<12} {cnt:>5} ảnh  ({pct:.1f}%)")

    print("\n  Theo keyword:")
    for kw, cnt in sorted(by_kw.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100 if total else 0
        print(f"    {kw:<12} {cnt:>5} ảnh  ({pct:.1f}%)")

    print(f"\n  Tiếp theo: jupyter notebook notebooks/eda.ipynb")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()