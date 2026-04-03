# ================================================================
#  TEST CONNECTION: MongoDB + MinIO
# ================================================================

# Ensure project root is on sys.path so imports like `src.*` and `configs.*` work
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.mongodb_client import MongoDBClient
from src.storage.minio_client   import MinioClient
from configs.config import MINIO_BUCKET

def test_mongodb():
    print("\n=== TEST MONGODB ===")

    try:
        mongo = MongoDBClient()

        # Test lấy collection raw
        col = mongo.get_col("raw")

        # Test insert
        test_doc = {
            "filename": "test.jpg",
            "source": "test",
            "keyword": "test",
        }

        col.insert_one(test_doc)
        print("✅ Insert OK")

        # Test count
        count = col.count_documents({"source": "test"})
        print(f"✅ Count OK: {count}")

        # Cleanup (xóa dữ liệu test)
        col.delete_many({"source": "test"})
        print("🧹 Cleanup OK")

        print("🎉 MongoDB OK!")

    except Exception as e:
        print("❌ MongoDB FAILED:", e)


def test_minio():
    print("\n=== TEST MINIO ===")

    try:
        minio = MinioClient()

        # Test upload bytes giả
        test_data = b"hello minio"

        object_name = "test/test.txt"

        ok = minio.put_object(
            MINIO_BUCKET,
            object_name,
            data=__import__("io").BytesIO(test_data),
            length=len(test_data),
            content_type="text/plain"
        )

        if ok:
            print("✅ Upload OK")

            # Test list objects
            objects = minio.list_objects(prefix="test/")
            print(f"✅ List OK: {objects}")

            print("🎉 MinIO OK!")
        else:
            print("❌ Upload failed")

    except Exception as e:
        print("❌ MinIO FAILED:", e)


# ================================================================
# RUN TEST
# ================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("   TEST CONNECTIONS")
    print("=" * 50)

    test_mongodb()
    test_minio()