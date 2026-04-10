"""
MongoDB client – hỗ trợ multi-collection (raw, clean, features, clusters)
pip install pymongo
"""

from pymongo import MongoClient, ASCENDING
from configs.config import MONGO_URI, MONGO_DB, MONGO_COLLECTIONS


class MongoDBClient:
    def __init__(self):
        # Kết nối MongoDB Atlas
        self.client = MongoClient(
            MONGO_URI,
            #serverSelectionTimeoutMS=5000
        )

        try:
            self.client.server_info()
            print("[OK] Connected to MongoDB Atlas")
        except Exception as e:
            print("[ERROR] MongoDB connection failed:", e)

        self.db = self.client[MONGO_DB]

    # ============================================================
    # Lấy collection theo stage (raw, clean, features, clusters)
    # ============================================================
    def get_col(self, stage: str):
        if stage not in MONGO_COLLECTIONS:
            raise ValueError(f"Invalid stage: {stage}")
        return self.db[MONGO_COLLECTIONS[stage]]

    # ============================================================
    # Tạo index (gọi 1 lần cho mỗi collection)
    # ============================================================
    def create_indexes(self, col):
        col.create_index([("filename", ASCENDING)], unique=True)
        col.create_index([("source", ASCENDING)])
        col.create_index([("keyword", ASCENDING)])

    # ============================================================
    # Helper queries (dùng cho EDA)
    # ============================================================
    def count(self, col, query=None):
        return col.count_documents(query or {})

    def find_all(self, col, query=None, projection=None):
        cursor = col.find(query or {}, projection or {"_id": 0})
        return list(cursor)

    def count_by_source(self, col):
        pipeline = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}}
        ]
        return {r["_id"]: r["count"] for r in col.aggregate(pipeline)}

    def count_by_keyword(self, col):
        pipeline = [
            {"$group": {"_id": "$keyword", "count": {"$sum": 1}}}
        ]
        return {r["_id"]: r["count"] for r in col.aggregate(pipeline)}