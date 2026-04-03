# src/storage/minio_client.py
"""
MinIO client – upload ảnh trực tiếp từ memory (không lưu file tạm).
Yêu cầu: pip install minio
"""
import io
from minio import Minio
from minio.error import S3Error
from configs.config import (MINIO_ENDPOINT, MINIO_ACCESS_KEY,
                             MINIO_SECRET_KEY, MINIO_BUCKET, MINIO_SECURE)


class MinioClient:
    def __init__(self):
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(MINIO_BUCKET):
            self.client.make_bucket(MINIO_BUCKET)
            print(f"[MinIO] Bucket '{MINIO_BUCKET}' created.")

    def upload_bytes(self, object_name: str, data: bytes,
                     content_type: str = "image/jpeg") -> bool:
        """Upload ảnh từ bytes (in-memory) lên MinIO."""
        try:
            self.client.put_object(
                MINIO_BUCKET,
                object_name,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            return True
        except S3Error as e:
            print(f"[MinIO] Upload lỗi: {e}")
            return False

    def put_object(self, bucket_name: str, object_name: str,
                   data, length: int,
                   content_type: str = "image/jpeg") -> bool:
        """
        Wrapper put_object – dùng trực tiếp trong các file crawler
        (giống interface minio raw client).
        """
        try:
            self.client.put_object(
                bucket_name,
                object_name,
                data,
                length=length,
                content_type=content_type,
            )
            return True
        except S3Error as e:
            print(f"[MinIO] put_object lỗi: {e}")
            return False

    def get_url(self, object_name: str) -> str:
        """Trả về URL public của object."""
        return f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_name}"

    def list_objects(self, prefix: str = "") -> list[str]:
        return [
            obj.object_name
            for obj in self.client.list_objects(
                MINIO_BUCKET, prefix=prefix, recursive=True
            )
        ]