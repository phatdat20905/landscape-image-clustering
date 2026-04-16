# src/preprocessing/step4_encoding.py
# ================================================================
#  STEP 4 – ENCODING (Feature Extraction) → Ready Data
#  Workflow: Transformation → [Encoding] → Ready Data
#
#  Pipeline (đồng bộ với test2.py):
#    final_norm:  Normalize(ImageNet) → ToTensorV2
#    model:       ResNet50 pretrained → bỏ FC → GAP → 2048 dim
#    post:        Z-score normalization
#
#  Storage: vector lưu thẳng vào MongoDB image_features.resnet_vector
#           (list[float] 2048, ~16 KB/doc) – không dùng MinIO
#
#  Resume: index unique object_name → skip doc đã encode
#
#  Input : MongoDB images_transformed
#          MinIO preprocessed/images/{label}/*.png
#  Output: MongoDB image_features
#
#  Cài  : pip install torch torchvision albumentations
#  Chạy : python src/preprocessing/step4_encoding.py
# ================================================================

import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from PIL import Image
from datetime import datetime
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

import torch
import torchvision.models as models
import torch.nn as nn

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET

# Import final_norm pipeline từ step3 (đồng bộ test2.py)
from src.preprocessing.step3_transformation import get_final_norm_pipeline

# ── Config ────────────────────────────────────────────────────────
BATCH_SIZE = 32   # giảm xuống 16 nếu OOM GPU


# ================================================================
#  FeatureExtractor (giống test2.py)
#  ResNet50 pretrained, bỏ FC → output 2048 dim
# ================================================================
class FeatureExtractor(nn.Module):
    """
    ResNet50 pretrained (ImageNet), bỏ lớp FC cuối.
    Forward: tensor (N,3,224,224) → vector (N, 2048)

    Giống test2.py:
        resnet = models.resnet50(weights=...)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    """

    def __init__(self):
        super().__init__()
        resnet         = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone  = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)            # (N, 2048, 1, 1)
            return feat.view(feat.size(0), -1) # (N, 2048)


# ================================================================
#  Z-score normalization
# ================================================================
def zscore(vec: np.ndarray) -> np.ndarray:
    std = vec.std()
    return (vec - vec.mean()) / std if std > 1e-8 else vec - vec.mean()


# ================================================================
#  MAIN
# ================================================================
def run_encoding(minio: MinioClient, mongo: MongoDBClient):
    col_transformed = mongo.get_col("transformed")
    col_features    = mongo.get_col("features")

    # ── Index ────────────────────────────────────────────────────
    try:
        col_features.create_index(
            [("object_name", ASCENDING)],
            unique=True, name="obj_unique",
        )
        col_features.create_index([("label",          ASCENDING)])
        col_features.create_index([("filename", ASCENDING)])
        col_features.create_index([("is_augmented",   ASCENDING)])
    except Exception:
        pass

    total  = col_transformed.count_documents({"transformed": True})
    n_done = col_features.count_documents({})
    print(f"[Encoding] Tổng transformed: {total:,} | Đã encode: {n_done:,}")

    # ── Khởi tạo model & final_norm ──────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = FeatureExtractor().to(device)
    final_norm = get_final_norm_pipeline()   # đồng bộ với test2.py

    print(f"  [ResNet50] Device: {device} | dim: 2048")
    print(f"  [final_norm] Normalize(ImageNet) → ToTensorV2")

    # ── Resume ───────────────────────────────────────────────────
    encoded_set: set[str] = set(
        d["object_name"] for d in
        col_features.find({}, {"object_name": 1, "_id": 0})
    )
    print(f"  Đã encode trước: {len(encoded_set):,} → skip")

    all_docs = list(col_transformed.find({"transformed": True}, {"_id": 0}))
    pending  = [d for d in all_docs
                if d.get("object_name", "") not in encoded_set]

    stats = {
        "ok":      0,
        "error":   0,
        "skipped": len(all_docs) - len(pending),
    }
    print(f"  Cần encode thêm: {len(pending):,} ảnh")

    # ── Batch encoding ───────────────────────────────────────────
    for b_start in range(0, len(pending), BATCH_SIZE):
        batch      = pending[b_start: b_start + BATCH_SIZE]
        pil_imgs   = []
        valid_docs = []

        for doc in batch:
            obj = doc.get("object_name", "")
            if not obj:
                stats["error"] += 1
                continue
            try:
                resp = minio.client.get_object(MINIO_BUCKET, obj)
                raw  = resp.read(); resp.close()
                pil  = Image.open(io.BytesIO(raw)).convert("RGB")
                pil_imgs.append(pil)
                valid_docs.append(doc)
            except Exception as e:
                print(f"\n  [!] MinIO tải lỗi {obj}: {e}")
                stats["error"] += 1

        if not pil_imgs:
            continue

        # ── Áp dụng final_norm (giống test2.py) ──────────────────
        tensors_list = []
        for pil in pil_imgs:
            img_np = np.array(pil)              # RGB uint8
            t = final_norm(image=img_np)["image"]  # tensor (3, H, W)
            tensors_list.append(t)

        batch_tensor = torch.stack(tensors_list).to(device)  # (N, 3, H, W)

        # ── Forward ResNet50 ─────────────────────────────────────
        with torch.no_grad():
            vectors = model(batch_tensor).cpu().numpy()  # (N, 2048)

        # ── Z-score + lưu MongoDB ────────────────────────────────
        for j, (vec_raw, doc) in enumerate(zip(vectors, valid_docs)):
            # Z-score and cast to float32 to reduce storage and match model
            # numeric precision expectations.
            vec_norm = zscore(vec_raw).astype(np.float32)   # np.ndarray (2048,)

            feat_doc = {
                # Traceability
                "filename":        doc.get("filename", ""),
                "label":           doc.get("label", "unknown"),
                "object_name":     doc.get("object_name", ""),
                "is_augmented":    doc.get("is_augmented", False),
                "aug_index":       doc.get("aug_index"),
                "width":           doc.get("width", 640),
                "height":          doc.get("height", 640),

                # ResNet50 embedding – lưu thẳng vào MongoDB
                "resnet_vector":   vec_norm.tolist(),   # list[float32] 2048
                "resnet_dim":      int(vec_norm.shape[0]),
                "encoded_at": datetime.now().strftime("%Y-%m-%d"),
                "encoded": True,
            }

            try:
                col_features.insert_one(feat_doc)
                stats["ok"] += 1
            except DuplicateKeyError:
                stats["skipped"] += 1
            except Exception as e:
                print(f"\n  [!] MongoDB insert lỗi {doc.get('filename','')}: {e}")
                stats["error"] += 1

        done = b_start + len(batch)
        print(
            f"  [{done}/{len(pending)}] "
            f"OK={stats['ok']} Skip={stats['skipped']} Err={stats['error']}",
            end="\r",
        )

    print(f"""
[Encoding] HOÀN THÀNH  →  image_features
  ┌──────────────────────────────────────────────────────┐
  │  Đã encode     : {stats['ok']:>6,} ảnh (mới)                 │
  │  Bỏ qua (done) : {stats['skipped']:>6,}                          │
  │  Lỗi           : {stats['error']:>6,}                            │
  ├──────────────────────────────────────────────────────┤
  │  Model         : ResNet50 ImageNet IMAGENET1K_V2     │
  │  Pipeline      : Normalize(ImageNet) → ToTensorV2   │
  │  Output        : resnet_vector (2048 dim, Z-score)  │
  │  Storage       : MongoDB image_features              │
  └──────────────────────────────────────────────────────┘
""")
    return stats


# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 4 – ENCODING (ResNet50 → MongoDB)")
    print("  final_norm: Normalize(ImageNet) → ToTensorV2")
    print("=" * 60)
    minio = MinioClient()
    mongo = MongoDBClient()
    run_encoding(minio, mongo)