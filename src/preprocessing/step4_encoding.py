# src/preprocessing/step4_encoding.py
# ================================================================
#  STEP 4 – ENCODING (Feature Extraction) → Ready Data
#  Workflow: Transformation → [Encoding] → Ready Data
#
#  Kỹ thuật:
#    1. HOG (Histogram of Oriented Gradients)
#    2. Color Histogram (RGB, 32 bins/kênh, L1-normalize)
#    3. ResNet50 Pretrained → 2048-dim embedding (Global Avg Pool)
#    4. Z-score normalization trên mỗi vector
#    5. Concat → feature_vector dùng cho Clustering
#
#  Input : MongoDB images_transformed (transformed=True)
#          MinIO preprocessed/images/{keyword}/*.png
#  Output: MongoDB image_features
#          {filename, keyword, source, hog_vector, color_hist_vector,
#           resnet_vector, feature_vector, feature_dim, encoded_at}
#
#  Cài : pip install torch torchvision scikit-image
#  Chạy: python src/preprocessing/step4_encoding.py
# ================================================================

import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2, numpy as np
from PIL import Image
from datetime import datetime
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError
from skimage.feature import hog

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARNING] PyTorch không có – chỉ dùng HOG + Color Histogram")

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET

# ── Config ──────────────────────────────────────────────────────
PREPROC_PREFIX     = "preprocessed/images"
FEATURES_PREFIX    = "features"  # MinIO path for vector storage
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9
COLOR_HIST_BINS     = 32
BATCH_SIZE          = 32


# ================================================================
#  1. HOG
# ================================================================
def extract_hog(pil_img: Image.Image) -> np.ndarray:
    gray = np.array(pil_img.convert("L"))
    feat = hog(gray,
               orientations=HOG_ORIENTATIONS,
               pixels_per_cell=HOG_PIXELS_PER_CELL,
               cells_per_block=HOG_CELLS_PER_BLOCK,
               block_norm="L2-Hys", feature_vector=True)
    return feat.astype(np.float32)


# ================================================================
#  2. Color Histogram
# ================================================================
def extract_color_hist(pil_img: Image.Image) -> np.ndarray:
    arr  = np.array(pil_img, dtype=np.uint8)
    hist = np.concatenate([
        np.histogram(arr[:,:,c], bins=COLOR_HIST_BINS, range=(0,256))[0]
        for c in range(3)
    ]).astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


# ================================================================
#  3. ResNet50 Encoder
# ================================================================
class ResNetEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Bỏ lớp FC → Global Average Pooling → 2048 dim
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)
        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        print(f"  [ResNet50] Device={self.device} | Output=2048 dim")

    @torch.no_grad()
    def encode(self, pil_imgs: list) -> np.ndarray:
        t = torch.stack([self.tf(im) for im in pil_imgs]).to(self.device)
        f = self.model(t).squeeze(-1).squeeze(-1)
        return f.cpu().numpy().astype(np.float32)


# ================================================================
#  4. Z-score normalization
# ================================================================
def zscore(vec: np.ndarray) -> np.ndarray:
    std = vec.std()
    return (vec - vec.mean()) / std if std > 1e-8 else vec - vec.mean()


# ================================================================
#  5. Save vectors to MinIO as .npz (compressed)
# ================================================================
def save_vectors_to_minio(minio: MinioClient, label: str, basename: str,
                          hog_v: np.ndarray, color_v: np.ndarray,
                          resnet_v: np.ndarray, concat_v: np.ndarray) -> str:
    """
    Save all vectors to MinIO as .npz (numpy compressed archive).
    Returns the MinIO object path.
    """
    vector_obj = f"{FEATURES_PREFIX}/{label}/{os.path.splitext(basename)[0]}.npz"
    
    # Create npz in memory
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        hog=hog_v,
        color_hist=color_v,
        resnet=resnet_v if resnet_v is not None else np.array([]),
        feature=concat_v,
    )
    buf.seek(0)
    npz_bytes = buf.getvalue()
    
    # Upload to MinIO
    try:
        minio.put_object(
            MINIO_BUCKET, vector_obj,
            data=io.BytesIO(npz_bytes),
            length=len(npz_bytes),
            content_type="application/octet-stream",
        )
        return vector_obj
    except Exception as e:
        print(f"  [!] Lỗi upload vector {vector_obj}: {e}")
        return None

def run_encoding(minio: MinioClient, mongo: MongoDBClient):
    col_transformed = mongo.get_col("transformed")
    col_features = mongo.get_col("features")
    # Use composite unique index (filename + object_name) so multiple aug variants
    # from the same filename can coexist but the same object won't be re-inserted.
    col_features.create_index([("filename", ASCENDING), ("object_name", ASCENDING)], unique=True)
    col_features.create_index([("label",  ASCENDING)])

    # We do NOT modify images_transformed here. To make the step resumable
    # we consider any transformed=True doc and skip it when a corresponding
    # feature document already exists in `features` (by object_name).
    query = {"transformed": True}
    total = col_transformed.count_documents(query)
    print(f"[Encoding] Tổng transformed docs: {total:,} (skipping already-encoded ones)")

    resnet_enc = None
    if TORCH_OK:
        try:
            resnet_enc = ResNetEncoder()
        except Exception as e:
            print(f"  [!] ResNet50 lỗi: {e}")

    stats  = {"ok": 0, "error": 0, "skipped": 0}
    docs   = list(col_transformed.find(query, {"_id": 0}))

    for b_start in range(0, len(docs), BATCH_SIZE):
        batch    = docs[b_start: b_start + BATCH_SIZE]
        pil_imgs, valid_docs = [], []

        for doc in batch:
            fname   = doc.get("filename", "")
            label = doc.get("label", "unknown")
            obj     = doc.get("object_name", "")
            
            if not obj:
                print(f"  [!] {fname}: object_name không có")
                stats["error"] += 1
                continue
            # Skip if features already contains this object (resumable)
            try:
                if col_features.find_one({"object_name": obj}, {"_id": 1}):
                    stats["skipped"] += 1
                    continue
            except Exception:
                # if features lookup fails for any reason, we'll attempt download/encode
                pass
            try:
                resp = minio.client.get_object(MINIO_BUCKET, obj)
                raw  = resp.read(); resp.close()
                pil  = Image.open(io.BytesIO(raw)).convert("RGB")
                pil_imgs.append(pil)
                valid_docs.append(doc)
            except Exception as e:
                print(f"  [!] Lỗi tải {fname}: {e}")
                stats["error"] += 1

        if not pil_imgs:
            continue

        # ResNet batch
        resnet_vecs = None
        if resnet_enc:
            try:
                resnet_vecs = resnet_enc.encode(pil_imgs)
            except Exception as e:
                print(f"  [!] ResNet batch lỗi: {e}")

        for j, (pil, doc) in enumerate(zip(pil_imgs, valid_docs)):
            fname   = doc.get("filename", "")
            label = doc.get("label", "unknown")
            obj_name = doc.get("object_name", "")
            is_aug = doc.get("is_augmented", False)
            aug_idx = doc.get("aug_index")

            hog_v   = zscore(extract_hog(pil))
            color_v = extract_color_hist(pil)
            resnet_v = zscore(resnet_vecs[j]) if resnet_vecs is not None else None

            parts  = [hog_v, color_v] + ([resnet_v] if resnet_v is not None else [])
            concat = zscore(np.concatenate(parts).astype(np.float32))

            # ── Save vectors to MinIO ──────────────────────────────────
            vector_obj = save_vectors_to_minio(
                minio, label, fname, hog_v, color_v, resnet_v, concat
            )
            if not vector_obj:
                stats["error"] += 1
                continue

            # ── Save metadata to MongoDB (no vectors) ─────────────────
            feat_doc = {
                "filename": fname,
                "parent_filename": doc.get("parent_filename", None),
                "label": label,
                "object_name": obj_name,
                "vector_object": vector_obj,  # ← MinIO path for vectors
                "is_augmented": is_aug,
                "aug_index": aug_idx,
                "width": doc.get("width", 0),
                "height": doc.get("height", 0),
                "encoded": True,
                # Dimensions for reference (to reconstruct without needing vectors)
                "hog_dim": len(hog_v),
                "color_hist_dim": len(color_v),
                "resnet_dim": len(resnet_v) if resnet_v is not None else 0,
                "feature_dim": len(concat),
                "encoded_at": datetime.now().strftime("%Y-%m-%d"),
            }
            try:
                col_features.insert_one(feat_doc)
                # NOTE: do not modify images_transformed here (no side-effects in Step 4)
                stats["ok"] += 1
            except DuplicateKeyError:
                # race: another worker inserted the feature concurrently — skip
                stats["skipped"] += 1
            except Exception:
                stats["error"] += 1

        done = b_start + len(batch)
        print(f"  {done}/{total} | OK={stats['ok']}", end="\r")

    dim_hog   = len(extract_hog(Image.new("RGB", (640, 640))))
    dim_color = COLOR_HIST_BINS * 3
    print(f"""
[Encoding] HOÀN THÀNH
  Đã encode   : {stats['ok']:,} ảnh
  Bỏ qua (đã có) : {stats['skipped']:,} ảnh
  Lỗi         : {stats['error']:,}
  
  Storage:
    Vectors  → MinIO: features/{{label}}/{{basename}}.npz (~1-2 KB mỗi ảnh)
    Metadata → MongoDB: image_features (chỉ dims + paths, không vectors)
  
  Dims: HOG={dim_hog}  ColorHist={dim_color}  ResNet={'2048' if TORCH_OK else 'N/A'}
  Feature_dim = {dim_hog} + {dim_color} + {'2048' if TORCH_OK else '0'} = {dim_hog+dim_color+(2048 if TORCH_OK else 0)}
""")
    return stats


if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 4 – ENCODING (Feature Extraction)")
    print("=" * 55)
    minio = MinioClient()
    mongo = MongoDBClient()
    run_encoding(minio, mongo)
