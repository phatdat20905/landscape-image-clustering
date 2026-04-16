# src/preprocessing/step34_pipeline.py
# ================================================================
#  STEP 3+4 COMBINED PIPELINE
#  Transform → Encode liên tiếp, xử lý từng ảnh xong encode ngay.
#
#  Lợi ích so với chạy step3 xong step4:
#    - Ảnh chỉ cần tải từ MinIO 1 lần (thay vì 2 lần)
#    - Tiết kiệm memory: không cần lưu toàn bộ ảnh đã transform
#    - Nếu dừng giữa chừng, resume sẽ tự skip đúng ảnh
#
#  Resume:
#    - Kiểm tra images_transformed (source_filename, aug_index)
#    - Kiểm tra image_features (object_name)
#    → Chỉ xử lý ảnh chưa có trong cả 2 collection
#
#  Cài  : pip install albumentations torch torchvision opencv-python
#  Chạy :
#    python src/preprocessing/step34_pipeline.py
#    python src/preprocessing/step34_pipeline.py --no-aug    # tắt augmentation
#    python src/preprocessing/step34_pipeline.py --batch 16  # batch size nhỏ hơn
# ================================================================

import sys, os, io, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

import torch
import torch.nn as nn
import torchvision.models as models

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET

# Import pipelines từ step3
from src.preprocessing.step3_transformation import (
    STANDARD_PIPELINE,
    AUG_PIPELINE,
    get_final_norm_pipeline,
    compute_norm_stats,
    load_rgb_from_minio,
    rgb_to_png_bytes,
    upload_png,
    build_doc,
    TARGET_SIZE,
    PAD_FILL,
    PREPROC_PREFIX,
    RAW_PREFIX,
    AUGMENT_PER_IMAGE,
)

# Import model từ step4
from src.preprocessing.step4_encoding import FeatureExtractor, zscore

# ── Config ────────────────────────────────────────────────────────
DEFAULT_BATCH = 32


# ================================================================
#  COMBINED PROCESSOR
# ================================================================
class Step34Processor:
    """
    Kết hợp transform + encode trong 1 pass.
    Mỗi ảnh: load → standard → encode → augment → encode aug
    """

    def __init__(self, minio: MinioClient, mongo: MongoDBClient,
                 augment_enabled: bool = True,
                 augment_per_image: int = AUGMENT_PER_IMAGE,
                 batch_size: int = DEFAULT_BATCH):

        self.minio     = minio
        self.mongo     = mongo
        self.aug_on    = augment_enabled
        self.n_aug     = augment_per_image
        self.batch_sz  = batch_size

        self.col_int   = mongo.get_col("integrated")
        self.col_trn   = mongo.get_col("transformed")
        self.col_feat  = mongo.get_col("features")

        # Model
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = FeatureExtractor().to(self.device)
        self.final_norm = get_final_norm_pipeline()

        print(f"  [Model] ResNet50 on {self.device}")
        print(f"  [Pipeline] standard→CLAHE, aug×{self.n_aug}: GaussBlur+HFlip+BC")

    def _ensure_indexes(self):
        try:
            self.col_trn.create_index(
                [("filename", ASCENDING), ("aug_index", ASCENDING)],
                unique=True, name="src_aug_unique")
            self.col_trn.create_index([("label", ASCENDING)])
            self.col_feat.create_index(
                [("object_name", ASCENDING)],
                unique=True, name="obj_unique")
            self.col_feat.create_index([("label", ASCENDING)])
            self.col_feat.create_index([("is_augmented", ASCENDING)])
        except Exception:
            pass

    def _load_done_sets(self) -> tuple[set, set]:
        """Load tập đã xử lý để resume."""
        # trn_done contains stems of filenames (without extension) for resume
        trn_done = set()
        for d in self.col_trn.find({"is_augmented": False}, {"filename": 1, "_id": 0}):
            fn = d.get("filename")
            if fn:
                trn_done.add(os.path.splitext(fn)[0])
        feat_done = set(
            d["object_name"] for d in
            self.col_feat.find({}, {"object_name": 1, "_id": 0})
        )
        return trn_done, feat_done

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """PIL RGB → tensor qua final_norm."""
        img_np = np.array(pil_img)
        return self.final_norm(image=img_np)["image"]

    def _encode_batch(self, pil_imgs: list) -> np.ndarray:
        """Encode batch PIL images → np.ndarray (N, 2048)."""
        tensors = torch.stack(
            [self._pil_to_tensor(p) for p in pil_imgs]
        ).to(self.device)
        with torch.no_grad():
            feats = self.model(tensors).cpu().numpy()
        return feats

    def _save_transform(self, fname_out, obj_name, label,
                        orig_w, orig_h, norm, is_aug, aug_idx):
        doc = build_doc(fname_out, obj_name, label, orig_w, orig_h,
                        norm, is_aug, aug_idx)
        try:
            self.col_trn.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False
        except Exception as e:
            print(f"\n  [!] MongoDB transform {fname_out}: {e}")
            return False

    def _save_feature(self, transform_doc: dict, vec_raw: np.ndarray):
        # Z-score then cast to float32 to reduce size/precision before storing
        vec_norm = zscore(vec_raw).astype(np.float32)
        feat_doc = {
            "filename":        transform_doc["filename"],
            "label":           transform_doc["label"],
            "object_name":     transform_doc["object_name"],
            "is_augmented":    transform_doc["is_augmented"],
            "aug_index":       transform_doc["aug_index"],
            "width":           transform_doc["width"],
            "height":          transform_doc["height"],
            "resnet_vector":   vec_norm.tolist(),
            "resnet_dim":      int(vec_norm.shape[0]),
            "encoded_at":      datetime.now().strftime("%Y-%m-%d"),
            "encoded": True,
        }
        try:
            self.col_feat.insert_one(feat_doc)
            return True
        except DuplicateKeyError:
            return False
        except Exception as e:
            print(f"\n  [!] MongoDB feature {transform_doc['filename']}: {e}")
            return False

    def run(self):
        self._ensure_indexes()
        trn_done, feat_done = self._load_done_sets()

        total = self.col_int.count_documents({})
        print(f"[Pipeline 3+4] Tổng integrated: {total:,}")
        print(f"  Đã transform trước: {len(trn_done):,} | Đã encode: {len(feat_done):,}")

        stats = {"transform_ok": 0, "aug_ok": 0,
                 "encode_ok": 0, "error": 0, "skip": 0}

        all_docs = list(self.col_int.find({}, {"_id": 0}))
        pending  = [d for d in all_docs
                    if os.path.splitext(d.get("filename", ""))[0] not in trn_done]
        stats["skip"] = len(all_docs) - len(pending)

        # ── Batch processing ──────────────────────────────────────
        for b_start in range(0, len(pending), self.batch_sz):
            batch = pending[b_start: b_start + self.batch_sz]

            # Tải ảnh raw
            raw_items = []   # (doc, img_rgb_standard, orig_w, orig_h)
            for doc in batch:
                fname   = doc.get("filename", "")
                raw_obj = doc.get("object_name", f"{RAW_PREFIX}/{fname}")
                try:
                    img_rgb = load_rgb_from_minio(self.minio, raw_obj)
                    orig_h, orig_w = img_rgb.shape[:2]
                    img_std = STANDARD_PIPELINE(image=img_rgb)["image"]
                    raw_items.append((doc, img_std, orig_w, orig_h))
                except Exception as e:
                    print(f"\n  [!] Load {fname}: {e}")
                    stats["error"] += 1

            if not raw_items:
                continue

            # ── Encode standard images (batch) ────────────────────
            std_pils  = [Image.fromarray(it[1]) for it in raw_items]
            std_vecs  = self._encode_batch(std_pils)   # (N, 2048)

            for k, (doc, img_std, orig_w, orig_h) in enumerate(raw_items):
                fname = doc.get("filename", "")
                label = doc.get("label", "unknown")
                stem  = os.path.splitext(fname)[0]

                # ── Upload standard PNG ───────────────────────────
                main_obj  = f"{PREPROC_PREFIX}/{label}/{stem}.png"
                norm_std  = compute_norm_stats(img_std)
                try:
                    upload_png(self.minio, main_obj, rgb_to_png_bytes(img_std))
                except Exception as e:
                    print(f"\n  [!] Upload {main_obj}: {e}")
                    stats["error"] += 1
                    continue

                # Lưu transform doc
                main_trn_doc = {
                    "filename": f"{stem}.png",
                    "label": label,
                    "object_name": main_obj,
                    "width_raw": orig_w, "height_raw": orig_h,
                    "width": TARGET_SIZE, "height": TARGET_SIZE,
                    "format": "PNG",
                    "is_augmented": False, "aug_index": None,
                    "transformed": True,
                    "transformed_at": datetime.now().strftime("%Y-%m-%d"),
                    **norm_std,
                }
                if self._save_transform(
                        f"{stem}.png", main_obj, label,
                        orig_w, orig_h, norm_std,
                        False, None):
                    stats["transform_ok"] += 1

                # Lưu feature doc
                if main_obj not in feat_done:
                    if self._save_feature(main_trn_doc, std_vecs[k]):
                        stats["encode_ok"] += 1

            # ── Augmentation batch ────────────────────────────────
            if self.aug_on:
                for j in range(1, self.n_aug + 1):
                    aug_items = []
                    for doc, img_std, orig_w, orig_h in raw_items:
                        fname = doc.get("filename", "")
                        label = doc.get("label", "unknown")
                        stem  = os.path.splitext(fname)[0]
                        aug_obj = f"{PREPROC_PREFIX}/{label}/{stem}_aug{j}.png"

                        if aug_obj in feat_done:
                            continue

                        img_aug = AUG_PIPELINE(image=img_std)["image"]
                        aug_items.append((doc, img_aug, aug_obj,
                                          orig_w, orig_h, stem, label))

                    if not aug_items:
                        continue

                    aug_pils = [Image.fromarray(it[1]) for it in aug_items]
                    aug_vecs = self._encode_batch(aug_pils)

                    for k2, (doc, img_aug, aug_obj,
                             orig_w, orig_h, stem, label) in enumerate(aug_items):
                        fname    = doc.get("filename", "")
                        norm_aug = compute_norm_stats(img_aug)
                        try:
                            upload_png(self.minio, aug_obj,
                                       rgb_to_png_bytes(img_aug))
                        except Exception:
                            continue

                        aug_trn_doc = {
                            "filename": os.path.basename(aug_obj),
                            "label": label,
                            "object_name": aug_obj,
                            "width_raw": orig_w, "height_raw": orig_h,
                            "width": TARGET_SIZE, "height": TARGET_SIZE,
                            "format": "PNG",
                            "is_augmented": True, "aug_index": j,
                            "transformed": True,
                            "transformed_at": datetime.now().strftime("%Y-%m-%d"),
                            **norm_aug,
                        }
                        self._save_transform(
                            os.path.basename(aug_obj), aug_obj, label,
                            orig_w, orig_h, norm_aug, True, j)

                        if self._save_feature(aug_trn_doc, aug_vecs[k2]):
                            stats["aug_ok"] += 1
                            stats["encode_ok"] += 1

            done = b_start + len(batch)
            print(
                f"  [{done}/{len(pending)}] "
                f"Trn={stats['transform_ok']} Aug={stats['aug_ok']} "
                f"Enc={stats['encode_ok']} Err={stats['error']}",
                end="\r",
            )

        total_enc = stats["encode_ok"]
        print(f"""
[Pipeline 3+4] HOÀN THÀNH
  ┌──────────────────────────────────────────────────────┐
  │  Transform standard : {stats['transform_ok']:>6,}                      │
  │  Augmentation       : {stats['aug_ok']:>6,}                      │
  │  Encoded (total)    : {total_enc:>6,}                      │
  │  Bỏ qua (done)      : {stats['skip']:>6,}                      │
  │  Lỗi                : {stats['error']:>6,}                      │
  └──────────────────────────────────────────────────────┘
""")
        return stats


# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3+4 Combined: Transform + Encode")
    parser.add_argument("--no-aug",  action="store_true",
                        help="Tắt augmentation")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--n-aug",   type=int, default=AUGMENT_PER_IMAGE,
                        help=f"Số augmentation / ảnh (default: {AUGMENT_PER_IMAGE})")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 3+4 COMBINED PIPELINE")
    print("  Transform (Albumentations) + Encode (ResNet50)")
    print("=" * 60)

    minio = MinioClient()
    mongo = MongoDBClient()

    processor = Step34Processor(
        minio=minio, mongo=mongo,
        augment_enabled=not args.no_aug,
        augment_per_image=args.n_aug,
        batch_size=args.batch,
    )
    processor.run()