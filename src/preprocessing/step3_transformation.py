# src/preprocessing/step3_transformation.py
# ================================================================
#  STEP 3 – DATA TRANSFORMATION
#  Workflow: Integration → [Transformation] → Encoding
#
#  Kỹ thuật:
#    1. Tải ảnh từ raw/images/{stem}
#    2. Resize 640×640 PNG (letter-box, LANCZOS)
#    3. Histogram Equalization (cân bằng ánh sáng kênh Y)
#    4. Normalization stats (mean/std RGB → lưu vào MongoDB)
#    5. Augmentation (Flip, Rotate, Brightness, Crop, Noise)
#
#  Input : MinIO raw/images/{stem}
#          MongoDB images_integrated (integrated=True)
#  Output: MinIO preprocessed/images/{keyword}/{stem}.png
#          MinIO preprocessed/images/{keyword}/{stem}_aug1.png ...
#          MongoDB images_transformed:
#          {filename, keyword, source, transform_object_name,
#           aug_object_names, width=640, height=640, format=PNG,
#           norm_mean_*, norm_std_*, norm_brightness, transformed_at}
#
#  Chạy: python src/preprocessing/step3_transformation.py
# ================================================================

import sys, os, io, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from datetime import datetime

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET

# ── Config ──────────────────────────────────────────────────────
TARGET_SIZE       = (640, 640)
RAW_PREFIX        = "raw/images"              # nguồn ảnh gốc
PREPROC_PREFIX    = "preprocessed/images"    # đích sau transform
AUGMENT_ENABLED   = True
AUGMENT_PER_IMAGE = 2
HIST_EQ_ENABLED   = True


# ================================================================
#  HELPERS
# ================================================================
def resize_640(pil_img: Image.Image) -> Image.Image:
    """Resize → 640×640, giữ aspect ratio + letter-box trắng."""
    img = pil_img.copy()
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)
    padded = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
    ox = (TARGET_SIZE[0] - img.width)  // 2
    oy = (TARGET_SIZE[1] - img.height) // 2
    padded.paste(img, (ox, oy))
    return padded


def hist_equalization(pil_img: Image.Image) -> Image.Image:
    """Cân bằng ánh sáng kênh Y trong YCbCr."""
    ycbcr    = pil_img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_eq     = Image.fromarray(cv2.equalizeHist(np.array(y)))
    return Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB")


def compute_norm_stats(pil_img: Image.Image) -> dict:
    """Tính mean/std RGB → lưu MongoDB, dùng Z-score ở Encoding."""
    arr = np.array(pil_img, dtype=np.float32)
    return {
        "norm_mean_r": round(float(arr[:,:,0].mean()), 4),
        "norm_mean_g": round(float(arr[:,:,1].mean()), 4),
        "norm_mean_b": round(float(arr[:,:,2].mean()), 4),
        "norm_std_r":  round(float(arr[:,:,0].std()),  4),
        "norm_std_g":  round(float(arr[:,:,1].std()),  4),
        "norm_std_b":  round(float(arr[:,:,2].std()),  4),
        "norm_brightness": round(float(np.array(pil_img.convert("L")).mean()), 4),
    }


def augment_one(pil_img: Image.Image) -> Image.Image:
    """Sinh 1 biến thể: Flip, Rotate, Brightness, Crop, Noise."""
    img = pil_img.copy()
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(random.uniform(-30, 30), resample=Image.BILINEAR)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    w, h = img.size
    cr   = random.uniform(0.8, 1.0)
    cw, ch = int(w*cr), int(h*cr)
    l = random.randint(0, w-cw); t = random.randint(0, h-ch)
    img = img.crop((l, t, l+cw, t+ch)).resize(TARGET_SIZE, Image.LANCZOS)
    if random.random() > 0.5:
        arr   = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(0, 10), arr.shape)
        img   = Image.fromarray(np.clip(arr+noise, 0, 255).astype(np.uint8))
    return img


def to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def upload_png(minio: MinioClient, obj_name: str, png_bytes: bytes) -> bool:
    return minio.put_object(
        MINIO_BUCKET, obj_name,
        data=io.BytesIO(png_bytes),
        length=len(png_bytes),
        content_type="image/png",
    )


# ================================================================
#  MAIN
# ================================================================
def run_transformation(minio: MinioClient, mongo: MongoDBClient):
    col_integrated = mongo.get_col("integrated")
    col_transformed = mongo.get_col("transformed")
    # Ensure unique index on object_name so we can detect already-processed objects
    try:
        col_transformed.create_index([("object_name", 1)], unique=True)
    except Exception:
        pass

    query = {"integrated": True, "transformed": {"$ne": True}}
    total = col_integrated.count_documents(query)
    print(f"[Transformation] Ảnh cần transform: {total:,}")
    print(f"  Resize  : {TARGET_SIZE[0]}×{TARGET_SIZE[1]} PNG")
    print(f"  HistEQ  : {'ON' if HIST_EQ_ENABLED else 'OFF'}")
    print(f"  Augment : {'ON ×'+str(AUGMENT_PER_IMAGE) if AUGMENT_ENABLED else 'OFF'}")
    print(f"  Input   : {RAW_PREFIX}/{{stem}}")
    print(f"  Output  : {PREPROC_PREFIX}/{{label}}/")

    stats = {"ok": 0, "aug": 0, "error": 0}

    for i, doc in enumerate(col_integrated.find(query, {"_id": 0}), 1):
        fname = doc.get("filename", "")
        label = doc.get("label", "unknown")
        stem = os.path.splitext(fname)[0]

        # ── Tải ảnh từ MinIO raw (nguồn gốc) ────────────────────
        raw_obj = f"{RAW_PREFIX}/{fname}"
        try:
            resp = minio.client.get_object(MINIO_BUCKET, raw_obj)
            raw = resp.read(); resp.close()
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            # capture original dimensions before any resize
            orig_w, orig_h = pil.width, pil.height
        except Exception as e:
            print(f"  [!] Lỗi tải {fname} từ {raw_obj}: {e}")
            stats["error"] += 1
            continue

        # ── 1. Resize 640×640 ───────────────────────────────────
        pil = resize_640(pil)

        # ── 2. Histogram Equalization ────────────────────────────
        if HIST_EQ_ENABLED:
            pil = hist_equalization(pil)

        # ── 3. Tính normalization stats ─────────────────────────
        norm = compute_norm_stats(pil)

        # ── Upload vào preprocessed/ ────────────────────────────
        preproc_obj = f"{PREPROC_PREFIX}/{label}/{stem}.png"
        png_bytes = to_png_bytes(pil)

        # If this object has already been recorded in images_transformed, skip
        if col_transformed.find_one({"object_name": preproc_obj}):
            # already processed (main image)
            continue

        try:
            ok = upload_png(minio, preproc_obj, png_bytes)
            if not ok:
                stats["error"] += 1
                continue
            # Verify object exists
            minio.client.stat_object(MINIO_BUCKET, preproc_obj)
        except Exception as e:
            print(f"  [!] Lỗi upload {preproc_obj}: {e}")
            stats["error"] += 1
            continue

        # ── Lưu document cho ảnh chính vào collection images_transformed ────
        main_doc = {
            # filename should match the stored object (preprocessed PNG basename)
            "filename": os.path.basename(preproc_obj),
            "parent_filename": None,
            "label": label,
            "object_name": preproc_obj,
            "width_raw": orig_w if 'orig_w' in locals() else doc.get("width", 0),
            "height_raw": orig_h if 'orig_h' in locals() else doc.get("height", 0),
            "width": TARGET_SIZE[0],
            "height": TARGET_SIZE[1],
            "format": "PNG",
            "is_augmented": False,
            "aug_index": None,
            "transformed": True,
            "transformed_at": datetime.now().strftime("%Y-%m-%d"),
            **norm,
        }

        try:
            col_transformed.insert_one(main_doc)
            stats["ok"] += 1
        except Exception as e:
            print(f"  [!] Lỗi lưu MongoDB (main) {fname}: {e}")
            stats["error"] += 1

        # ── 4. Augmentation → cùng thư mục preprocessed/ ───────
        if AUGMENT_ENABLED:
            for j in range(AUGMENT_PER_IMAGE):
                aug_obj = f"{PREPROC_PREFIX}/{label}/{stem}_aug{j+1}.png"
                aug_png = to_png_bytes(augment_one(pil))
                try:
                    if upload_png(minio, aug_obj, aug_png):
                        # Verify
                        minio.client.stat_object(MINIO_BUCKET, aug_obj)
                        
                        # ── Lưu document riêng cho augmented ────────────────
                        aug_doc = {
                            "filename": os.path.basename(aug_obj),
                            "parent_filename": os.path.basename(preproc_obj),
                            "label": label,
                            "object_name": aug_obj,
                            "width_raw": orig_w if 'orig_w' in locals() else doc.get("width", 0),
                            "height_raw": orig_h if 'orig_h' in locals() else doc.get("height", 0),
                            "width": TARGET_SIZE[0],
                            "height": TARGET_SIZE[1],
                            "format": "PNG",
                            "is_augmented": True,
                            "aug_index": j + 1,
                            "transformed": True,
                            "transformed_at": datetime.now().strftime("%Y-%m-%d"),
                            **norm,
                        }
                        # if augmented object already exists in DB, skip inserting/uploading
                        if col_transformed.find_one({"object_name": aug_obj}):
                            continue
                        try:
                            col_transformed.insert_one(aug_doc)
                            stats["aug"] += 1
                        except Exception as e:
                            print(f"  [!] Lỗi lưu MongoDB (aug) {fname} aug{j+1}: {e}")
                except Exception:
                    pass
        
        # ── Đánh dấu integrated đã được transform ────────────────
        # Note: do not modify the previous collection (images_integrated) here.
        # Step 3 writes into images_transformed only. Status is determined
        # by presence of documents in images_transformed (or features collection).

        if i % 100 == 0:
            print(f"  {i}/{total} | OK={stats['ok']} Aug={stats['aug']}", end="\r")

    print(f"""
[Transformation] HOÀN THÀNH
  Đã transform  : {stats['ok']:,} ảnh → 640×640 PNG
  Augmentation  : {stats['aug']:,} ảnh mới
  Lỗi           : {stats['error']:,}
  Input         : {RAW_PREFIX}/
  Output        : {PREPROC_PREFIX}/{{label}}/
  Collection    : images_transformed
""")
    return stats


if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 3 – DATA TRANSFORMATION")
    print("=" * 55)
    minio = MinioClient()
    mongo = MongoDBClient()
    run_transformation(minio, mongo)
