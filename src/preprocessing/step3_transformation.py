# src/preprocessing/step3_transformation.py
# ================================================================
#  STEP 3 – DATA TRANSFORMATION
#  Workflow: Integration → [Transformation] → Encoding
#
#  Pipeline 3 bước (theo test2.py):
#
#    standard_pipeline:
#      LongestMaxSize(640) → PadIfNeeded(640×640, fill=114) → CLAHE
#    aug_pipeline  (áp dụng lên ảnh standard):
#      GaussianBlur → HorizontalFlip → RandomBrightnessContrast
#    final_norm    (dùng ở step4 khi đưa vào model):
#      Normalize(ImageNet mean/std) → ToTensorV2
#      (KHÔNG áp dụng ở step3, chỉ lưu PNG gốc)
#
#  Kết quả lưu trên MinIO:
#    preprocessed/images/{label}/{stem}.png      ← ảnh standard
#    preprocessed/images/{label}/{stem}_aug1.png ← ảnh augmented
#    preprocessed/images/{label}/{stem}_aug2.png
#
#  Resume:
#    Index unique (source_filename, aug_index) trong images_transformed
#    → dừng bất cứ lúc nào, chạy tiếp không bị trùng.
#
#  Input : MongoDB images_integrated
#          MinIO raw/images/{filename}
#  Output: MongoDB images_transformed
#          MinIO preprocessed/images/{label}/
#
#  Cài  : pip install albumentations opencv-python pillow
#  Chạy : python src/preprocessing/step3_transformation.py
# ================================================================

import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
from datetime import datetime
from pymongo import ASCENDING

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET

# ── Config ────────────────────────────────────────────────────────
TARGET_SIZE       = 640
PAD_FILL          = 114          # gray padding – chuẩn ImageNet/YOLO
RAW_PREFIX        = "raw/images"
PREPROC_PREFIX    = "preprocessed/images"
AUGMENT_ENABLED   = True
AUGMENT_PER_IMAGE = 1


# ================================================================
#  PIPELINES ALBUMENTATIONS  (cấu trúc giống test2.py)
# ================================================================

def get_standard_pipeline(target_size: int = TARGET_SIZE) -> A.Compose:
    """
    Pipeline chuẩn cho ảnh gốc (không blur):
      1. LongestMaxSize   – scale xuống sao cho cạnh dài ≤ target_size
      2. PadIfNeeded      – pad bằng màu xám (114) về đúng target×target
      3. CLAHE            – tăng tương phản cục bộ (tốt hơn globalHistEQ)

    Tương ứng test2.py:
        standard_pipeline = A.Compose([
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(min_height=..., min_width=..., border_mode=0, fill=114),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ])
    """
    return A.Compose([
        A.LongestMaxSize(max_size=target_size),
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=PAD_FILL,
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    ])


def get_aug_pipeline() -> A.Compose:
    """
    Pipeline augmentation (áp dụng lên ảnh standard):
      1. GaussianBlur          – làm mờ nhẹ, tăng robustness noise
      2. HorizontalFlip        – lật ngang ngẫu nhiên
      3. RandomBrightnessContrast – đa dạng ánh sáng

    Tương ứng test2.py:
        aug_pipeline = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
    """
    return A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.6,
        ),
        # Thêm augmentation nâng cao
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.4,
        ),
        A.RandomResizedCrop(
            size=(TARGET_SIZE, TARGET_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.4,
        ),
    ])


def get_final_norm_pipeline() -> A.Compose:
    """
    Pipeline chuẩn hoá cuối cùng để đưa vào model (step4).
    KHÔNG dùng ở step3 – chỉ export để step4 import.

    Tương ứng test2.py:
        final_norm = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            AT.ToTensorV2()
        ])
    """
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


# Khởi tạo pipeline 1 lần
STANDARD_PIPELINE = get_standard_pipeline(TARGET_SIZE)
AUG_PIPELINE      = get_aug_pipeline()


# ================================================================
#  HELPERS
# ================================================================
def compute_norm_stats(img_rgb: np.ndarray) -> dict:
    """Tính mean/std RGB từ ảnh uint8 [0,255]. Lưu MongoDB."""
    f = img_rgb.astype(np.float32)
    return {
        "norm_mean_r":     round(float(f[:, :, 0].mean()), 4),
        "norm_mean_g":     round(float(f[:, :, 1].mean()), 4),
        "norm_mean_b":     round(float(f[:, :, 2].mean()), 4),
        "norm_std_r":      round(float(f[:, :, 0].std()),  4),
        "norm_std_g":      round(float(f[:, :, 1].std()),  4),
        "norm_std_b":      round(float(f[:, :, 2].std()),  4),
        "norm_brightness": round(float(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).mean()
        ), 4),
    }


def load_rgb_from_minio(minio: MinioClient, object_name: str) -> np.ndarray:
    """Tải ảnh từ MinIO → NumPy RGB uint8."""
    resp = minio.client.get_object(MINIO_BUCKET, object_name)
    raw  = resp.read(); resp.close()
    arr  = np.frombuffer(raw, np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2.imdecode thất bại: {object_name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_png_bytes(img_rgb: np.ndarray) -> bytes:
    """RGB → PNG bytes in-memory."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode thất bại")
    return buf.tobytes()


def upload_png(minio: MinioClient, obj_name: str, png_bytes: bytes) -> bool:
    return minio.put_object(
        MINIO_BUCKET, obj_name,
        data=io.BytesIO(png_bytes),
        length=len(png_bytes),
        content_type="image/png",
    )


def build_doc(fname_out: str, obj_name: str, label: str,
              orig_w: int, orig_h: int, norm: dict,
              is_aug: bool, aug_idx: int | None,
              ) -> dict:
    return {
        "filename":        fname_out,
        "label":           label,
        "object_name":     obj_name,
        "width_raw":       orig_w,
        "height_raw":      orig_h,
        "width":           TARGET_SIZE,
        "height":          TARGET_SIZE,
        "format":          "PNG",
        "is_augmented":    is_aug,
        "aug_index":       aug_idx,
        "transformed":     True,
        "transformed_at":  datetime.now().strftime("%Y-%m-%d"),
        **norm,
    }


# ================================================================
#  MAIN
# ================================================================
def run_transformation(minio: MinioClient, mongo: MongoDBClient):
    col_integrated  = mongo.get_col("integrated")
    col_transformed = mongo.get_col("transformed")

    # ── Index để resume ──────────────────────────────────────────
    try:
        col_transformed.create_index(
            [("filename", ASCENDING), ("aug_index", ASCENDING)],
            unique=True, name="src_aug_unique",
        )
        col_transformed.create_index([("label",        ASCENDING)])
        col_transformed.create_index([("is_augmented", ASCENDING)])
    except Exception:
        pass

    total = col_integrated.count_documents({})
    print(f"[Transformation] Tổng integrated: {total:,}")
    print(f"  standard_pipeline: LongestMaxSize({TARGET_SIZE}) → PadIfNeeded(fill={PAD_FILL}) → CLAHE")
    print(f"  aug_pipeline     : GaussianBlur → HFlip → BrightnessContrast → HSV → RndCrop")
    print(f"  AUGMENT_PER_IMAGE: {AUGMENT_PER_IMAGE}")
    print(f"  Output MinIO     : {PREPROC_PREFIX}/{{label}}/")

    # ── Resume: load set ảnh gốc đã xử lý ───────────────────────
    # Some older/partial documents may miss 'source_filename'. Use .get() and
    # filter out None to avoid KeyError when building the resume set.
    done_set = set()
    for d in col_transformed.find({"is_augmented": False}, {"filename": 1, "_id": 0}):
        fn = d.get("filename")
        if fn:
            done_set.add(os.path.splitext(fn)[0])
    print(f"  Đã xử lý trước : {len(done_set):,} → skip")

    stats = {"ok": 0, "aug": 0, "error": 0, "skip": len(done_set)}

    for i, doc in enumerate(col_integrated.find({}, {"_id": 0}), 1):
        fname = doc.get("filename", "")
        label = doc.get("label", "unknown")
        stem  = os.path.splitext(fname)[0]

        # ── RESUME ───────────────────────────────────────────────
        # done_set contains stems (filename without extension) of already
        # transformed standard images. Compare by stem.
        if stem in done_set:
            continue

        raw_obj = doc.get("object_name", f"{RAW_PREFIX}/{fname}")

        # ── Tải ảnh gốc từ MinIO raw/ ────────────────────────────
        try:
            img_rgb = load_rgb_from_minio(minio, raw_obj)
        except Exception as e:
            print(f"\n  [!] Tải {fname}: {e}")
            stats["error"] += 1
            continue

        orig_h, orig_w = img_rgb.shape[:2]

        # ── BƯỚC 1: standard_pipeline (giống test2.py) ──────────
        img_standard = STANDARD_PIPELINE(image=img_rgb)["image"]
        # img_standard: RGB uint8 640×640, đã qua CLAHE

        # ── Normalization stats ───────────────────────────────────
        norm_std = compute_norm_stats(img_standard)

        # ── Upload ảnh standard vào MinIO ────────────────────────
        main_obj = f"{PREPROC_PREFIX}/{label}/{stem}.png"
        try:
            if not upload_png(minio, main_obj, rgb_to_png_bytes(img_standard)):
                raise RuntimeError("put_object False")
        except Exception as e:
            print(f"\n  [!] Upload {main_obj}: {e}")
            stats["error"] += 1
            continue

        # ── Lưu doc ảnh standard ─────────────────────────────────
        try:
            col_transformed.insert_one(build_doc(
                fname_out=f"{stem}.png",
                obj_name=main_obj,
                label=label,
                orig_w=orig_w, orig_h=orig_h,
                norm=norm_std,
                is_aug=False, aug_idx=None,
            ))
            stats["ok"] += 1
        except Exception as e:
            print(f"\n  [!] MongoDB main {fname}: {e}")
            stats["error"] += 1
            continue

        # ── BƯỚC 2: aug_pipeline (giống test2.py) ────────────────
        if AUGMENT_ENABLED:
            for j in range(1, AUGMENT_PER_IMAGE + 1):
                aug_obj = f"{PREPROC_PREFIX}/{label}/{stem}_aug{j}.png"

                # Resume augmented: check transformed documents by augmented
                # filename (stem_aug{j}.png) and aug_index
                if col_transformed.find_one(
                        {"filename": f"{stem}_aug{j}.png", "aug_index": j},
                        {"_id": 1}):
                    continue

                # Áp dụng aug lên img_standard (đã chuẩn 640×640)
                img_aug  = AUG_PIPELINE(image=img_standard)["image"]
                norm_aug = compute_norm_stats(img_aug)

                try:
                    if not upload_png(minio, aug_obj, rgb_to_png_bytes(img_aug)):
                        raise RuntimeError("put_object False")
                    col_transformed.insert_one(build_doc(
                        fname_out=f"{stem}_aug{j}.png",
                        obj_name=aug_obj,
                        label=label,
                        orig_w=orig_w, orig_h=orig_h,
                        norm=norm_aug,
                        is_aug=True, aug_idx=j,
                    ))
                    stats["aug"] += 1
                except Exception:
                    pass

        if i % 100 == 0:
            print(
                f"  [{i}/{total}] OK={stats['ok']} "
                f"Aug={stats['aug']} Err={stats['error']}",
                end="\r",
            )

    print(f"""
[Transformation] HOÀN THÀNH  →  images_transformed
  ┌──────────────────────────────────────────────────┐
  │  Đã transform  : {stats['ok']:>6,} ảnh standard              │
  │  Augmentation  : {stats['aug']:>6,} ảnh (×{AUGMENT_PER_IMAGE})                │
  │  Bỏ qua (done) : {stats['skip']:>6,}                          │
  │  Lỗi           : {stats['error']:>6,}                          │
  └──────────────────────────────────────────────────┘
  Pipeline: LongestMaxSize→PadGray→CLAHE + GaussBlur→HFlip→BC
""")
    return stats


# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 3 – DATA TRANSFORMATION (Albumentations)")
    print("  standard: LongestMaxSize + PadIfNeeded(gray) + CLAHE")
    print("  aug     : GaussianBlur + HFlip + BrightnessContrast")
    print("=" * 60)
    minio = MinioClient()
    mongo = MongoDBClient()
    run_transformation(minio, mongo)