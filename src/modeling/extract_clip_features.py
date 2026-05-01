#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP Feature Extraction for Landscape Images
=============================================

Purpose: Extract CLIP embeddings instead of ResNet50
Expected improvement: purity 33% → 50-55%
Time: ~30 minutes total

CLIP trained on 400M image-caption pairs
Understands semantic concepts: "mountain", "desert", "sea", etc.
Better generalization for landscape classification

Usage:
  python src/modeling/extract_clip_features.py
"""

import sys
sys.path.insert(0, '.')

import io
import torch
import numpy as np
from datetime import datetime
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError
from PIL import Image
from collections import Counter

from src.storage.minio_client import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET

# Try to import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[!] CLIP not installed. Install with: pip install openai-clip")


def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load CLIP model."""
    print(f"\n[CLIP] Loading model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"[CLIP] Model loaded: ViT-B/32 (512-dim embeddings)")
    return model, preprocess


def extract_clip_features(minio: MinioClient, mongo: MongoDBClient):
    """Extract CLIP features for all landscape images."""
    
    if not CLIP_AVAILABLE:
        print("[!] CLIP not installed.")
        print("    Install: pip install openai-clip")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_model(device)
    
    col_image_features = mongo.db["images_transformed"]
    col_clip_features = mongo.db["clip_features"]
    
    # Create index
    try:
        col_clip_features.create_index(
            [("object_name", ASCENDING)],
            unique=True, name="obj_unique",
        )
        col_clip_features.create_index([("label", ASCENDING)])
        col_clip_features.create_index([("filename", ASCENDING)])
    except Exception:
        pass
    
    # Count existing
    total = col_image_features.count_documents({})
    done = col_clip_features.count_documents({})
    print(f"\n[Extract] Total images: {total:,} | Already extracted: {done:,}")
    
    # Resume
    extracted_set = set(
        d["object_name"] for d in
        col_clip_features.find({}, {"object_name": 1, "_id": 0})
    )
    
    # Only process original images (not augmented)
    all_docs = list(col_image_features.find({"is_augmented": False}))
    pending = [d for d in all_docs if d.get("object_name", "") not in extracted_set]
    
    print(f"[Extract] Processing ONLY original images (not augmented)")
    print(f"[Extract] Total original images: {len(all_docs):,} | Pending: {len(pending):,}")
    
    stats = {
        "ok": 0,
        "error": 0,
        "skipped": len(all_docs) - len(pending),
    }
    print(f"  Need to extract: {len(pending):,} images\n")
    
    # Batch processing
    BATCH_SIZE = 32
    
    for b_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[b_start: b_start + BATCH_SIZE]
        pil_imgs = []
        valid_docs = []
        
        # Load images from MinIO
        for doc in batch:
            obj = doc.get("object_name", "")
            if not obj:
                stats["error"] += 1
                continue
            try:
                resp = minio.client.get_object(MINIO_BUCKET, obj)
                raw = resp.read()
                resp.close()
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                pil_imgs.append(pil)
                valid_docs.append(doc)
            except Exception as e:
                print(f"\n  [!] MinIO error {obj}: {e}")
                stats["error"] += 1
        
        if not pil_imgs:
            continue
        
        # Preprocess and stack
        try:
            tensors = torch.stack([preprocess(img).to(device) for img in pil_imgs])
            
            # Extract CLIP features
            with torch.no_grad():
                features = model.encode_image(tensors)  # (N, 512)
                features = features.cpu().numpy()
            
            # Normalize features (CLIP already normalized, but ensure consistency)
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            # Save to MongoDB
            for i, (feat, doc) in enumerate(zip(features, valid_docs)):
                clip_doc = {
                    "filename": doc.get("filename", ""),
                    "label": doc.get("label", "unknown"),
                    "object_name": doc.get("object_name", ""),
                    "is_augmented": doc.get("is_augmented", False),
                    "aug_index": doc.get("aug_index"),
                    "width": doc.get("width", 224),
                    "height": doc.get("height", 224),
                    
                    # CLIP embeddings - 512 dims
                    "clip_vector": feat.astype(np.float32).tolist(),  # list[float32] 512
                    "clip_dim": 512,
                    "extracted_at": datetime.now().strftime("%Y-%m-%d"),
                }
                
                try:
                    col_clip_features.insert_one(clip_doc)
                    stats["ok"] += 1
                except DuplicateKeyError:
                    stats["skipped"] += 1
                except Exception as e:
                    print(f"\n  [!] MongoDB error: {e}")
                    stats["error"] += 1
        
        except Exception as e:
            print(f"\n  [!] Batch processing error: {e}")
            stats["error"] += len(pil_imgs)
        
        done_count = b_start + len(batch)
        print(
            f"  [{done_count}/{len(pending)}] OK={stats['ok']} Skip={stats['skipped']} Err={stats['error']}",
            end="\r"
        )
    
    print(f"""
[Extract] CLIP COMPLETE → clip_features collection
  ┌────────────────────────────────────────────────────┐
  │  Extracted    : {stats['ok']:>6,} new embeddings          │
  │  Skipped (done): {stats['skipped']:>6,}                       │
  │  Errors       : {stats['error']:>6,}                         │
  ├────────────────────────────────────────────────────┤
  │  Model        : CLIP ViT-B/32                      │
  │  Dimension    : 512 (vs ResNet50's 2048)          │
  │  Type         : Vision-Language embeddings         │
  │  Storage      : MongoDB clip_features              │
  │  Normalization: L2-normalized                      │
  └────────────────────────────────────────────────────┘
""")
    
    return stats


# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CLIP FEATURE EXTRACTION")
    print("  Vision-Language Model Embeddings for Landscape Images")
    print("=" * 60)
    
    if not CLIP_AVAILABLE:
        print("\n[!] CLIP not installed.")
        print("    Install with: pip install openai-clip")
        print("\n    Then run: python src/modeling/extract_clip_features.py")
        sys.exit(1)
    
    minio = MinioClient()
    mongo = MongoDBClient()
    
    try:
        stats = extract_clip_features(minio, mongo)
        if stats:
            print("\n[OK] CLIP feature extraction complete!")
            print("\nNext step: Run clustering with CLIP features")
            print("  python src/processing/processing_complete.py --features clip --k 5")
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
