#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEMO: CLUSTERING NEW IMAGES USING TRAINED MODELS
================================================

Load pre-trained models (KMeans + PCA + Scaler) from checkpoints/
Extract CLIP features for new images
Apply clustering WITHOUT saving to MongoDB

Usage:
  python src/processing/demo_clustering.py --model-id 06912ab1 --image-dir /path/to/images
  python src/processing/demo_clustering.py --model-id 06912ab1 --image-file /path/to/single/image.jpg
"""

import sys
sys.path.insert(0, '.')

import os
import glob
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[!] CLIP not installed. Install with: pip install openai-clip")

try:
    from src.storage.mongodb_client import MongoDBClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class ClusteringDemo:
    """Demo for clustering new images using trained models."""
    
    def __init__(self, model_id: str):
        """
        Initialize with trained model artifacts.
        
        Args:
            model_id: First 8 chars of run_id (e.g., "06912ab1")
        """
        self.model_id = model_id
        self.artifact_dir = "checkpoints"
        self.cluster_labels = {}  # Maps cluster_id -> label
        
        # Load models
        self._load_models()
        
        # Try to load cluster labels from MongoDB
        self._load_cluster_labels()
        
        # Load CLIP
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP not available. Install with: pip install openai-clip")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = self._load_clip()
    
    def _load_cluster_labels(self):
        """Load cluster dominant labels from MongoDB."""
        if not MONGODB_AVAILABLE:
            print(f"  [~] MongoDB not available, cluster labels unknown")
            return
        
        try:
            mongo = MongoDBClient()
            col_prof = mongo.db["cluster_profiles"]
            
            # Find profiles for this run_id
            profiles = list(col_prof.find(
                {"run_id": self.model_id[:8]},  # Try to match with model_id prefix
                {"cluster_id": 1, "dominant_label": 1, "_id": 0}
            ))
            
            if not profiles:
                # Try to get latest profiles (if run_id doesn't match model_id)
                profiles = list(col_prof.find(
                    {},
                    {"cluster_id": 1, "dominant_label": 1, "_id": 0}
                ).sort("_id", -1).limit(self.kmeans.n_clusters))
                profiles = sorted(profiles, key=lambda p: p["cluster_id"])
            
            for p in profiles:
                self.cluster_labels[int(p["cluster_id"])] = p["dominant_label"]
            
            if self.cluster_labels:
                print(f"  [OK] Cluster labels loaded from MongoDB:")
                for cid, label in sorted(self.cluster_labels.items()):
                    print(f"       C{cid} -> {label}")
            else:
                print(f"  [~] No cluster labels found")
        
        except Exception as e:
            print(f"  [~] Could not load cluster labels: {e}")
    
    def _load_models(self):
        """Load KMeans, PCA, and Scaler from pickle files."""
        print(f"\n[Models] Loading artifacts for model_id={self.model_id}")
        
        # KMeans model
        km_path = os.path.join(self.artifact_dir, f"kmeans_{self.model_id}.pkl")
        if not os.path.exists(km_path):
            raise FileNotFoundError(f"KMeans model not found: {km_path}")
        
        with open(km_path, "rb") as f:
            self.kmeans = pickle.load(f)
        print(f"  [OK] KMeans: {self.kmeans.n_clusters} clusters")
        
        # PCA + Scaler
        pca_path = os.path.join(self.artifact_dir, f"pca_scaler_{self.model_id}.pkl")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        
        with open(pca_path, "rb") as f:
            pca_data = pickle.load(f)
            self.pca = pca_data["pca"]
            self.scaler = pca_data["scaler"]
        print(f"  [OK] PCA: {self.pca.n_components_} components")
        print(f"  [OK] Scaler: fitted")
    
    def _load_clip(self):
        """Load CLIP model."""
        print(f"\n[CLIP] Loading model on {self.device}...")
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        model.eval()
        print(f"  [OK] CLIP ViT-B/32 (512-dim embeddings)")
        return model, preprocess
    
    def extract_clip_features(self, images: list) -> np.ndarray:
        """
        Extract CLIP features from PIL images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Array of shape (N, 512) with L2-normalized embeddings
        """
        print(f"\n[Extract] Extracting CLIP features for {len(images)} images...")
        
        features_list = []
        
        for i, img in enumerate(images):
            try:
                # Preprocess and extract
                tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    feat = self.clip_model.encode_image(tensor)  # (1, 512)
                    feat = feat.cpu().numpy()[0]
                
                # L2 normalize
                feat = feat / (np.linalg.norm(feat) + 1e-8)
                features_list.append(feat)
                
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(images)}] extracted", end="\r")
            
            except Exception as e:
                print(f"\n  [!] Error extracting image {i}: {e}")
                continue
        
        print(f"  [OK] Extracted {len(features_list)} features")
        return np.array(features_list, dtype=np.float32)
    
    def preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize and apply PCA to raw CLIP features.
        
        Args:
            X: Raw CLIP features (N, 512)
            
        Returns:
            PCA-transformed features (N, n_components)
        """
        print(f"\n[Preprocess] Standardizing and applying PCA...")
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        
        # PCA
        X_pca = self.pca.transform(X_scaled)
        
        print(f"  [OK] Input shape: {X.shape}")
        print(f"  [OK] Output shape: {X_pca.shape}")
        
        return X_pca
    
    def predict_clusters(self, X_pca: np.ndarray) -> tuple:
        """
        Predict cluster assignments for preprocessed features.
        
        Args:
            X_pca: PCA-transformed features (N, n_components)
            
        Returns:
            (labels, distances) where labels are cluster IDs (0-k-1)
                                 distances are distances to assigned centroids
        """
        print(f"\n[KMeans] Predicting clusters...")
        
        labels = self.kmeans.predict(X_pca)
        distances = np.min(self.kmeans.transform(X_pca), axis=1)
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  [OK] Cluster distribution:")
        for cid, cnt in zip(unique, counts):
            pct = 100.0 * cnt / len(labels)
            print(f"       C{cid}: {cnt:>6} images ({pct:>5.1f}%)")
        
        return labels, distances
    
    def cluster_images_from_dir(self, image_dir: str) -> dict:
        """
        Cluster all images in a directory.
        
        Args:
            image_dir: Path to directory containing images
            
        Returns:
            Dict with results: {"filenames", "labels", "distances", "cluster_assignments"}
        """
        print(f"\n[Images] Loading images from {image_dir}")
        
        # Find image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext), recursive=False))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper()), recursive=False))
        
        image_files = list(set(image_files))  # Remove duplicates
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"  [OK] Found {len(image_files)} images")
        
        # Load and convert to PIL
        pil_images = []
        valid_files = []
        
        for fpath in sorted(image_files):
            try:
                img = Image.open(fpath).convert("RGB")
                pil_images.append(img)
                valid_files.append(os.path.basename(fpath))
            except Exception as e:
                print(f"  [!] Error loading {fpath}: {e}")
        
        print(f"  [OK] Loaded {len(pil_images)} images successfully")
        
        # Extract features
        X = self.extract_clip_features(pil_images)
        X_pca = self.preprocess_features(X)
        labels, distances = self.predict_clusters(X_pca)
        
        # Build results
        results = {
            "filenames": valid_files,
            "labels": labels,
            "distances": distances,
            "cluster_assignments": self._build_cluster_assignments(valid_files, labels, distances),
        }
        
        return results
    
    def cluster_single_image(self, image_path: str) -> dict:
        """
        Cluster a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with result: {"filename", "label", "distance", "confidence"}
        """
        print(f"\n[Image] Loading {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image: {e}")
        
        # Extract features
        X = self.extract_clip_features([img])
        X_pca = self.preprocess_features(X)
        labels, distances = self.predict_clusters(X_pca)
        
        label = int(labels[0])
        distance = float(distances[0])
        
        # Compute confidence (inverse of distance to nearest centroid)
        # closer = more confident
        confidence = 1.0 / (1.0 + distance)
        
        # Get dominant label for this cluster
        dominant_label = self.cluster_labels.get(label, "unknown")
        
        result = {
            "filename": os.path.basename(image_path),
            "cluster_id": label,
            "cluster_label": dominant_label,
            "distance_to_centroid": round(distance, 4),
            "confidence": round(confidence, 4),
        }
        
        return result
    
    @staticmethod
    def _build_cluster_assignments(filenames: list, labels: np.ndarray, distances: np.ndarray) -> dict:
        """Build cluster assignment dict grouped by cluster."""
        assignments = {}
        for cid in np.unique(labels):
            mask = labels == cid
            cluster_files = [f for f, m in zip(filenames, mask) if m]
            cluster_dists = distances[mask]
            
            assignments[int(cid)] = {
                "count": int(mask.sum()),
                "files": cluster_files,
                "distances": [round(float(d), 4) for d in cluster_dists],
            }
        
        return assignments


def print_results(results: dict, result_type: str = "batch", cluster_labels: dict = None):
    """Pretty print results."""
    print(f"\n{'='*70}")
    print(f"  CLUSTERING RESULTS")
    print(f"{'='*70}")
    
    if result_type == "single":
        r = results
        cluster_label = cluster_labels.get(r['cluster_id'], "unknown") if cluster_labels else "unknown"
        print(f"  Image           : {r['filename']}")
        print(f"  Cluster ID      : {r['cluster_id']}")
        print(f"  Cluster Label   : {cluster_label}")
        print(f"  Distance        : {r['distance_to_centroid']}")
        print(f"  Confidence      : {r['confidence']:.1%}")
    
    else:  # batch
        print(f"  Total images    : {len(results['filenames'])}")
        print(f"  Cluster distribution:")
        
        assignments = results["cluster_assignments"]
        for cid in sorted(assignments.keys()):
            info = assignments[cid]
            cluster_label = cluster_labels.get(cid, "unknown") if cluster_labels else "unknown"
            pct = 100.0 * info["count"] / len(results["filenames"])
            print(f"    C{cid} ({cluster_label:<8}): {info['count']:>6} images ({pct:>5.1f}%)")
        
        print(f"\n  Top 5 images by cluster:")
        for cid in sorted(assignments.keys()):
            info = assignments[cid]
            print(f"\n    Cluster {cid}:")
            for fname in info["files"][:5]:
                print(f"      - {fname}")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Cluster new images using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cluster all images in a directory
  python src/processing/demo_clustering.py --model-id 06912ab1 --image-dir ./test_images
  
  # Cluster a single image
  python src/processing/demo_clustering.py --model-id 06912ab1 --image-file ./test.jpg
        """
    )
    
    parser.add_argument("--model-id", type=str, required=True, 
                        help="Model ID (first 8 chars of run_id, e.g., 06912ab1)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing images to cluster")
    parser.add_argument("--image-file", type=str, default=None,
                        help="Single image file to cluster")
    
    args = parser.parse_args()
    
    if not args.image_dir and not args.image_file:
        parser.error("Must provide either --image-dir or --image-file")
    
    if args.image_dir and args.image_file:
        parser.error("Provide only one of --image-dir or --image-file")
    
    print("\n" + "=" * 70)
    print("  CLUSTERING DEMO - NEW IMAGES")
    print("=" * 70)
    
    try:
        # Initialize demo
        demo = ClusteringDemo(model_id=args.model_id)
        
        # Cluster images
        if args.image_dir:
            results = demo.cluster_images_from_dir(args.image_dir)
            print_results(results, result_type="batch", cluster_labels=demo.cluster_labels)
        
        else:  # single image
            result = demo.cluster_single_image(args.image_file)
            print_results(result, result_type="single", cluster_labels=demo.cluster_labels)
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
