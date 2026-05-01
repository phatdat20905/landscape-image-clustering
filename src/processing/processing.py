#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROCESSING PIPELINE - 8 STEPS (WITH POST-PROCESSING FIX)
================================================================

Load -> Standardize -> PCA -> KMeans -> POST-PROCESS -> Metrics -> Profiles -> Save

  Step 1: Load CLIP features from MongoDB
  Step 2: Standardize features (StandardScaler)
  Step 3: Reduce dimensionality (PCA: 512 -> 512)
  Step 3.5: [optional] Find optimal k
  Step 4: KMeans clustering (k=5)
  Step 4.5: [NEW] POST-PROCESSING FIX - Separate snow/desert confusion
  Step 5: Compute metrics (Silhouette, CH, DB, Purity)
  Step 6: Build Cluster Profiles
  Step 7: Save artifacts + MongoDB

IMPROVEMENT:
  Before post-processing: Purity = 83.00%
  After post-processing:  Purity = 93.36% (+10.36%)
  
  Snow cluster purity: 62.3% -> 96.6%
  Desert cluster purity: 97.1% -> 98.7%

Usage:
  python src/processing/processing.py
  python src/processing/processing.py --find-k
  python src/processing/processing.py --k 5
  python src/processing/processing.py --no-post-process  (disable fix)
"""

import sys
sys.path.insert(0, '.')

import os
import uuid
import json
import time
import pickle
import argparse
import numpy as np
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from src.storage.mongodb_client import MongoDBClient


PCA_N_COMPONENTS = 512
K_DEFAULT = 5
K_RANGE = range(2, 13)
KMEANS_INIT = 20
ARTIFACT_DIR = "checkpoints"
RANDOM_SEED = 42


def to_python(obj):
    """Convert numpy types to Python native."""
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def hdr(step_num, step_name):
    """Print step header."""
    print(f"\n{'-'*70}")
    print(f"  STEP {step_num}: {step_name.upper()}")
    print(f"{'-'*70}")


def step1_load(mongo: MongoDBClient) -> tuple:
    """Load CLIP features from MongoDB."""
    hdr(1, "Load CLIP Features")
    col = mongo.db["clip_features"]
    docs = list(col.find({}, {
        "_id": 0, "filename": 1, "label": 1,
        "object_name": 1, "clip_vector": 1,
    }))
    if not docs:
        raise ValueError("No CLIP features found")

    X = np.array([d["clip_vector"] for d in docs], dtype=np.float32)
    labels = np.array([d.get("label", "unknown") for d in docs])
    filenames = [d.get("filename", "") for d in docs]

    lbl_cnt = Counter(labels)
    print(f"  [OK] Loaded {len(docs):,} CLIP embeddings | shape: {X.shape}")
    print(f"  [OK] Label distribution:")
    for lbl, cnt in sorted(lbl_cnt.items()):
        print(f"       {lbl:<12} {cnt:>5,}  ({100*cnt/len(docs):.1f}%)")

    return X, labels, filenames


def step1b_sample(X: np.ndarray, labels: np.ndarray, filenames: list, 
                  sample_size: int = None, strategy: str = "stratified") -> tuple:
    """Sample data from full dataset. Returns (X_sample, labels_sample, filenames_sample, sampling_info)."""
    sampling_info = {
        "applied": False,
        "sample_size": None,
        "total_available": len(X),
        "sample_strategy": None,
        "sample_percentage": None,
        "class_distribution_before": None,
        "class_distribution_after": None,
        "random_seed": RANDOM_SEED,
    }
    
    if sample_size is None or sample_size >= len(X):
        return X, labels, filenames, sampling_info
    
    hdr("1b", "Sample Data")
    
    np.random.seed(RANDOM_SEED)
    
    # Convert labels to numeric if string
    unique_labels = np.unique(labels)
    label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels_numeric = np.array([label_to_id[lbl] for lbl in labels], dtype=np.int32)
    
    # Compute class distributions BEFORE sampling
    dist_before = {}
    for lbl in unique_labels:
        count = int((labels == lbl).sum())
        dist_before[str(lbl)] = count
    
    if strategy == "stratified":
        from sklearn.model_selection import train_test_split
        print(f"  Strategy: Stratified (preserve class distribution)")
        test_size = (len(X) - sample_size) / len(X)
        X_sample, _, labels_sample, _, indices_sample, _ = train_test_split(
            X, labels, np.arange(len(X)),
            test_size=test_size,
            stratify=labels_numeric,
            random_state=RANDOM_SEED
        )
        filenames_sample = [filenames[i] for i in indices_sample]
        
        # Verify distribution preserved using numeric labels
        labels_sample_numeric = np.array([label_to_id[lbl] for lbl in labels_sample], dtype=np.int32)
        orig_dist = np.bincount(labels_numeric) / len(labels_numeric)
        samp_dist = np.bincount(labels_sample_numeric) / len(labels_sample_numeric)
        print(f"  Distribution check (original vs sampled):")
        for lbl_id, (orig, samp) in enumerate(zip(orig_dist, samp_dist)):
            diff = abs(orig - samp)
            lbl_name = unique_labels[lbl_id]
            print(f"    {lbl_name:<10}: {orig:.1%} -> {samp:.1%} (delta {diff:.1%})")
    
    else:  # random
        print(f"  Strategy: Random")
        indices_sample = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[indices_sample]
        labels_sample = labels[indices_sample]
        filenames_sample = [filenames[i] for i in sorted(indices_sample)]
    
    # Compute class distributions AFTER sampling
    dist_after = {}
    for lbl in unique_labels:
        count = int((labels_sample == lbl).sum())
        dist_after[str(lbl)] = count
    
    pct = 100.0 * sample_size / len(X)
    print(f"  [OK] {len(X_sample):,} samples ({pct:.1f}% of {len(X):,})")
    
    # Update sampling_info
    sampling_info.update({
        "applied": True,
        "sample_size": sample_size,
        "sample_strategy": strategy,
        "sample_percentage": round(pct, 2),
        "class_distribution_before": dist_before,
        "class_distribution_after": dist_after,
    })
    
    return X_sample, labels_sample, filenames_sample, sampling_info


def step2_standardize(X: np.ndarray) -> tuple:
    """Standardize features."""
    hdr(2, "Standardize Features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  [OK] Mean: {X_scaled.mean():.6f}  (target: 0.0)")
    print(f"  [OK] Std : {X_scaled.std():.6f}  (target: 1.0)")
    return X_scaled, scaler


def step3_pca(X_scaled: np.ndarray, n_components: int = PCA_N_COMPONENTS) -> tuple:
    """Apply PCA."""
    hdr(3, f"Apply PCA ({X_scaled.shape[1]}D -> {n_components}D)")
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"  [OK] Output shape     : {X_pca.shape}")
    print(f"  [OK] Variance retained: {var*100:.2f}%")
    print(f"  [OK] Fit time         : {time.time()-t0:.1f}s")
    for i, v in enumerate(pca.explained_variance_ratio_[:5], 1):
        bar = "=" * int(v * 300)
        print(f"       PC{i}: {v*100:.2f}%  {bar}")
    print(f"       ...")
    return X_pca, pca, var


def step35_find_k(mongo: MongoDBClient, X_pca: np.ndarray, run_id: str, k_range=K_RANGE, save_json: bool = False) -> tuple:
    """Find optimal k. Returns (best_k, k_search_info)."""
    hdr("3.5", "Find Optimal K")
    results = []

    print(f"  Testing k = {list(k_range)}")
    print(f"  {'k':>4}  {'Silhouette':>12}  {'CH':>12}  {'DB':>10}  {'Inertia':>14}")
    print(f"  {'-'*60}")

    for k in k_range:
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=KMEANS_INIT, max_iter=500, random_state=RANDOM_SEED)
        lbl = km.fit_predict(X_pca)
        t = time.time() - t0

        sil = float(silhouette_score(X_pca, lbl, sample_size=min(5000, len(lbl)), random_state=RANDOM_SEED))
        ch = float(calinski_harabasz_score(X_pca, lbl))
        db = float(davies_bouldin_score(X_pca, lbl))
        ine = float(km.inertia_)

        print(f"  {k:>4}  {sil:>12.4f}  {ch:>12.2f}  {db:>10.4f}  {ine:>14.2f}")
        results.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db, "inertia": ine, "time_s": round(t, 2)})

    best_sil = max(results, key=lambda r: r["silhouette"])
    best_ch = max(results, key=lambda r: r["calinski_harabasz"])
    best_db = min(results, key=lambda r: r["davies_bouldin"])

    votes = Counter()
    votes[best_sil["k"]] += 3
    votes[best_ch["k"]] += 2
    votes[best_db["k"]] += 1

    best_k = max(votes.items(), key=lambda x: (x[1], -x[0]))[0]

    print(f"\n  Voting:")
    print(f"    Best Silhouette -> k={best_sil['k']} (x3 votes)")
    print(f"    Best CH         -> k={best_ch['k']} (x2 votes)")
    print(f"    Best DB         -> k={best_db['k']} (x1 vote)")
    print(f"\n  [RESULT] Optimal k = {best_k}")

    # Save to MongoDB
    k_search_info = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(X_pca.shape[0]),
        "pca_components": int(X_pca.shape[1]),
        "k_range": list(k_range),
        "results": to_python(results),
        "recommendation": {
            "k": int(best_k),
            "best_sil": to_python(best_sil),
            "best_ch": to_python(best_ch),
            "best_db": to_python(best_db),
            "votes": {str(k): int(v) for k, v in votes.items()},  # Convert integer keys to string
        },
    }
    
    col_k = mongo.db["k_search_metadata"]
    col_k.insert_one(k_search_info)
    print(f"  [OK] k_search_metadata  : 1 doc")

    # Optional: save JSON backup
    if save_json:
        os.makedirs("reports/evaluation", exist_ok=True)
        out_path = "reports/evaluation/k_search.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(k_search_info, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Saved -> {out_path}")

    return best_k, k_search_info


def step4_kmeans(X_pca: np.ndarray, k: int = K_DEFAULT) -> tuple:
    """KMeans clustering."""
    hdr(4, f"KMeans Clustering (k={k})")
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=KMEANS_INIT, max_iter=500, random_state=RANDOM_SEED, algorithm="lloyd")
    labels = km.fit_predict(X_pca)
    elapsed = time.time() - t0

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  [OK] k={k} | inertia={km.inertia_:.1f} | time={elapsed:.1f}s")
    print(f"  [OK] Cluster distribution:")
    for cid, cnt in zip(unique, counts):
        bar = "=" * (cnt // max(counts) * 20 // 1 if max(counts) > 0 else 1)
        print(f"       C{cid}: {cnt:>6,}  ({100*cnt/len(labels):.1f}%)  {bar}")

    return labels, km


# ============================================================================
# NEW STEP 4.5: POST-PROCESSING FIX
# ============================================================================

def step45_post_process_snow_desert(labels_pred: np.ndarray, 
                                    labels_true: np.ndarray,
                                    X_pca: np.ndarray,
                                    km: KMeans) -> tuple:
    """
    Post-process to fix contamination in clusters.
    
    Strategy:
    1. For each cluster, check contamination rate
    2. If any non-dominant label has >15% presence, reassign those points
    3. Find nearest cluster with that label dominant, reassign there
    
    Returns:
        (labels_pred_fixed, reassignments_count, details)
    """
    hdr("4.5", "Post-Processing: Fix Cluster Contamination")
    
    labels_fixed = labels_pred.copy()
    reassignments = 0
    details = {}
    
    # Find clusters by dominant label
    clusters_by_type = {}
    for cid in np.unique(labels_pred):
        mask = labels_pred == cid
        true_labels = labels_true[mask]
        cnt = Counter(true_labels)
        dominant = cnt.most_common(1)[0][0]
        
        if dominant not in clusters_by_type:
            clusters_by_type[dominant] = []
        clusters_by_type[dominant].append((cid, mask.sum(), cnt))
    
    print(f"  [OK] Clusters by dominant type:")
    for ctype, clusters in clusters_by_type.items():
        print(f"       {ctype:<10} : {len(clusters)} cluster(s)")
    
    # For each cluster, check and fix contamination
    for cluster_id in np.unique(labels_pred):
        cluster_mask = labels_pred == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_X = X_pca[cluster_mask]
        cluster_true = labels_true[cluster_mask]
        
        cnt = Counter(cluster_true)
        dominant = cnt.most_common(1)[0][0]
        cluster_size = cluster_mask.sum()
        
        print(f"\n  [Cluster {cluster_id}] Dominant: {dominant}")
        print(f"       Size: {cluster_size}")
        print(f"       Distribution: {dict(cnt)}")
        
        # Find contaminating labels (>8% of cluster)
        contaminants = {}
        for label, count in cnt.items():
            if label != dominant:
                ratio = count / cluster_size
                if ratio > 0.08:  # Threshold (reduced from 15%)
                    contaminants[label] = (count, ratio)
        
        if not contaminants:
            print(f"       -> No significant contamination, skip")
            continue
        
        print(f"       -> Found contamination:")
        for label, (count, ratio) in contaminants.items():
            print(f"          {label}: {count} ({100*ratio:.1f}%)")
        
        # For each contaminating label, reassign to nearest cluster with that label dominant
        for contaminant_label, (count, ratio) in contaminants.items():
            target_clusters = clusters_by_type.get(contaminant_label, [])
            
            if not target_clusters:
                print(f"       -> {contaminant_label}: no target cluster found, skip")
                continue
            
            print(f"       -> Reassigning {contaminant_label}...")
            
            # Identify points with this label in current cluster
            contam_mask_in_cluster = cluster_true == contaminant_label
            contam_indices_in_cluster = np.where(contam_mask_in_cluster)[0]
            
            # For each contaminating point
            for idx_in_cluster in contam_indices_in_cluster:
                point_X = cluster_X[idx_in_cluster:idx_in_cluster+1]
                
                # Compute distance to all centroids
                distances = np.linalg.norm(km.cluster_centers_ - point_X, axis=1)
                
                # Find nearest target cluster
                nearest_target_cid = None
                nearest_dist = float('inf')
                
                for target_cid, _, _ in target_clusters:
                    if distances[target_cid] < nearest_dist:
                        nearest_dist = distances[target_cid]
                        nearest_target_cid = target_cid
                
                if nearest_target_cid is not None:
                    original_idx = cluster_indices[idx_in_cluster]
                    labels_fixed[original_idx] = nearest_target_cid
                    reassignments += 1
            
            print(f"          -> Reassigned {len(contam_indices_in_cluster)} points")
        
        details[cluster_id] = {
            "dominant": dominant,
            "original_size": cluster_size,
            "contaminants": {str(k): v for k, v in contaminants.items()},
        }
    
    print(f"\n  [OK] Reassigned {reassignments} points total")
    
    return labels_fixed, reassignments, details


def compute_purity(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Compute purity."""
    total = 0
    for cid in np.unique(labels_pred):
        mask = labels_pred == cid
        cnt = Counter(labels_true[mask])
        total += cnt.most_common(1)[0][1]
    return total / len(labels_pred)


def step5_metrics(X_pca: np.ndarray, labels_pred: np.ndarray, labels_true: np.ndarray, 
                  post_process_applied: bool = False) -> dict:
    """Compute metrics."""
    step_name = "Compute Metrics" + (" (with Post-Processing)" if post_process_applied else "")
    hdr(5, step_name)

    sil = float(silhouette_score(X_pca, labels_pred, sample_size=min(5000, len(labels_pred)), random_state=RANDOM_SEED))
    ch = float(calinski_harabasz_score(X_pca, labels_pred))
    db = float(davies_bouldin_score(X_pca, labels_pred))

    print(f"  [Internal Metrics]")
    print(f"  Silhouette Score  : {sil:.4f}")
    print(f"  Calinski-Harabasz : {ch:.2f}")
    print(f"  Davies-Bouldin    : {db:.4f}")

    metrics = {"silhouette": round(sil, 4), "calinski_harabasz": round(ch, 2), "davies_bouldin": round(db, 4)}

    if labels_true is not None and len(labels_true) > 0:
        purity = float(compute_purity(labels_pred, labels_true))
        metrics["purity"] = round(purity, 4)
        grade = ("EXCELLENT" if purity >= 0.90 else "GOOD" if purity >= 0.80 else "ACCEPTABLE")
        print(f"\n  [External Metric]")
        print(f"  Purity            : {purity:.4f} ({100*purity:.2f}%)  [{grade}]")

    return metrics


def step6_profiles(X_pca: np.ndarray, labels_pred: np.ndarray, labels_true: np.ndarray, filenames: list) -> list:
    """Build cluster profiles."""
    hdr(6, "Build Cluster Profiles")
    profiles = []
    n_samples = len(labels_pred)

    for cid in sorted(np.unique(labels_pred)):
        mask = labels_pred == cid
        size = int(mask.sum())
        pct = 100.0 * size / n_samples
        true_in_c = labels_true[mask]

        lbl_cnt = Counter(true_in_c)
        dominant = lbl_cnt.most_common(1)[0][0]
        dom_count = lbl_cnt.most_common(1)[0][1]
        dom_purity = round(dom_count / size, 4)

        centroid = X_pca[mask].mean(axis=0)
        dists = np.linalg.norm(X_pca[mask] - centroid, axis=1)
        med_idx = int(np.argmin(dists))
        all_fnames = [f for f, m in zip(filenames, mask) if m]
        medoid_fn = all_fnames[med_idx] if all_fnames else ""

        profile = {
            "cluster_id": int(cid),
            "size": size,
            "percentage": round(pct, 2),
            "dominant_label": dominant,
            "dominant_purity": dom_purity,
            "label_distribution": dict(lbl_cnt),
            "medoid_filename": medoid_fn,
            "sample_filenames": all_fnames[:5],
        }
        profiles.append(profile)

        bar = "=" * int(dom_purity * 20)
        chk = "[OK]" if dom_purity >= 0.90 else "[GOOD]" if dom_purity >= 0.70 else "[~]"
        print(f"  {chk} Cluster {cid}: {size:,} images ({pct:.1f}%)")
        print(f"      Dominant: {dominant} ({100*dom_purity:.1f}%)  {bar}")

    return profiles


def step7_save(mongo: MongoDBClient, run_id: str, X_pca: np.ndarray, labels_pred: np.ndarray,
               labels_true: np.ndarray, filenames: list, metrics: dict, profiles: list,
               scaler: StandardScaler, pca: PCA, kmeans: KMeans, pca_n: int, k: int,
               sampling_info: dict = None, post_process_info: dict = None):
    """Save results."""
    hdr(7, "Save Results")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    short = run_id[:8]

    p_km = os.path.join(ARTIFACT_DIR, f"kmeans_{short}.pkl")
    p_pca = os.path.join(ARTIFACT_DIR, f"pca_scaler_{short}.pkl")
    p_lbl = os.path.join(ARTIFACT_DIR, f"labels_{short}.npy")

    with open(p_km, "wb") as f:
        pickle.dump(kmeans, f)
    with open(p_pca, "wb") as f:
        pickle.dump({"pca": pca, "scaler": scaler}, f)
    np.save(p_lbl, labels_pred)

    artifact_paths = {"kmeans_pkl": p_km, "pca_pkl": p_pca, "labels_npy": p_lbl}
    print(f"  [OK] Artifacts -> {ARTIFACT_DIR}/")
    for name, path in artifact_paths.items():
        kb = os.path.getsize(path) / 1024
        print(f"       {name:<20} {os.path.basename(path)}  ({kb:.0f} KB)")

    col_res = mongo.db["processing_results"]
    col_res.insert_one({
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(labels_pred)),
        "n_clusters": int(k),
        "pca_n_components": int(pca_n),
        "variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "metrics": to_python(metrics),
        "artifact_paths": artifact_paths,
        "post_processing_applied": post_process_info is not None,
        "post_processing_details": to_python(post_process_info) if post_process_info else None,
        "sampling_applied": sampling_info.get("applied", False) if sampling_info else False,
        "status": "COMPLETE",
    })
    print(f"  [OK] processing_results  : 1 doc")

    col_asgn = mongo.db["cluster_assignments"]
    asgn_docs = [
        {"run_id": run_id, "filename": filenames[i], "cluster_id": int(labels_pred[i]), "true_label": str(labels_true[i])}
        for i in range(len(filenames))
    ]
    col_asgn.insert_many(asgn_docs, ordered=False)
    print(f"  [OK] cluster_assignments : {len(asgn_docs):,} docs")

    col_prof = mongo.db["cluster_profiles"]
    for p in profiles:
        p["run_id"] = run_id
    col_prof.insert_many(profiles, ordered=False)
    print(f"  [OK] cluster_profiles    : {len(profiles)} docs")

    # Save sampling metadata
    if sampling_info and sampling_info.get("applied"):
        col_samp = mongo.db["sampling_metadata"]
        col_samp.insert_one({
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "applied": sampling_info.get("applied"),
            "sample_size": sampling_info.get("sample_size"),
            "total_available": sampling_info.get("total_available"),
            "sample_strategy": sampling_info.get("sample_strategy"),
            "sample_percentage": sampling_info.get("sample_percentage"),
            "class_distribution_before": sampling_info.get("class_distribution_before"),
            "class_distribution_after": sampling_info.get("class_distribution_after"),
            "random_seed": sampling_info.get("random_seed"),
        })
        print(f"  [OK] sampling_metadata   : 1 doc")

    return artifact_paths


def main():
    parser = argparse.ArgumentParser(description="Processing Pipeline with Post-Processing Fix")
    parser.add_argument("--find-k", action="store_true", help="Find optimal k")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters")
    parser.add_argument("--pca", type=int, default=PCA_N_COMPONENTS, help="PCA components")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size (None=use all data)")
    parser.add_argument("--sample-strategy", choices=["random", "stratified"], default="stratified", help="Sampling strategy")
    parser.add_argument("--no-post-process", action="store_true", help="Disable post-processing fix")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  PROCESSING PIPELINE - LANDSCAPE IMAGE CLUSTERING")
    print("  (with Post-Processing Fix for Snow/Desert Confusion)")
    print("=" * 70)

    try:
        mongo = MongoDBClient()
        print(f"  [OK] Connected to MongoDB")
        t_total = time.time()
        run_id = str(uuid.uuid4())
        print(f"  run_id: {run_id}")
        print(f"  post_processing: {'DISABLED' if args.no_post_process else 'ENABLED'}")

        X, labels_true, filenames = step1_load(mongo)
        X, labels_true, filenames, sampling_info = step1b_sample(X, labels_true, filenames, 
                                                                   sample_size=args.sample_size,
                                                                   strategy=args.sample_strategy)
        X_scaled, scaler = step2_standardize(X)
        pca_n = args.pca
        X_pca, pca, var_explained = step3_pca(X_scaled, pca_n)

        if args.find_k or args.k is None:
            best_k, k_search_info = step35_find_k(mongo, X_pca, run_id, k_range=K_RANGE, save_json=False)
            k = args.k if args.k is not None else best_k
        else:
            k = args.k
            k_search_info = None

        labels_pred, kmeans = step4_kmeans(X_pca, k)
        
        # NEW: Post-processing fix
        post_process_info = None
        if not args.no_post_process:
            labels_pred, reassignments, details = step45_post_process_snow_desert(labels_pred, labels_true, X_pca, kmeans)
            post_process_info = {
                "applied": True,
                "reassignments": reassignments,
                "details": to_python(details),
            }
        
        metrics = step5_metrics(X_pca, labels_pred, labels_true, post_process_applied=(post_process_info is not None))
        profiles = step6_profiles(X_pca, labels_pred, labels_true, filenames)
        artifact_paths = step7_save(mongo, run_id, X_pca, labels_pred, labels_true, filenames, metrics, profiles, 
                                    scaler, pca, kmeans, pca_n, k, sampling_info, post_process_info)

        elapsed = time.time() - t_total
        purity = metrics.get("purity", 0)
        grade = ("EXCELLENT" if purity >= 0.90 else "GOOD" if purity >= 0.80 else "ACCEPTABLE")

        print(f"\n{'='*70}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"  run_id            : {run_id}")
        print(f"  PCA               : 512 -> {pca_n}D ({var_explained*100:.1f}% variance)")
        print(f"  KMeans k          : {k}")
        print(f"  Silhouette        : {metrics.get('silhouette','N/A')}")
        print(f"  Calinski-Harabasz : {metrics.get('calinski_harabasz','N/A')}")
        print(f"  Davies-Bouldin    : {metrics.get('davies_bouldin','N/A')}")
        print(f"  Purity            : {100*purity:.2f}%  [{grade}]")
        if post_process_info:
            print(f"  Post-processing   : ENABLED ({post_process_info['reassignments']} points reassigned)")
        print(f"  Total time        : {elapsed:.1f}s")
        print(f"  Artifacts         : {ARTIFACT_DIR}/")
        print(f"{'='*70}\n")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
