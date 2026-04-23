#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
+============================================================================+
|                          EVALUATION MODULE                                 |
|         Đánh giá kết quả Clustering từ complete_pipeline.py               |
+============================================================================+

Đọc kết quả từ MongoDB (do complete_pipeline.py sinh ra) và tạo:

  PHẦN 1 – METRICS TỔNG HỢP
    1.1  Bảng tóm tắt metrics (Silhouette, CH, DB, Purity)
    1.2  So sánh k_search: metrics vs k (Elbow curve)

  PHẦN 2 – CLUSTER VISUALIZATION
    2.1  2D PCA scatter (màu theo cluster_id)
    2.2  2D PCA scatter (màu theo ground-truth label)
    2.3  Confusion matrix: Cluster × Label gốc

  PHẦN 3 – CLUSTER PROFILES
    3.1  Bar chart: cluster size + purity
    3.2  Stacked bar: label distribution trong mỗi cluster
    3.3  Silhouette plot chi tiết từng cluster

  PHẦN 4 – SAMPLING ANALYSIS
    4.1  So sánh phân bố trước/sau sampling (nếu có)

  PHẦN 5 – GALLERY ẢNH
    5.1  Ảnh medoid (đại diện) mỗi cluster từ MinIO

Cách dùng:
  python src/processing/evaluation.py
  python src/processing/evaluation.py --run-id <uuid>
  python src/processing/evaluation.py --latest          # run mới nhất (mặc định)
  python src/processing/evaluation.py --no-gallery      # bỏ qua tải ảnh MinIO
"""

import sys
sys.path.insert(0, '.')

import os
import io
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from datetime import datetime
from PIL import Image

from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA

from src.storage.mongodb_client import MongoDBClient

try:
    from src.storage.minio_client import MinioClient
    from configs.config import MINIO_BUCKET
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────
REPORT_DIR = "reports/evaluation"
KEYWORDS   = ["mountain", "forest", "sea", "desert", "snow"]

KW_COLORS = {
    "mountain": "#5B8DB8",
    "forest":   "#4CAF50",
    "sea":      "#2EC4B6",
    "desert":   "#E8A838",
    "snow":     "#A0C4D8",
}
CLUSTER_CMAP = plt.cm.get_cmap("tab10")

plt.rcParams.update({
    "figure.dpi":          130,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.22,
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
})


# ================================================================
#  LOAD DỮ LIỆU TỪ MONGODB
# ================================================================

def get_run_id(mongo: MongoDBClient, run_id: str = None) -> str:
    """Trả về run_id (mới nhất nếu không chỉ định)."""
    col = mongo.db["processing_results"]
    if run_id:
        doc = col.find_one({"run_id": run_id}, {"run_id": 1})
        if not doc:
            raise ValueError(f"run_id không tồn tại: {run_id}")
        return run_id
    doc = col.find_one({}, {"run_id": 1}, sort=[("timestamp", -1)])
    if not doc:
        raise ValueError("Không tìm thấy kết quả clustering. Chạy complete_pipeline.py trước.")
    return doc["run_id"]


def load_run_data(mongo: MongoDBClient, run_id: str) -> dict:
    """Load toàn bộ dữ liệu của một run từ MongoDB."""
    run_doc   = mongo.db["processing_results"].find_one({"run_id": run_id}, {"_id": 0})
    asgn_docs = list(mongo.db["cluster_assignments"].find({"run_id": run_id}, {"_id": 0}))
    profiles  = list(mongo.db["cluster_profiles"].find({"run_id": run_id}, {"_id": 0}))

    # k_search metadata (nếu có)
    k_search = mongo.db["k_search_metadata"].find_one({"run_id": run_id}, {"_id": 0})

    # Sampling metadata (nếu có)
    sampling = mongo.db["sampling_metadata"].find_one({"run_id": run_id}, {"_id": 0})

    # Reconstruct arrays
    filenames    = [d["filename"]  for d in asgn_docs]
    labels_pred  = np.array([d["cluster_id"] for d in asgn_docs], dtype=int)
    labels_true  = np.array([d["true_label"] for d in asgn_docs])
    n_clusters   = int(run_doc.get("n_clusters", len(set(labels_pred))))

    print(f"  [Load] run_id={run_id[:8]}... | n={len(filenames):,} | k={n_clusters}")
    return {
        "run_id":      run_id,
        "run_doc":     run_doc,
        "filenames":   filenames,
        "labels_pred": labels_pred,
        "labels_true": labels_true,
        "profiles":    sorted(profiles, key=lambda p: p["cluster_id"]),
        "k_search":    k_search,
        "sampling":    sampling,
        "n_clusters":  n_clusters,
        "metrics":     run_doc.get("metrics", {}),
    }


def reduce_pca_2d(mongo: MongoDBClient, run_id: str,
                  n_samples: int = 5000) -> tuple:
    """
    Lấy clip_vector từ clip_features → PCA 2D để visualize.
    Chỉ lấy tối đa n_samples để vẽ nhanh.
    """
    asgn_docs = list(mongo.db["cluster_assignments"].find(
        {"run_id": run_id}, {"_id": 0, "filename": 1, "cluster_id": 1, "true_label": 1}
    ))
    filenames_all = [d["filename"] for d in asgn_docs]

    # Sample nếu quá nhiều
    if len(filenames_all) > n_samples:
        idx = np.random.choice(len(filenames_all), n_samples, replace=False)
        filenames_sample = [filenames_all[i] for i in idx]
        labels_pred_s    = np.array([asgn_docs[i]["cluster_id"] for i in idx])
        labels_true_s    = np.array([asgn_docs[i]["true_label"] for i in idx])
    else:
        filenames_sample = filenames_all
        labels_pred_s    = np.array([d["cluster_id"] for d in asgn_docs])
        labels_true_s    = np.array([d["true_label"]  for d in asgn_docs])

    col_feat = mongo.db["clip_features"]
    vec_map  = {
        d["filename"]: d["clip_vector"]
        for d in col_feat.find(
            {"filename": {"$in": filenames_sample}},
            {"_id": 0, "filename": 1, "clip_vector": 1},
        )
    }

    vecs, valid_pred, valid_true = [], [], []
    for fn, lp, lt in zip(filenames_sample, labels_pred_s, labels_true_s):
        vec = vec_map.get(fn)
        if vec:
            vecs.append(vec)
            valid_pred.append(lp)
            valid_true.append(lt)

    if not vecs:
        return None, None, None

    X = np.array(vecs, dtype=np.float32)
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X)
    var  = float(pca2.explained_variance_ratio_.sum())
    print(f"  [PCA-2D] {X.shape[1]}D → 2D | variance retained: {var*100:.1f}%")
    return X_2d, np.array(valid_pred), np.array(valid_true)


# ================================================================
#  PHẦN 1 – METRICS TỔNG HỢP
# ================================================================

def plot_metrics_summary(data: dict, out_dir: str):
    """
    1.1  Bảng metrics + gauge chart.
    """
    metrics = data["metrics"]
    run_doc = data["run_doc"]
    sil  = metrics.get("silhouette", 0)
    ch   = metrics.get("calinski_harabasz", 0)
    db   = metrics.get("davies_bouldin", 0)
    pur  = metrics.get("purity", 0)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(
        f"Metrics Tổng hợp – run {data['run_id'][:8]}...\n"
        f"CLIP 512D → PCA {run_doc.get('pca_n_components',64)}D → "
        f"KMeans k={data['n_clusters']}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.5)

    items = [
        ("Silhouette\nScore",    sil,  -1,    1,    "↑ tốt", "#378ADD"),
        ("Calinski\nHarabasz",   ch,    0, 1500,    "↑ tốt", "#4CAF50"),
        ("Davies\nBouldin",      db,    0,    5,    "↓ tốt", "#E05C3A"),
        ("Purity\n(ground truth)",pur,  0,    1,    "↑ tốt", "#E8A838"),
    ]
    for col_i, (name, val, vmin, vmax, note, color) in enumerate(items):
        ax = fig.add_subplot(gs[col_i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        # Circle indicator
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(0.5 + 0.38*np.cos(theta), 0.5 + 0.38*np.sin(theta),
                color="#EEEEEE", lw=10, solid_capstyle="round")
        # Normalized fill
        norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
        norm_val = np.clip(norm_val, 0, 1)
        if name.startswith("Davies"):  # ↓ tốt → invert
            norm_val = 1 - norm_val
        t_end = 2 * np.pi * norm_val
        theta_fill = np.linspace(-np.pi/2, -np.pi/2 + t_end, 100)
        ax.plot(0.5 + 0.38*np.cos(theta_fill), 0.5 + 0.38*np.sin(theta_fill),
                color=color, lw=10, solid_capstyle="round")

        fmt = f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}"
        ax.text(0.5, 0.5, fmt, ha="center", va="center",
                fontsize=15, fontweight="bold", color=color)
        ax.text(0.5, 0.08, name, ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(0.5, 0.93, note, ha="center", va="top",
                fontsize=8, color="#777777")

        # Purity grade
        if name.startswith("Purity"):
            grade = ("EXCELLENT" if pur >= 0.80
                     else "GOOD" if pur >= 0.70 else "ACCEPTABLE")
            ax.text(0.5, 0.18, f"[{grade}]", ha="center", va="bottom",
                    fontsize=9, color=color, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    path = os.path.join(out_dir, "01_metrics_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_k_search(data: dict, out_dir: str):
    """
    1.2  Elbow curve: metrics vs k từ k_search_metadata.
    """
    k_search = data.get("k_search")
    if not k_search:
        print("  [SKIP] Không có k_search_metadata (chạy với --find-k)")
        return

    results  = k_search.get("results", [])
    rec      = k_search.get("recommendation", {})
    best_k   = rec.get("k", 5)
    if not results:
        return

    ks   = [r["k"]              for r in results]
    sils = [r["silhouette"]     for r in results]
    chs  = [r["calinski_harabasz"] for r in results]
    dbs  = [r["davies_bouldin"] for r in results]
    ines = [r["inertia"]        for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Find Optimal K – Elbow & Metric Analysis", fontsize=14, fontweight="bold")

    configs = [
        (axes[0,0], sils, "Silhouette Score (↑ tốt)", "#378ADD", True),
        (axes[0,1], chs,  "Calinski-Harabasz (↑ tốt)", "#4CAF50", True),
        (axes[1,0], dbs,  "Davies-Bouldin (↓ tốt)",    "#E05C3A", False),
        (axes[1,1], ines, "Inertia / SSE (Elbow)",      "#888888", False),
    ]
    for ax, vals, title, color, up in configs:
        ax.plot(ks, vals, "o-", color=color, linewidth=2, markersize=7,
                markerfacecolor="white", markeredgewidth=2)
        ax.axvline(best_k, color="navy", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"k={best_k} (optimal)")
        ax.scatter([best_k], [vals[ks.index(best_k)] if best_k in ks else 0],
                   s=120, color="navy", zorder=5)
        ax.set_title(title)
        ax.set_xlabel("k (số cluster)")
        ax.legend(fontsize=9)
        ax.set_xticks(ks)

    # Voting summary
    votes_str = "  Voting: "
    for kk, vv in sorted(rec.get("votes", {}).items(), key=lambda x: -x[1]):
        votes_str += f"k={kk}→{vv}pts  "
    fig.text(0.5, 0.01, votes_str.strip(), ha="center", fontsize=10,
             color="#444444", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(out_dir, "02_k_search.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


# ================================================================
#  PHẦN 2 – CLUSTER VISUALIZATION
# ================================================================

def plot_scatter_2d(X_2d, labels_pred, labels_true,
                    n_clusters: int, out_dir: str):
    """
    2.1 + 2.2  Scatter plot 2D PCA (cluster color | label color).
    """
    if X_2d is None:
        print("  [SKIP] Không có dữ liệu CLIP vector để vẽ scatter")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("2D PCA Scatter – CLIP Features", fontsize=14, fontweight="bold")

    # --- Left: cluster color ---
    for cid in range(n_clusters):
        mask  = labels_pred == cid
        color = CLUSTER_CMAP(cid / max(n_clusters, 1))
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[color], s=8, alpha=0.55,
                        label=f"C{cid} (n={mask.sum():,})",
                        edgecolors="none")
    axes[0].set_title("Màu theo Cluster ID (KMeans)", fontweight="bold")
    axes[0].legend(markerscale=3, fontsize=9, loc="upper right",
                   framealpha=0.8)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].axis("equal")

    # --- Right: ground-truth label color ---
    for kw in KEYWORDS:
        mask = labels_true == kw
        if mask.sum() == 0:
            continue
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[KW_COLORS.get(kw, "#888888")],
                        s=8, alpha=0.55, label=f"{kw} (n={mask.sum():,})",
                        edgecolors="none")
    axes[1].set_title("Màu theo Label gốc (Ground Truth)", fontweight="bold")
    axes[1].legend(markerscale=3, fontsize=9, loc="upper right",
                   framealpha=0.8)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].axis("equal")

    plt.tight_layout()
    path = os.path.join(out_dir, "03_scatter_2d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_confusion_matrix(labels_pred, labels_true,
                          n_clusters: int, out_dir: str):
    """
    2.3  Confusion matrix: Cluster × Ground-truth Label.
    """
    import pandas as pd

    df    = pd.DataFrame({"cluster": labels_pred, "label": labels_true})
    pivot = (df.groupby(["cluster", "label"])
               .size().unstack(fill_value=0)
               .reindex(columns=KEYWORDS, fill_value=0))

    # Normalize theo hàng (cluster) để xem purity
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Confusion Matrix: Cluster × Ground-Truth Label",
                 fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, linecolor="white", ax=axes[0],
                cbar_kws={"label": "Số ảnh"},
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    axes[0].set_title("Số ảnh thực tế")
    axes[0].set_xlabel("Label gốc"); axes[0].set_ylabel("Cluster ID")
    axes[0].tick_params(axis="y", rotation=0)

    # Normalized (purity per row)
    sns.heatmap(pivot_norm, annot=True, fmt=".2f", cmap="Greens",
                linewidths=0.5, linecolor="white", ax=axes[1],
                vmin=0, vmax=1,
                cbar_kws={"label": "Tỉ lệ"},
                annot_kws={"fontsize": 11})
    axes[1].set_title("Tỉ lệ (normalized per cluster)")
    axes[1].set_xlabel("Label gốc"); axes[1].set_ylabel("Cluster ID")
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "04_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


# ================================================================
#  PHẦN 3 – CLUSTER PROFILES
# ================================================================

def plot_cluster_profiles(data: dict, out_dir: str):
    """
    3.1  Bar chart: size + purity.
    3.2  Stacked bar: label distribution.
    """
    profiles = data["profiles"]
    if not profiles:
        return

    cluster_ids = [p["cluster_id"]      for p in profiles]
    sizes       = [p["size"]            for p in profiles]
    purities    = [p["dominant_purity"] for p in profiles]
    dominants   = [p["dominant_label"]  for p in profiles]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cluster Profiles – Phân tích chi tiết",
                 fontsize=13, fontweight="bold")

    # --- 3.1a: Size ---
    colors_bar = [KW_COLORS.get(d, "#888888") for d in dominants]
    bars = axes[0].bar([f"C{c}" for c in cluster_ids], sizes,
                       color=colors_bar, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Cluster Size (số ảnh)")
    axes[0].set_ylabel("Số ảnh")
    for bar, v in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     v + max(sizes)*0.01, f"{v:,}",
                     ha="center", fontsize=10, fontweight="bold")
    # Legend
    handles = [mpatches.Patch(color=KW_COLORS.get(kw, "#888"),
               label=kw) for kw in KEYWORDS if kw in dominants]
    axes[0].legend(handles=handles, fontsize=8)

    # --- 3.1b: Purity ---
    bars2 = axes[1].bar([f"C{c}" for c in cluster_ids], purities,
                         color=colors_bar, edgecolor="white", linewidth=0.8)
    axes[1].axhline(0.70, color="orange", linestyle="--", linewidth=1.2,
                    label="70% threshold")
    axes[1].axhline(0.80, color="green",  linestyle="--", linewidth=1.2,
                    label="80% threshold")
    axes[1].set_title("Dominant Purity mỗi Cluster")
    axes[1].set_ylabel("Purity"); axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=9)
    for bar, v in zip(bars2, purities):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     v + 0.02, f"{v:.1%}",
                     ha="center", fontsize=10, fontweight="bold")

    # --- 3.2: Stacked label distribution ---
    import pandas as pd
    dist_data = {}
    for p in profiles:
        dist_data[f"C{p['cluster_id']}"] = p.get("label_distribution", {})
    df_dist  = pd.DataFrame(dist_data).T.fillna(0).astype(int)
    df_dist  = df_dist.reindex(columns=[k for k in KEYWORDS if k in df_dist.columns])
    bottom   = np.zeros(len(df_dist))
    for kw in df_dist.columns:
        vals = df_dist[kw].values
        axes[2].bar(df_dist.index, vals, bottom=bottom,
                    color=KW_COLORS.get(kw, "#888888"),
                    label=kw, edgecolor="white", linewidth=0.3)
        bottom += vals
    axes[2].set_title("Phân bố Label gốc trong mỗi Cluster")
    axes[2].set_ylabel("Số ảnh")
    axes[2].legend(title="Label", bbox_to_anchor=(1.01, 1),
                   loc="upper left", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "05_cluster_profiles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_silhouette_detail(labels_pred, labels_true,
                           mongo: MongoDBClient, run_id: str,
                           n_clusters: int, out_dir: str):
    """
    3.3  Silhouette plot chi tiết: từng điểm trong mỗi cluster.
    Lấy PCA vectors từ MongoDB để tính silhouette samples.
    """
    # Load PCA-reduced features (lấy từ clip_features → PCA lại)
    asgn_docs = list(mongo.db["cluster_assignments"].find(
        {"run_id": run_id},
        {"_id": 0, "filename": 1, "cluster_id": 1}
    ))
    filenames  = [d["filename"]   for d in asgn_docs]
    lab_arr    = np.array([d["cluster_id"] for d in asgn_docs])

    col_feat   = mongo.db["clip_features"]
    vec_map    = {
        d["filename"]: d["clip_vector"]
        for d in col_feat.find(
            {"filename": {"$in": filenames[:5000]}},
            {"_id": 0, "filename": 1, "clip_vector": 1},
        )
    }

    sample_n = 3000
    vecs, lbs = [], []
    for fn, lb in zip(filenames, lab_arr):
        if fn in vec_map:
            vecs.append(vec_map[fn])
            lbs.append(lb)
        if len(vecs) >= sample_n:
            break

    if not vecs:
        print("  [SKIP] Không đủ dữ liệu cho silhouette plot")
        return

    X_sub  = np.array(vecs, dtype=np.float32)
    lbs    = np.array(lbs)

    # PCA 64D cho nhanh
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as PCA_sk
    Xs     = StandardScaler().fit_transform(X_sub)
    Xp     = PCA_sk(n_components=min(64, Xs.shape[1]), random_state=42).fit_transform(Xs)

    sil_vals = silhouette_samples(Xp, lbs)
    sil_avg  = sil_vals.mean()

    fig, ax = plt.subplots(figsize=(10, max(5, n_clusters)))
    y_lower  = 10
    colors_c = [CLUSTER_CMAP(c / max(n_clusters, 1)) for c in range(n_clusters)]

    for cid in range(n_clusters):
        sil_c   = np.sort(sil_vals[lbs == cid])
        y_upper = y_lower + len(sil_c)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_c,
                         facecolor=colors_c[cid], alpha=0.75)
        ax.text(-0.08, y_lower + len(sil_c)/2, f"C{cid}",
                fontsize=9, va="center")
        y_lower = y_upper + 15

    ax.axvline(sil_avg, color="red", linestyle="--", linewidth=1.8,
               label=f"Avg = {sil_avg:.4f}")
    ax.set_title(f"Silhouette Plot – Chi tiết từng Cluster\n"
                 f"(sample {len(vecs):,}/{len(filenames):,} ảnh)",
                 fontweight="bold")
    ax.set_xlabel("Silhouette coefficient  [-1, 1]")
    ax.set_ylabel("Cluster")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.3, 0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, "06_silhouette_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


# ================================================================
#  PHẦN 4 – SAMPLING ANALYSIS
# ================================================================

def plot_sampling_analysis(data: dict, out_dir: str):
    """
    4.1  Phân bố trước/sau sampling (chỉ hiển thị khi sampling được áp dụng).
    """
    sampling = data.get("sampling")
    if not sampling or not sampling.get("applied"):
        print("  [SKIP] Sampling không được áp dụng trong run này")
        return

    dist_before = sampling.get("class_distribution_before", {})
    dist_after  = sampling.get("class_distribution_after",  {})
    strategy    = sampling.get("sample_strategy", "N/A")
    sample_pct  = sampling.get("sample_percentage", 0)

    labels = sorted(set(list(dist_before.keys()) + list(dist_after.keys())))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Sampling Analysis – Strategy: {strategy} | "
        f"Sample: {sample_pct:.1f}% of data",
        fontsize=13, fontweight="bold"
    )

    before_vals = [dist_before.get(l, 0) for l in labels]
    after_vals  = [dist_after.get(l, 0)  for l in labels]
    colors_lbl  = [KW_COLORS.get(l, "#888") for l in labels]

    # Counts
    x    = np.arange(len(labels))
    w    = 0.35
    axes[0].bar(x - w/2, before_vals, w, label="Trước sampling",
                color=colors_lbl, edgecolor="white", alpha=1.0)
    axes[0].bar(x + w/2, after_vals,  w, label="Sau sampling",
                color=colors_lbl, edgecolor="white", alpha=0.55)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_title("Phân bố trước/sau Sampling"); axes[0].set_ylabel("Số ảnh")
    axes[0].legend()

    # Percentage comparison
    total_b = sum(before_vals) or 1
    total_a = sum(after_vals)  or 1
    pct_b   = [v/total_b*100 for v in before_vals]
    pct_a   = [v/total_a*100 for v in after_vals]
    axes[1].bar(x - w/2, pct_b, w, label="Trước",
                color=colors_lbl, edgecolor="white", alpha=1.0)
    axes[1].bar(x + w/2, pct_a, w, label="Sau",
                color=colors_lbl, edgecolor="white", alpha=0.55)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_title("Tỉ lệ % (kiểm tra stratified balance)")
    axes[1].set_ylabel("Tỉ lệ (%)")
    axes[1].legend()
    # Đường 20% reference (5 classes equally distributed)
    axes[1].axhline(100/len(labels), color="gray", linestyle=":",
                    linewidth=1.2, label=f"Equal ({100/len(labels):.0f}%)")

    plt.tight_layout()
    path = os.path.join(out_dir, "07_sampling_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


# ================================================================
#  PHẦN 5 – GALLERY ẢNH MEDOID
# ================================================================

def plot_gallery(data: dict, out_dir: str):
    """
    5.1  Gallery: ảnh medoid (đại diện) của mỗi cluster từ MinIO.
    """
    if not MINIO_AVAILABLE:
        print("  [SKIP] MinIO không khả dụng")
        return

    profiles = data["profiles"]
    if not profiles:
        return

    try:
        minio = MinioClient()
    except Exception as e:
        print(f"  [SKIP] MinIO lỗi: {e}")
        return

    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Gallery – Ảnh Medoid (đại diện) mỗi Cluster",
                 fontsize=13, fontweight="bold")

    # Load object_name từ clip_features → preprocessed/ path
    for i, prof in enumerate(profiles):
        ax = axes[i]
        ax.axis("off")
        cid      = prof["cluster_id"]
        medoid   = prof.get("medoid_filename", "")
        dominant = prof.get("dominant_label", "?")
        purity   = prof.get("dominant_purity", 0)

        # Tìm object_name cho medoid filename
        feat_doc = mongo_ref.db["clip_features"].find_one(
            {"filename": medoid},
            {"_id": 0, "object_name": 1}
        )
        obj_name = feat_doc.get("object_name", "") if feat_doc else ""

        try:
            if obj_name:
                resp = minio.client.get_object(MINIO_BUCKET, obj_name)
                img  = Image.open(io.BytesIO(resp.read())).convert("RGB")
                resp.close()
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
        except Exception:
            ax.text(0.5, 0.5, "Load error", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")

        color  = KW_COLORS.get(dominant, "#333333")
        bar_w  = int(purity * 15)
        ax.set_title(
            f"C{cid} – {dominant}\n"
            f"Purity: {purity:.1%}  {'█'*bar_w}",
            fontsize=10, color=color, fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "08_medoid_gallery.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


# ================================================================
#  SUMMARY TEXT
# ================================================================

def print_summary(data: dict):
    """In tóm tắt đánh giá ra console."""
    metrics  = data["metrics"]
    profiles = data["profiles"]
    sil  = metrics.get("silhouette", 0)
    pur  = metrics.get("purity", 0)
    grade = ("EXCELLENT" if pur >= 0.80 else "GOOD" if pur >= 0.70 else "ACCEPTABLE")

    print(f"\n{'='*65}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"  run_id  : {data['run_id']}")
    print(f"  n_samples: {len(data['labels_pred']):,}  |  k={data['n_clusters']}")
    print(f"\n  Metrics:")
    print(f"    Silhouette Score  : {sil:.4f}        ([-1,1] ↑ tốt)")
    print(f"    Calinski-Harabasz : {metrics.get('calinski_harabasz',0):.2f}     (↑ tốt)")
    print(f"    Davies-Bouldin    : {metrics.get('davies_bouldin',0):.4f}    (↓ tốt)")
    print(f"    Purity            : {pur:.4f} ({100*pur:.2f}%)  [{grade}]")
    print(f"\n  Cluster Profiles:")
    print(f"  {'Cluster':<10} {'Size':>7}  {'%':>6}  {'Dominant':>12}  {'Purity':>8}")
    print(f"  {'─'*52}")
    for p in profiles:
        bar = "█" * int(p["dominant_purity"] * 15)
        chk = "✓" if p["dominant_purity"] >= 0.70 else "~"
        print(f"  {chk} C{p['cluster_id']:<8} {p['size']:>7,}  "
              f"{p['percentage']:>5.1f}%  "
              f"{p['dominant_label']:>12}  "
              f"{p['dominant_purity']:>7.1%}  {bar}")
    print(f"{'='*65}")


# ================================================================
#  MAIN
# ================================================================

mongo_ref = None   # global ref cho plot_gallery


def main():
    global mongo_ref
    parser = argparse.ArgumentParser(
        description="Evaluation – Đánh giá kết quả Clustering"
    )
    parser.add_argument("--run-id",     default=None,
                        help="run_id cụ thể (mặc định: run mới nhất)")
    parser.add_argument("--latest",     action="store_true",
                        help="Dùng run mới nhất (mặc định)")
    parser.add_argument("--no-gallery", action="store_true",
                        help="Bỏ qua gallery ảnh MinIO")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Bỏ qua scatter 2D (nhanh hơn)")
    parser.add_argument("--out-dir",    default=REPORT_DIR,
                        help=f"Thư mục lưu biểu đồ (default: {REPORT_DIR})")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  EVALUATION – Landscape Image Clustering")
    print("=" * 65)

    mongo      = MongoDBClient()
    mongo_ref  = mongo
    print(f"  [OK] Connected to MongoDB")

    # Xác định run_id
    run_id = get_run_id(mongo, args.run_id)
    print(f"  [OK] run_id = {run_id}")

    # Load dữ liệu
    print(f"\n  Loading data from MongoDB...")
    data = load_run_data(mongo, run_id)
    print_summary(data)

    # ── PHẦN 1: Metrics ─────────────────────────────────────────
    print(f"\n  ── PHẦN 1: Metrics ──")
    plot_metrics_summary(data, out_dir)
    plot_k_search(data, out_dir)

    # ── PHẦN 2: Visualization ────────────────────────────────────
    print(f"\n  ── PHẦN 2: Cluster Visualization ──")
    if not args.no_scatter:
        print(f"  Loading CLIP vectors for 2D scatter (có thể mất 1-2 phút)...")
        X_2d, lp_2d, lt_2d = reduce_pca_2d(mongo, run_id, n_samples=5000)
        plot_scatter_2d(X_2d, lp_2d, lt_2d, data["n_clusters"], out_dir)
    plot_confusion_matrix(data["labels_pred"], data["labels_true"],
                          data["n_clusters"], out_dir)

    # ── PHẦN 3: Cluster Profiles ─────────────────────────────────
    print(f"\n  ── PHẦN 3: Cluster Profiles ──")
    plot_cluster_profiles(data, out_dir)
    plot_silhouette_detail(data["labels_pred"], data["labels_true"],
                           mongo, run_id, data["n_clusters"], out_dir)

    # ── PHẦN 4: Sampling ─────────────────────────────────────────
    print(f"\n  ── PHẦN 4: Sampling Analysis ──")
    plot_sampling_analysis(data, out_dir)

    # ── PHẦN 5: Gallery ─────────────────────────────────────────
    if not args.no_gallery:
        print(f"\n  ── PHẦN 5: Gallery ảnh Medoid ──")
        plot_gallery(data, out_dir)
    else:
        print(f"\n  [SKIP] Gallery (--no-gallery)")

    # ── Tổng kết ────────────────────────────────────────────────
    saved_files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    print(f"\n{'='*65}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Reports → {out_dir}/  ({len(saved_files)} PNG files)")
    for f in sorted(saved_files):
        print(f"    {f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()