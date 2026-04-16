# src/modeling/evaluation.py
# ================================================================
#  EVALUATION – Đánh giá kết quả Clustering
#
#  Metrics:
#    1. Silhouette Score      [-1, 1]    → càng cao càng tốt
#    2. Davies-Bouldin Index  [0, ∞)     → càng thấp càng tốt
#    3. Calinski-Harabasz     [0, ∞)     → càng cao càng tốt
#    4. Purity (so với label gốc từ crawler)
#
#  Visualization:
#    1. t-SNE / UMAP 2D scatter plot (màu theo cluster)
#    2. Scatter plot màu theo label gốc (so sánh)
#    3. Confusion matrix: cluster vs label
#    4. Silhouette plot per cluster
#    5. Gallery ảnh mẫu từng cluster (download từ MinIO)
#    6. Bar chart phân bố cluster
#
#  Input : MongoDB clusters + image_features
#  Output: reports/evaluation/*.png
#
#  Cài  : pip install scikit-learn matplotlib seaborn
#  Chạy : python src/modeling/evaluation.py
#         python src/modeling/evaluation.py --run-id <uuid>
#         python src/modeling/evaluation.py --algo kmeans
# ================================================================

import sys, os, io, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from datetime import datetime

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_OK = True
except ImportError:
    UMAP_OK = False

from src.storage.minio_client   import MinioClient
from src.storage.mongodb_client import MongoDBClient
from configs.config import MINIO_BUCKET, KEYWORDS

# ── Config ────────────────────────────────────────────────────────
REPORT_DIR    = "reports/evaluation"
TSNE_PERP     = 30
GALLERY_N     = 5     # số ảnh mẫu mỗi cluster

KW_COLORS = {
    "mountain": "#5B8DB8", "forest": "#4CAF50",
    "sea":      "#2EC4B6", "desert": "#E8A838", "snow": "#A0C4D8",
}
CLUSTER_PALETTE = plt.cm.get_cmap("tab20")


# ================================================================
#  LOAD DATA
# ================================================================
def load_cluster_data(col_clusters, col_features,
                      run_id: str = None,
                      algo: str = None) -> tuple:
    """
    Load cluster assignments + feature vectors.
    Trả về (cluster_docs, X, labels, true_labels)
    """
    # Tìm run mới nhất nếu không chỉ định
    query = {}
    if run_id:
        query["run_id"] = run_id
    if algo:
        query["algo"] = algo

    # Nếu không có filter → lấy run mới nhất
    if not query:
        latest = col_clusters.find_one(
            {}, {"run_id": 1}, sort=[("created_at", -1)]
        )
        if not latest:
            return [], np.array([]), np.array([]), np.array([])
        query["run_id"] = latest["run_id"]

    cluster_docs = list(col_clusters.find(query, {"_id": 0}))
    if not cluster_docs:
        return [], np.array([]), np.array([]), np.array([])

    print(f"  [Load] {len(cluster_docs):,} cluster docs "
          f"(algo={cluster_docs[0].get('algo','?')}, "
          f"k={cluster_docs[0].get('n_clusters','?')})")

    # Lấy vectors tương ứng từ image_features
    obj_names   = [d["object_name"] for d in cluster_docs]
    feat_map    = {
        d["object_name"]: d["resnet_vector"]
        for d in col_features.find(
            {"object_name": {"$in": obj_names}},
            {"object_name": 1, "resnet_vector": 1, "_id": 0},
        )
    }

    rows = []
    labels_list      = []
    true_labels_list = []

    for d in cluster_docs:
        vec = feat_map.get(d["object_name"])
        if vec is None:
            continue
        rows.append(vec)
        labels_list.append(d["cluster_id"])
        true_labels_list.append(d.get("label", "unknown"))

    X           = np.array(rows,          dtype=np.float32)
    labels      = np.array(labels_list,   dtype=int)
    true_labels = np.array(true_labels_list)

    print(f"  [Load] X shape: {X.shape} | unique clusters: {len(set(labels_list))}")
    return cluster_docs, X, labels, true_labels


# ================================================================
#  METRICS
# ================================================================
def compute_metrics(X: np.ndarray, labels: np.ndarray,
                    true_labels: np.ndarray) -> dict:
    """Tính tất cả evaluation metrics."""
    mask    = labels >= 0          # bỏ noise (-1 của DBSCAN)
    X_valid = X[mask]
    L_valid = labels[mask]

    metrics = {}

    if len(set(L_valid)) < 2:
        print("  [!] Chỉ có 1 cluster – không tính silhouette/DB/CH")
        return metrics

    metrics["silhouette"]       = float(silhouette_score(X_valid, L_valid,
                                                          metric="euclidean"))
    metrics["davies_bouldin"]   = float(davies_bouldin_score(X_valid, L_valid))
    metrics["calinski_harabasz"]= float(calinski_harabasz_score(X_valid, L_valid))

    # Purity (so với label gốc)
    tl_valid = true_labels[mask]
    n = len(L_valid)
    if n > 0:
        purity_sum = 0
        for cid in set(L_valid):
            mask_c   = L_valid == cid
            true_in_c = tl_valid[mask_c]
            if len(true_in_c) > 0:
                counts    = {kw: int((true_in_c == kw).sum()) for kw in KEYWORDS}
                purity_sum += max(counts.values())
        metrics["purity"] = purity_sum / n

    print(f"\n  {'Metric':<28} {'Value':>10}")
    print(f"  {'─'*40}")
    for k, v in metrics.items():
        arrow = "↑ tốt" if k in ("silhouette","calinski_harabasz","purity") else "↓ tốt"
        print(f"  {k:<28} {v:>10.4f}  ({arrow})")

    return metrics


# ================================================================
#  DIMENSIONALITY REDUCTION 2D
# ================================================================
def reduce_2d(X: np.ndarray, method: str = "tsne") -> np.ndarray:
    """Giảm chiều về 2D để visualize."""
    # Trước tiên PCA xuống 50 để TSNE/UMAP nhanh hơn
    n_pca = min(50, X.shape[0], X.shape[1])
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    if method == "umap" and UMAP_OK:
        reducer = umap.UMAP(n_components=2, n_neighbors=15,
                            min_dist=0.1, random_state=42)
        X_2d = reducer.fit_transform(X_pca)
        print(f"  [UMAP 2D] shape: {X_2d.shape}")
    else:
        tsne = TSNE(n_components=2, perplexity=min(TSNE_PERP, len(X)//4),
                    random_state=42, n_jobs=-1)
        X_2d = tsne.fit_transform(X_pca)
        print(f"  [t-SNE 2D] shape: {X_2d.shape}")
    return X_2d


# ================================================================
#  VISUALIZATIONS
# ================================================================
def plot_scatter_clusters(X_2d, labels, true_labels,
                          algo_name: str, run_id: str, out_dir: str):
    """Scatter 2D: left=cluster color, right=true label color."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"2D Embedding – {algo_name.upper()}  (run {run_id[:8]}...)",
                 fontsize=14, fontweight="bold")

    unique_clusters = sorted(set(labels))
    n_c = len([c for c in unique_clusters if c >= 0])

    # Left: cluster colors
    for cid in unique_clusters:
        mask = labels == cid
        color = "black" if cid == -1 else CLUSTER_PALETTE(cid / max(n_c, 1))
        label = "noise" if cid == -1 else f"Cluster {cid}"
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color], s=8, alpha=0.6, label=label, edgecolors="none")
    axes[0].set_title("Theo Cluster ID", fontweight="bold")
    axes[0].legend(markerscale=3, fontsize=7,
                   loc="upper right", ncol=2)
    axes[0].axis("off")

    # Right: true label colors
    for kw in KEYWORDS:
        mask = true_labels == kw
        if mask.sum() == 0:
            continue
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[KW_COLORS.get(kw, "#888")],
                       s=8, alpha=0.6, label=kw, edgecolors="none")
    axes[1].set_title("Theo Label gốc (ground truth)", fontweight="bold")
    axes[1].legend(markerscale=3, fontsize=8)
    axes[1].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"01_scatter_{algo_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_confusion_heatmap(labels, true_labels, algo_name: str, out_dir: str):
    """Heatmap cluster × true label."""
    import pandas as pd
    df = pd.DataFrame({"cluster": labels, "true": true_labels})
    df = df[df["cluster"] >= 0]  # bỏ noise
    pivot = df.groupby(["cluster", "true"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=KEYWORDS, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot)*0.6)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Số ảnh"},
                annot_kws={"fontsize": 10})
    ax.set_title(f"Confusion Matrix: Cluster × Label gốc ({algo_name.upper()})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Label gốc"); ax.set_ylabel("Cluster ID")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    path = os.path.join(out_dir, f"02_confusion_{algo_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_silhouette(X, labels, algo_name: str, out_dir: str):
    """Silhouette plot chi tiết từng cluster."""
    mask    = labels >= 0
    X_v     = X[mask]
    L_v     = labels[mask]
    if len(set(L_v)) < 2 or len(L_v) < 4:
        return

    sil_vals    = silhouette_samples(X_v, L_v)
    sil_avg     = sil_vals.mean()
    unique_ids  = sorted(set(L_v))

    fig, ax = plt.subplots(figsize=(9, max(5, len(unique_ids)*0.8)))
    y_lower = 10

    for cid in unique_ids:
        sil_c = np.sort(sil_vals[L_v == cid])
        size_c = len(sil_c)
        y_upper = y_lower + size_c
        color = CLUSTER_PALETTE(cid / max(len(unique_ids), 1))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_c,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + size_c/2, f"C{cid}", fontsize=8)
        y_lower = y_upper + 10

    ax.axvline(sil_avg, color="red", linestyle="--", linewidth=1.5,
               label=f"Avg = {sil_avg:.3f}")
    ax.set_title(f"Silhouette Plot – {algo_name.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Cluster")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"03_silhouette_{algo_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_cluster_distribution(labels, true_labels, algo_name: str, out_dir: str):
    """Bar chart phân bố ảnh trong từng cluster theo label gốc."""
    import pandas as pd
    df    = pd.DataFrame({"cluster": labels, "true": true_labels})
    df    = df[df["cluster"] >= 0]
    pivot = df.groupby(["cluster","true"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=KEYWORDS, fill_value=0)

    ax = pivot.plot(
        kind="bar", stacked=True,
        color=[KW_COLORS.get(k, "#888") for k in pivot.columns],
        figsize=(max(8, len(pivot)*0.8), 5),
        edgecolor="white", linewidth=0.5,
    )
    ax.set_title(f"Phân bố label gốc trong từng Cluster – {algo_name.upper()}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("Số ảnh")
    ax.legend(title="Label gốc", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    path = os.path.join(out_dir, f"04_distribution_{algo_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_gallery(cluster_docs: list, labels: np.ndarray,
                 algo_name: str, minio: MinioClient,
                 out_dir: str, n_per_cluster: int = GALLERY_N):
    """Gallery ảnh mẫu từng cluster – download từ MinIO."""
    unique_ids = sorted(set(labels[labels >= 0]))
    n_rows = len(unique_ids)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, n_per_cluster,
        figsize=(n_per_cluster * 2.5, n_rows * 2.8),
    )
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(f"Gallery mẫu – {algo_name.upper()} "
                 f"({n_per_cluster} ảnh/cluster)",
                 fontsize=14, fontweight="bold")

    # Map cluster_id → docs
    cid_to_docs: dict[int, list] = {}
    for doc, cid in zip(cluster_docs, labels):
        if cid >= 0:
            cid_to_docs.setdefault(int(cid), []).append(doc)

    for row_i, cid in enumerate(unique_ids):
        samples = cid_to_docs.get(cid, [])[:n_per_cluster]
        for col_i in range(n_per_cluster):
            ax = axes[row_i][col_i] if n_per_cluster > 1 else axes[row_i]
            ax.axis("off")
            if col_i >= len(samples):
                continue
            doc = samples[col_i]
            try:
                resp = minio.client.get_object(
                    MINIO_BUCKET, doc["object_name"]
                )
                img  = Image.open(io.BytesIO(resp.read())).convert("RGB")
                resp.close()
                ax.imshow(img)
                kw_label = doc.get("label", "?")
                ax.set_title(
                    f"C{cid} – {kw_label}",
                    fontsize=7,
                    color=KW_COLORS.get(kw_label, "#333"),
                )
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"05_gallery_{algo_name}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


def plot_metrics_comparison(all_metrics: dict, out_dir: str):
    """So sánh metrics giữa các thuật toán."""
    import pandas as pd
    if len(all_metrics) < 2:
        return

    df = pd.DataFrame(all_metrics).T
    metrics_plot = [c for c in ["silhouette","purity","calinski_harabasz","davies_bouldin"]
                    if c in df.columns]

    fig, axes = plt.subplots(1, len(metrics_plot),
                             figsize=(len(metrics_plot)*4, 4))
    if len(metrics_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_plot):
        values = df[metric].astype(float)
        bars   = ax.bar(df.index, values,
                        color=["#E05C3A","#378ADD","#4CAF50"][:len(df)],
                        edgecolor="white", linewidth=0.8, width=0.5)
        ax.set_title(metric.replace("_"," ").title(), fontweight="bold")
        ax.set_ylabel("Score")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v + values.max()*0.02,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("So sánh Metrics giữa các thuật toán Clustering",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "06_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [Saved] {path}")


# ================================================================
#  MAIN
# ================================================================
def run_evaluation(minio: MinioClient, mongo: MongoDBClient,
                   run_id: str = None, algo: str = None,
                   reduce_2d_method: str = "tsne"):

    os.makedirs(REPORT_DIR, exist_ok=True)
    col_clusters = mongo.get_col("clusters")
    col_features = mongo.get_col("features")

    # Lấy danh sách run_id cần evaluate
    if run_id:
        run_ids = [run_id]
    elif algo:
        run_ids = list({
            d["run_id"] for d in
            col_clusters.find({"algo": algo}, {"run_id": 1})
        })
    else:
        # Tất cả algo trong run mới nhất
        latest_date = col_clusters.find_one(
            {}, {"created_at": 1}, sort=[("created_at", -1)]
        )
        if not latest_date:
            print("[!] Chưa có kết quả clustering trong MongoDB")
            return
        run_ids = list({
            d["run_id"] for d in
            col_clusters.find(
                {"created_at": latest_date["created_at"]},
                {"run_id": 1}
            )
        })

    all_metrics = {}

    for rid in run_ids:
        cluster_docs, X, labels, true_labels = load_cluster_data(
            col_clusters, col_features, run_id=rid
        )
        if len(cluster_docs) == 0:
            continue

        algo_name = cluster_docs[0].get("algo", "unknown")
        print(f"\n{'─'*55}")
        print(f"  Evaluating: {algo_name.upper()}  (run_id={rid[:8]}...)")
        print(f"{'─'*55}")

        # Metrics
        metrics = compute_metrics(X, labels, true_labels)
        all_metrics[algo_name] = metrics

        # 2D reduction
        X_2d = reduce_2d(X, method=reduce_2d_method)

        # Plots
        plot_scatter_clusters(X_2d, labels, true_labels,
                              algo_name, rid, REPORT_DIR)
        plot_confusion_heatmap(labels, true_labels, algo_name, REPORT_DIR)
        plot_silhouette(X, labels, algo_name, REPORT_DIR)
        plot_cluster_distribution(labels, true_labels, algo_name, REPORT_DIR)
        plot_gallery(cluster_docs, labels, algo_name, minio, REPORT_DIR)

    # So sánh metrics
    if len(all_metrics) >= 2:
        plot_metrics_comparison(all_metrics, REPORT_DIR)

    # Summary
    print(f"\n{'='*55}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*55}")
    for a, m in all_metrics.items():
        print(f"\n  {a.upper()}")
        for k, v in m.items():
            print(f"    {k:<26} {v:.4f}")
    print(f"\n  Reports: {REPORT_DIR}/")

    return all_metrics


# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--algo",   default=None,
                        choices=["kmeans","dbscan","agglomerative"])
    parser.add_argument("--reduce", default="tsne",
                        choices=["tsne","umap"])
    args = parser.parse_args()

    print("=" * 60)
    print("  EVALUATION – Clustering Results")
    print("=" * 60)

    minio = MinioClient()
    mongo = MongoDBClient()
    run_evaluation(minio, mongo,
                   run_id=args.run_id,
                   algo=args.algo,
                   reduce_2d_method=args.reduce)
