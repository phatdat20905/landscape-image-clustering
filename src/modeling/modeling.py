# src/modeling/modeling.py
# ================================================================
#  MODELING – Clustering với ResNet50 features
#  Workflow: image_features → [Clustering] → clusters
#
#  Thuật toán:
#    1. K-Means      – nhanh, tốt cho cluster hình cầu
#    2. DBSCAN       – tự xác định số cluster, lọc noise
#    3. Agglomerative– hierarchical, hỗ trợ nhiều linkage
#
#  Dimensionality Reduction (trước clustering):
#    PCA  – tuyến tính, giữ variance
#    UMAP – phi tuyến, giữ local structure tốt hơn
#
#  Input : MongoDB image_features (resnet_vector 2048 dim)
#  Output: MongoDB clusters
#          {filename, label, object_name, is_augmented,
#           cluster_id, algo, n_clusters, run_id, created_at}
#
#  Cài  : pip install scikit-learn umap-learn
#  Chạy : python src/modeling/modeling.py
#         python src/modeling/modeling.py --algo kmeans --k 5
#         python src/modeling/modeling.py --algo all --reduce umap
# ================================================================

import sys, os, argparse, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from datetime import datetime
from pymongo import ASCENDING

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_OK = True
except ImportError:
    UMAP_OK = False

from src.storage.mongodb_client import MongoDBClient
from configs.config import KEYWORDS

# ── Config ────────────────────────────────────────────────────────
N_CLUSTERS_DEFAULT = len(KEYWORDS)   # 5
PCA_COMPONENTS     = 128             # giảm từ 2048 → 128 trước clustering
UMAP_COMPONENTS    = 50
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.1

# K-Means
KMEANS_INITS       = 10
KMEANS_MAX_ITER    = 500

# DBSCAN
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 5

# Agglomerative
AGG_LINKAGE        = "ward"


# ================================================================
#  LOAD FEATURES
# ================================================================
def load_features(col_features, include_augmented: bool = False) -> tuple:
    """
    Load vectors từ MongoDB image_features.
    Trả về (X: np.ndarray (N,2048), docs: list[dict]).
    """
    query = {} if include_augmented else {"is_augmented": False}
    projection = {
        "_id": 0,
        "filename": 1, "source_filename": 1, "label": 1,
        "object_name": 1, "is_augmented": 1, "aug_index": 1,
        "resnet_vector": 1,
    }
    docs = list(col_features.find(query, projection))
    if not docs:
        return np.array([]), []

    X = np.array([d["resnet_vector"] for d in docs], dtype=np.float32)
    # Loại vector (pop để tiết kiệm RAM)
    for d in docs:
        d.pop("resnet_vector", None)

    print(f"  [Features] Loaded {len(docs):,} docs  |  shape: {X.shape}")
    return X, docs


# ================================================================
#  DIMENSIONALITY REDUCTION
# ================================================================
def reduce_pca(X: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    """PCA: giảm chiều tuyến tính."""
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_r = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  [PCA] {X.shape[1]} → {n_comp} dims | variance retained: {explained:.3f}")
    return X_r


def reduce_umap(X: np.ndarray,
                n_components: int = UMAP_COMPONENTS,
                n_neighbors:  int = UMAP_N_NEIGHBORS,
                min_dist:     float = UMAP_MIN_DIST) -> np.ndarray:
    """UMAP: giảm chiều phi tuyến, giữ local structure."""
    if not UMAP_OK:
        print("  [!] UMAP không có – dùng PCA thay thế")
        return reduce_pca(X, n_components)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    X_r = reducer.fit_transform(X)
    print(f"  [UMAP] {X.shape[1]} → {n_components} dims (cosine metric)")
    return X_r


# ================================================================
#  CLUSTERING
# ================================================================
def run_kmeans(X: np.ndarray, n_clusters: int = N_CLUSTERS_DEFAULT) -> np.ndarray:
    """K-Means clustering."""
    km = KMeans(
        n_clusters=n_clusters,
        n_init=KMEANS_INITS,
        max_iter=KMEANS_MAX_ITER,
        random_state=42,
        verbose=0,
    )
    labels = km.fit_predict(X)
    unique = np.unique(labels[labels >= 0])
    print(f"  [K-Means] k={n_clusters} | clusters: {len(unique)}")
    return labels


def run_dbscan(X: np.ndarray,
               eps: float = DBSCAN_EPS,
               min_samples: int = DBSCAN_MIN_SAMPLES) -> np.ndarray:
    """DBSCAN clustering. Noise → cluster_id = -1."""
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  [DBSCAN] eps={eps} min_samples={min_samples} "
          f"| clusters: {n_clusters} | noise: {n_noise}")
    return labels


def run_agglomerative(X: np.ndarray,
                      n_clusters: int = N_CLUSTERS_DEFAULT,
                      linkage: str = AGG_LINKAGE) -> np.ndarray:
    """Agglomerative (Hierarchical) clustering."""
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )
    labels = agg.fit_predict(X)
    unique = np.unique(labels)
    print(f"  [Agglomerative] linkage={linkage} k={n_clusters} | clusters: {len(unique)}")
    return labels


# ================================================================
#  SAVE CLUSTERS
# ================================================================
def save_clusters(col_clusters, docs: list, labels: np.ndarray,
                  algo: str, n_clusters: int, run_id: str):
    """Lưu kết quả cluster vào MongoDB clusters collection."""
    inserts = []
    for doc, cid in zip(docs, labels):
        inserts.append({
            "filename":        doc.get("filename", ""),
            "source_filename": doc.get("source_filename", ""),
            "label":           doc.get("label", "unknown"),
            "object_name":     doc.get("object_name", ""),
            "is_augmented":    doc.get("is_augmented", False),
            "cluster_id":      int(cid),
            "algo":            algo,
            "n_clusters":      n_clusters,
            "run_id":          run_id,
            "created_at":      datetime.now().strftime("%Y-%m-%d"),
        })

    if inserts:
        col_clusters.insert_many(inserts, ordered=False)
    print(f"  [Saved] {len(inserts):,} docs vào clusters (run_id={run_id[:8]}...)")


# ================================================================
#  MAIN
# ================================================================
def run_modeling(mongo: MongoDBClient,
                 algo: str = "all",
                 n_clusters: int = N_CLUSTERS_DEFAULT,
                 reduce: str = "pca",
                 include_augmented: bool = False):

    col_features = mongo.get_col("features")
    col_clusters = mongo.get_col("clusters")

    try:
        col_clusters.create_index([("run_id",  ASCENDING)])
        col_clusters.create_index([("algo",    ASCENDING)])
        col_clusters.create_index([("label",   ASCENDING)])
        col_clusters.create_index([("cluster_id", ASCENDING)])
    except Exception:
        pass

    print(f"[Modeling] algo={algo} | k={n_clusters} | reduce={reduce}")
    print(f"  include_augmented: {include_augmented}")

    # ── Load features ─────────────────────────────────────────────
    X, docs = load_features(col_features, include_augmented)
    if len(docs) == 0:
        print("[!] Không có dữ liệu trong image_features – chạy step4 trước")
        return {}

    # ── Scale ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── Dimensionality Reduction ──────────────────────────────────
    if reduce == "umap":
        X_r = reduce_umap(X)
    else:
        X_r = reduce_pca(X)

    run_results = {}
    algos = ["kmeans", "dbscan", "agglomerative"] if algo == "all" else [algo]

    for a in algos:
        run_id = str(uuid.uuid4())
        print(f"\n  ── {a.upper()} ──")

        if a == "kmeans":
            labels = run_kmeans(X_r, n_clusters)
        elif a == "dbscan":
            labels = run_dbscan(X_r)
        elif a == "agglomerative":
            labels = run_agglomerative(X_r, n_clusters)
        else:
            print(f"  [!] Không nhận algo: {a}")
            continue

        n_actual = len(set(labels)) - (1 if -1 in labels else 0)
        save_clusters(col_clusters, docs, labels, a, n_actual, run_id)
        run_results[a] = {"run_id": run_id, "n_clusters": n_actual, "labels": labels}

    return run_results


# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering pipeline")
    parser.add_argument("--algo",   default="all",
                        choices=["kmeans", "dbscan", "agglomerative", "all"])
    parser.add_argument("--k",      type=int, default=N_CLUSTERS_DEFAULT,
                        help=f"Số cluster (default: {N_CLUSTERS_DEFAULT})")
    parser.add_argument("--reduce", default="pca",
                        choices=["pca", "umap"])
    parser.add_argument("--aug",    action="store_true",
                        help="Include augmented images")
    args = parser.parse_args()

    print("=" * 60)
    print("  MODELING – Landscape Image Clustering")
    print("=" * 60)

    mongo = MongoDBClient()
    results = run_modeling(
        mongo,
        algo=args.algo,
        n_clusters=args.k,
        reduce=args.reduce,
        include_augmented=args.aug,
    )
    print("\nKết quả:")
    for a, r in results.items():
        print(f"  {a:<15} k={r['n_clusters']}  run_id={r['run_id'][:8]}...")
