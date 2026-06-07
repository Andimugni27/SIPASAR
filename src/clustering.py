from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]


def scale_rfm(rfm: pd.DataFrame, scaler: StandardScaler | None = None):
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(rfm[RFM_FEATURES])
    else:
        X = scaler.transform(rfm[RFM_FEATURES])
    return X, scaler


def find_optimal_k(X: np.ndarray, k_range=range(2, 9), random_state: int = 42):
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(results)


def train_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def label_segments(rfm: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Map numeric cluster ID → business label berdasarkan rata-rata RFM."""
    rfm = rfm.copy()
    stats = rfm.groupby(cluster_col)[RFM_FEATURES].mean()

    stats["score"] = (
        -stats["Recency"].rank()
        + stats["Frequency"].rank()
        + stats["Monetary"].rank()
    )
    ranked = stats.sort_values("score", ascending=False).index.tolist()

    n = len(ranked)
    if n == 1:
        names = ["Regular"]
    elif n == 2:
        names = ["Champions", "At Risk"]
    elif n == 3:
        names = ["Champions", "Loyal", "At Risk"]
    elif n == 4:
        names = ["Champions", "Loyal", "Potential", "At Risk"]
    else:
        names = ["Champions", "Loyal", "Potential", "Need Attention", "At Risk"] + [
            f"Segment-{i}" for i in range(5, n)
        ]

    mapping = {cid: names[i] for i, cid in enumerate(ranked)}
    rfm["Segment"] = rfm[cluster_col].map(mapping)
    return rfm, mapping
