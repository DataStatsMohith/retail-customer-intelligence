"""
Customer Segmentation Models:
  - KMeans (interpretable, scalable)
  - DBSCAN  (density-based, handles noise)
  - Hierarchical Clustering (dendrogram insights)
Uses StandardScaler preprocessing + silhouette score evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = ["recency", "frequency", "monetary_log", "avg_basket_size",
                "unique_products", "weekend_ratio"]

class CustomerSegmentation:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters    = n_clusters
        self.random_state  = random_state
        self.scaler        = StandardScaler()
        self.model         = None
        self.labels_       = None
        self.feature_cols  = FEATURE_COLS

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0).values
        return self.scaler.fit_transform(X)

    def fit_kmeans(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_features(df)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.labels_ = self.model.fit_predict(X)
        sil = silhouette_score(X, self.labels_)
        db  = davies_bouldin_score(X, self.labels_)
        logger.info(f"KMeans | Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}")
        df = df.copy()
        df["segment"] = self.labels_
        return df, {"silhouette": sil, "davies_bouldin": db}

    def fit_dbscan(self, df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
        X = self._prepare_features(df)
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(X)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise    = list(self.labels_).count(-1)
        logger.info(f"DBSCAN | Clusters: {n_clusters} | Noise points: {n_noise}")
        df = df.copy()
        df["segment"] = self.labels_
        return df, {"n_clusters": n_clusters, "n_noise": n_noise}

    def find_optimal_k(self, df: pd.DataFrame, k_range=range(2, 11)) -> dict:
        """Elbow method + silhouette scores to choose best K."""
        X = self._prepare_features(df)
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X, labels))
        return {"k": list(k_range), "inertia": inertias, "silhouette": sil_scores}

    def label_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign business-friendly labels based on RFM profile per cluster."""
        df = df.copy()
        profile = df.groupby("segment")[["recency","frequency","monetary"]].mean()
        labels = {}
        for seg in profile.index:
            r, f, m = profile.loc[seg, ["recency","frequency","monetary"]]
            if r < 30 and f > 10:
                labels[seg] = "Champions"
            elif r < 60 and f > 5:
                labels[seg] = "Loyal Customers"
            elif r < 90:
                labels[seg] = "Potential Loyalists"
            elif r > 180 and f < 3:
                labels[seg] = "At Risk"
            else:
                labels[seg] = "Lost / Hibernating"
        df["segment_label"] = df["segment"].map(labels)
        return df

    def plot_segments_pca(self, df: pd.DataFrame, X_scaled: np.ndarray = None, save_path: str = None):
        if X_scaled is None:
            X_scaled = self._prepare_features(df)
        pca = PCA(n_components=2, random_state=self.random_state)
        components = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(components[:,0], components[:,1],
                             c=df["segment"], cmap="tab10", alpha=0.6, s=20)
        ax.set_title("Customer Segments (PCA projection)", fontsize=14)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(scatter, ax=ax, label="Segment")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        return fig
