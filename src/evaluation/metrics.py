"""
Evaluation metrics for both segmentation and recommendation models.
Covers: Silhouette Score, Precision@K, Recall@K, NDCG@K, Hit Rate.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_segmentation(X: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "silhouette_score":   round(silhouette_score(X, labels), 4),
        "davies_bouldin":     round(davies_bouldin_score(X, labels), 4),
        "n_clusters":         len(set(labels)) - (1 if -1 in labels else 0),
    }

def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0

def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0

def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    rec_k = recommended[:k]
    dcg   = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant)
    idcg  = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate(recommended: list, relevant: list) -> float:
    return 1.0 if set(recommended) & set(relevant) else 0.0

def evaluate_recommender(test_df: pd.DataFrame, recommender, k: int = 10) -> dict:
    """
    test_df: DataFrame with columns [customer_id, product_id] (held-out purchases)
    """
    precision, recall, ndcg, hr = [], [], [], []
    for cid, group in test_df.groupby("customer_id"):
        relevant = group["product_id"].tolist()
        try:
            recs = recommender.recommend(cid, n=k)["product_id"].tolist()
        except Exception:
            continue
        precision.append(precision_at_k(recs, relevant, k))
        recall.append(recall_at_k(recs, relevant, k))
        ndcg.append(ndcg_at_k(recs, relevant, k))
        hr.append(hit_rate(recs, relevant))

    return {
        f"precision@{k}": round(np.mean(precision), 4),
        f"recall@{k}":    round(np.mean(recall), 4),
        f"ndcg@{k}":      round(np.mean(ndcg), 4),
        "hit_rate":        round(np.mean(hr), 4),
    }
