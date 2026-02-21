import pytest
import pandas as pd
import numpy as np
import sys; sys.path.insert(0, ".")
from src.features.rfm_features import compute_rfm, score_rfm
from src.models.segmentation import CustomerSegmentation

def make_rfm():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id":     [f"C{i}" for i in range(n)],
        "recency":         np.random.randint(1, 365, n),
        "frequency":       np.random.randint(1, 50, n),
        "monetary":        np.random.uniform(10, 1000, n),
        "monetary_log":    np.log1p(np.random.uniform(10, 1000, n)),
        "avg_basket_size": np.random.uniform(5, 100, n),
        "unique_products": np.random.randint(1, 30, n),
        "weekend_ratio":   np.random.uniform(0, 1, n),
    })
    return df

def test_kmeans_produces_labels():
    rfm = make_rfm()
    model = CustomerSegmentation(n_clusters=3)
    result, metrics = model.fit_kmeans(rfm)
    assert "segment" in result.columns
    assert result["segment"].nunique() == 3

def test_silhouette_positive():
    rfm = make_rfm()
    model = CustomerSegmentation(n_clusters=3)
    _, metrics = model.fit_kmeans(rfm)
    assert metrics["silhouette"] > 0
