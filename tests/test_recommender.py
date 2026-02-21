import pytest
import pandas as pd
import numpy as np
import sys; sys.path.insert(0, ".")
from src.models.recommender import CollaborativeFilteringRecommender, ContentBasedRecommender

def make_transactions():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(n)],
        "customer_id":    np.random.choice([f"C{i}" for i in range(30)], n),
        "product_id":     np.random.choice([f"P{i}" for i in range(20)], n),
        "quantity":       np.random.randint(1, 5, n),
        "revenue":        np.random.uniform(5, 100, n),
    })

def make_products():
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(20)],
        "category":   np.random.choice(["Skincare","Vitamins","Haircare"], 20),
        "price":      np.random.uniform(5, 50, 20),
    })

def test_cf_recommender_returns_results():
    txn   = make_transactions()
    model = CollaborativeFilteringRecommender(min_interactions=1, n_recommendations=5)
    model.fit(txn)
    recs = model.recommend("C0", n=5)
    assert len(recs) <= 5
    assert "product_id" in recs.columns

def test_cb_recommender_returns_similar():
    products = make_products()
    model    = ContentBasedRecommender(n_recommendations=5)
    model.fit(products)
    recs = model.recommend_similar("P0", n=5)
    assert "product_id" in recs.columns
    assert "P0" not in recs["product_id"].values
