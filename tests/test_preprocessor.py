import pytest
import pandas as pd
import numpy as np
import sys; sys.path.insert(0, ".")
from src.data.preprocessor import clean_transactions, add_time_features

def make_sample_df():
    return pd.DataFrame({
        "transaction_id": ["T1","T2","T3","T1","T4"],
        "customer_id":    ["C1","C1","C2","C1","C3"],
        "product_id":     ["P1","P2","P1","P1","P3"],
        "date":           pd.to_datetime(["2024-01-01","2024-01-05","2024-01-07","2024-01-01","2024-01-10"]),
        "quantity":       [1,2,1,1,3],
        "price":          [5.0,10.0,5.0,5.0,15.0],
        "revenue":        [5.0,20.0,5.0,5.0,45.0],
        "is_weekend":     [0,1,0,0,1],
    })

def test_clean_removes_duplicates():
    df = make_sample_df()
    cleaned = clean_transactions(df)
    assert cleaned["transaction_id"].nunique() == len(cleaned)

def test_clean_removes_negative_revenue():
    df = make_sample_df()
    df.loc[0, "revenue"] = -5
    cleaned = clean_transactions(df)
    assert (cleaned["revenue"] > 0).all()

def test_time_features():
    df = make_sample_df()
    df = add_time_features(df)
    assert "month" in df.columns
    assert "is_weekend" in df.columns
