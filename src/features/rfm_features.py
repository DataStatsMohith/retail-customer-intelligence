"""
RFM (Recency, Frequency, Monetary) feature engineering.
Core technique for customer segmentation in retail.
"""
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency   = ("date",    lambda x: (snapshot_date - x.max()).days),
        frequency = ("transaction_id", "nunique"),
        monetary  = ("revenue", "sum"),
    ).reset_index()

    # Log-transform monetary to reduce skew
    rfm["monetary_log"] = np.log1p(rfm["monetary"])
    logger.info(f"RFM computed for {len(rfm):,} customers")
    return rfm

def score_rfm(rfm: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Assign quintile scores (1-5) for each RFM dimension."""
    rfm = rfm.copy()
    rfm["R_score"] = pd.qcut(rfm["recency"],   quantiles, labels=range(quantiles, 0, -1), duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"], quantiles, labels=range(1, quantiles+1),   duplicates="drop").astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"],  quantiles, labels=range(1, quantiles+1),   duplicates="drop").astype(int)
    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["RFM_total"] = rfm[["R_score","F_score","M_score"]].sum(axis=1)
    return rfm

def add_behavioural_features(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Enrich RFM with basket size, product variety, weekend shopping etc."""
    extras = df.groupby("customer_id").agg(
        avg_basket_size   = ("revenue",    "mean"),
        unique_products   = ("product_id", "nunique"),
        unique_categories = ("product_id", "nunique"),   # proxy if category not merged
        weekend_ratio     = ("is_weekend", "mean"),
    ).reset_index()
    return rfm.merge(extras, on="customer_id", how="left")
