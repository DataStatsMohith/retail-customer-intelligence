"""Cleans and prepares raw transaction data for feature engineering."""
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls, duplicates, negative values."""
    initial = len(df)
    df = df.dropna(subset=["customer_id", "product_id", "date", "revenue"])
    df = df.drop_duplicates(subset=["transaction_id"])
    df = df[df["revenue"] > 0]
    logger.info(f"Cleaned: {initial:,} -> {len(df):,} rows ({initial - len(df):,} removed)")
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features useful for segmentation."""
    df = df.copy()
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"]     = df["date"].dt.quarter
    return df

def get_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    return df["date"].max() + pd.Timedelta(days=1)
