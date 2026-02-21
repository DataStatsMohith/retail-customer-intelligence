"""Data loading utilities — wraps CSV/SQL sources."""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_transactions(path: str = "data/raw/transactions.csv") -> pd.DataFrame:
    logger.info(f"Loading transactions from {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"Loaded {len(df):,} rows")
    return df

def load_products(path: str = "data/raw/products.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def load_customers(path: str = "data/raw/customers.csv") -> pd.DataFrame:
    return pd.read_csv(path)
