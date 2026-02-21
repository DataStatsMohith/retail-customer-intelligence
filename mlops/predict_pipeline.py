"""Inference pipeline: load saved models and generate predictions."""
import joblib
import pandas as pd
import sys; sys.path.insert(0, ".")
from src.utils.logger import get_logger

logger = get_logger("predict_pipeline")

def load_models():
    seg_model   = joblib.load("mlops/model_registry/segmentation_model.pkl")
    recommender = joblib.load("mlops/model_registry/recommender_model.pkl")
    return seg_model, recommender

def predict_segment(customer_features: pd.DataFrame) -> pd.DataFrame:
    seg_model, _ = load_models()
    result, _ = seg_model.fit_kmeans(customer_features)
    return result

def get_recommendations(customer_id: str, last_product: str = None) -> pd.DataFrame:
    _, recommender = load_models()
    recs = recommender.recommend(customer_id, last_purchased_product=last_product)
    return recs

if __name__ == "__main__":
    _, recommender = load_models()
    cid  = "C00001"
    recs = recommender.recommend(cid)
    print(f"\nTop recommendations for {cid}:")
    print(recs.to_string(index=False))
