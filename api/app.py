"""
FastAPI REST API — serves segmentation and recommendation predictions.
Demonstrates production deployment capability (MLOps best practices).
"""
from fastapi import FastAPI, HTTPException
from api.schemas import RecommendationRequest, RecommendationResponse, SegmentResponse
from mlops.predict_pipeline import load_models, get_recommendations
from src.utils.logger import get_logger
import sys; sys.path.insert(0, ".")

logger = get_logger("api")
app    = FastAPI(title="Boots Customer Segmentation & Recommendation API", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "recommendation-engine"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    try:
        recs = get_recommendations(request.customer_id, request.last_purchased_product)
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=recs.to_dict(orient="records")
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segments/{customer_id}")
def get_segment(customer_id: str):
    import pandas as pd
    try:
        df = pd.read_csv("data/processed/segmented_customers.csv")
        row = df[df["customer_id"] == customer_id]
        if row.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        r = row.iloc[0]
        return SegmentResponse(
            customer_id=customer_id,
            segment=int(r.get("segment", -1)),
            segment_label=str(r.get("segment_label", "Unknown")),
            rfm_score=str(r.get("RFM_score", "N/A"))
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Run training pipeline first")
