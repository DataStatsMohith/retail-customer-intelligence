from pydantic import BaseModel
from typing import List, Optional

class RecommendationRequest(BaseModel):
    customer_id: str
    last_purchased_product: Optional[str] = None
    n_recommendations: int = 10

class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[dict]

class SegmentResponse(BaseModel):
    customer_id: str
    segment: int
    segment_label: str
    rfm_score: str
