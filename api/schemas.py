from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class TransactionRequest(BaseModel):
    sender_id: str
    receiver_id: str
    amount: float = Field(..., gt=0)
    timestamp: datetime = None
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict = {}


class FraudPredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    fraud_prediction: bool
    risk_level: str
    graph_scores: Dict[str, float]
    explanation: List[str]


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[FraudPredictionResponse]
    summary: Dict[str, Any]


class ModelMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    avg_precision: float
    optimal_threshold: float


class GraphStatisticsResponse(BaseModel):
    num_nodes: int
    num_edges: int
    num_transactions: int
    density: float
    communities: int
    suspicious_patterns: List[Dict]


class FeatureImportanceResponse(BaseModel):
    features: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_nodes: int
    num_edges: int
    uptime_seconds: float
