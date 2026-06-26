import networkx as nx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_nested_root = os.path.join(_repo_root, 'fraud_detection')
if os.path.isdir(_nested_root):
    _repo_root = _nested_root
sys.path.insert(0, _repo_root)

from api.schemas import (
    TransactionRequest, FraudPredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, ModelMetricsResponse, GraphStatisticsResponse,
    FeatureImportanceResponse, HealthResponse
)
from graph.transaction_graph import TransactionGraph, Transaction, build_transaction_graph_from_dataframe
from algorithms.graph_mining import GraphMiningAlgorithms, GraphMiningConfig, run_graph_mining_pipeline
from features.graph_features import GraphFeatureExtractor, FeatureConfig
from models.fraud_detector import FraudDetector, ModelConfig, train_fraud_detection_pipeline
from models.fraud_detector import AnomalyDetector, EnsembleFraudDetector

logger = logging.getLogger(__name__)

app_state = {
    'graph': None,
    'detector': None,
    'anomaly_detector': None,
    'feature_extractor': None,
    'miner': None,
    'start_time': time.time(),
    'predictions_history': []
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Fraud Detection API")
    initialize_system()
    yield
    logger.info("Shutting down Fraud Detection API")


app = FastAPI(
    title="Fraud Detection using Graph Mining",
    description="API for detecting fraudulent patterns using graph mining algorithms",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_system():
    app_state['graph'] = TransactionGraph()
    app_state['miner'] = GraphMiningAlgorithms()
    app_state['feature_extractor'] = GraphFeatureExtractor()
    app_state['detector'] = FraudDetector(ModelConfig(model_type='xgboost'))
    app_state['anomaly_detector'] = AnomalyDetector()
    logger.info("System initialized")


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Fraud Detection using Graph Mining API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    graph = app_state['graph']
    return HealthResponse(
        status="healthy",
        model_loaded=app_state['detector']._is_fitted if app_state['detector'] else False,
        num_nodes=graph.graph.number_of_nodes() if graph else 0,
        num_edges=graph.graph.number_of_edges() if graph else 0,
        uptime_seconds=time.time() - app_state['start_time']
    )


@app.post("/transactions", tags=["Transactions"])
async def add_transaction(tx: TransactionRequest):
    graph = app_state['graph']
    transaction = Transaction(
        transaction_id=f"tx_{len(graph.transactions) + 1}_{int(time.time())}",
        sender_id=tx.sender_id,
        receiver_id=tx.receiver_id,
        amount=tx.amount,
        timestamp=tx.timestamp or datetime.now(),
        merchant_id=tx.merchant_id,
        device_id=tx.device_id,
        location=tx.location,
        metadata=tx.metadata
    )
    graph.add_transaction(transaction)
    return {"message": "Transaction added", "transaction_id": transaction.transaction_id}


@app.post("/predict/single", response_model=FraudPredictionResponse, tags=["Prediction"])
async def predict_single(tx: TransactionRequest, background_tasks: BackgroundTasks):
    graph = app_state['graph']
    
    transaction = Transaction(
        transaction_id=f"tx_{len(graph.transactions) + 1}_{int(time.time())}",
        sender_id=tx.sender_id,
        receiver_id=tx.receiver_id,
        amount=tx.amount,
        timestamp=tx.timestamp or datetime.now(),
        merchant_id=tx.merchant_id,
        device_id=tx.device_id,
        location=tx.location,
        metadata=tx.metadata
    )
    graph.add_transaction(transaction)
    
    try:
        mining_results = run_graph_mining_pipeline(graph.graph)
        
        feature_df = app_state['feature_extractor'].extract_all_features(
            graph.graph,
            centralities=mining_results['centralities'],
            communities=mining_results['communities']
        )
        
        fraud_features = []
        for node_id, row in feature_df.iterrows():
            if node_id in [transaction.sender_id, transaction.receiver_id]:
                fraud_features.append(row)
        
        if fraud_features:
            avg_features = pd.DataFrame([f.to_dict() for f in fraud_features]).mean().to_frame().T
            needed_cols = getattr(app_state['detector'], 'feature_names', feature_df.columns)
            for col in needed_cols:
                if col not in avg_features.columns:
                    avg_features[col] = 0
            avg_features = avg_features[needed_cols]
            
            prediction_result = app_state['detector'].predict_with_score(avg_features)
            fraud_prob = float(prediction_result['fraud_probability'].iloc[0])
            fraud_pred = bool(prediction_result['fraud_prediction'].iloc[0])
            risk_level = str(prediction_result['risk_level'].iloc[0])
        else:
            fraud_prob = 0.0
            fraud_pred = False
            risk_level = 'low'
        
        graph_scores = {}
        for node_id, row in feature_df.iterrows():
            if node_id in [transaction.sender_id, transaction.receiver_id]:
                for col in row.index:
                    if 'centrality' in col or 'pagerank' in col:
                        graph_scores[f'{node_id}_{col}'] = float(row[col])
        
        explanation = generate_explanation(transaction, mining_results, fraud_prob)
        
        if fraud_pred:
            background_tasks.add_task(alert_fraud_detected, transaction, fraud_prob, explanation)
        
        return FraudPredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=fraud_prob,
            fraud_prediction=fraud_pred,
            risk_level=risk_level,
            graph_scores=graph_scores,
            explanation=explanation
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return FraudPredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=0.0,
            fraud_prediction=False,
            risk_level='low',
            graph_scores={},
            explanation=[f"Error during prediction: {str(e)}"]
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchPredictionRequest):
    results = []
    
    for tx in batch.transactions:
        result = await predict_single(tx, None)
        results.append(result)
    
    fraud_count = sum(1 for r in results if r.fraud_prediction)
    high_risk_count = sum(1 for r in results if r.risk_level == 'high')
    
    return BatchPredictionResponse(
        predictions=results,
        summary={
            'total': len(results),
            'fraud_detected': fraud_count,
            'high_risk': high_risk_count,
            'fraud_rate': fraud_count / max(len(results), 1)
        }
    )


@app.get("/graph/statistics", response_model=GraphStatisticsResponse, tags=["Graph"])
async def get_graph_statistics():
    graph = app_state['graph']
    
    try:
        mining_results = run_graph_mining_pipeline(graph.graph)
        communities = mining_results.get('communities', {}).get('louvain', {})
        num_communities = len(set(v for v in communities.values() if v >= 0))
        patterns = mining_results.get('patterns', [])
    except Exception as e:
        num_communities = 0
        patterns = []
    
    return GraphStatisticsResponse(
        num_nodes=graph.graph.number_of_nodes(),
        num_edges=graph.graph.number_of_edges(),
        num_transactions=len(graph.transactions),
        density=nx.density(graph.graph) if graph.graph.number_of_nodes() > 0 else 0,
        communities=num_communities,
        suspicious_patterns=patterns
    )


@app.get("/model/metrics", response_model=ModelMetricsResponse, tags=["Model"])
async def get_model_metrics():
    detector = app_state['detector']
    if not detector._is_fitted:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    return ModelMetricsResponse(
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        roc_auc=0.0,
        avg_precision=0.0,
        optimal_threshold=detector.best_threshold
    )


@app.get("/model/feature-importance", response_model=FeatureImportanceResponse, tags=["Model"])
async def get_feature_importance(top_n: int = 20):
    detector = app_state['detector']
    if not detector._is_fitted:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    importance = detector.get_feature_importance(top_n)
    return FeatureImportanceResponse(
        features=importance.to_dict('records')
    )


@app.post("/model/train", tags=["Model"])
async def train_model(data_path: str = None):
    if data_path:
        df = pd.read_csv(data_path)
        graph = build_transaction_graph_from_dataframe(df)
        app_state['graph'] = graph
        
        mining_results = run_graph_mining_pipeline(graph.graph)
        features = app_state['feature_extractor'].extract_all_features(
            graph.graph,
            centralities=mining_results['centralities'],
            communities=mining_results['communities']
        )
        
        labels = df.groupby(df.columns[0]).agg({'is_fraud': 'first'}).iloc[:, 0]
        node_labels = []
        for node in features.index:
            node_labels.append(labels.get(node, 0))
        y = pd.Series(node_labels, index=features.index)
        
        detector, metrics = train_fraud_detection_pipeline(features, y)
        app_state['detector'] = detector
        
        return {"message": "Model trained successfully", "metrics": metrics}
    else:
        raise HTTPException(status_code=400, detail="Data path is required")


@app.get("/history", tags=["Monitoring"])
async def get_prediction_history(limit: int = 100):
    history = app_state.get('predictions_history', [])
    return history[-limit:]


def generate_explanation(
    transaction: Transaction,
    mining_results: Dict,
    fraud_prob: float
) -> List[str]:
    explanations = []
    
    if fraud_prob >= 0.7:
        explanations.append(f"High fraud probability ({fraud_prob:.2%})")
    elif fraud_prob >= 0.5:
        explanations.append(f"Moderate fraud probability ({fraud_prob:.2%})")
    
    centralities = mining_results.get('centralities', {})
    for node_id in [transaction.sender_id, transaction.receiver_id]:
        pr = centralities.get('pagerank', {}).get(node_id, 0)
        if pr > 0.1:
            explanations.append(f"Node {node_id} has high PageRank ({pr:.4f})")
        
        bc = centralities.get('betweenness_centrality', {}).get(node_id, 0)
        if bc > 0.1:
            explanations.append(f"Node {node_id} is a central connector (betweenness: {bc:.4f})")
    
    patterns = mining_results.get('patterns', [])
    for pattern in patterns[:3]:
        if pattern.get('node') in [transaction.sender_id, transaction.receiver_id] or \
           'community_id' in pattern:
            explanations.append(f"Suspicious pattern: {pattern.get('description', '')}")
    
    return explanations if explanations else ["No suspicious patterns detected"]


async def alert_fraud_detected(transaction: Transaction, probability: float, explanation: List[str]):
    logger.warning(
        f"FRAUD ALERT: {transaction.transaction_id} | "
        f"Prob: {probability:.4f} | "
        f"Sender: {transaction.sender_id} | "
        f"Receiver: {transaction.receiver_id} | "
        f"Amount: {transaction.amount}"
    )
    app_state['predictions_history'].append({
        'transaction_id': transaction.transaction_id,
        'fraud_probability': probability,
        'timestamp': datetime.now().isoformat(),
        'amount': transaction.amount,
        'sender': transaction.sender_id,
        'receiver': transaction.receiver_id,
        'explanation': explanation
    })


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
