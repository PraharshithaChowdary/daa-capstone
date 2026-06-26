import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import os
import sys

from graph.transaction_graph import (
    TransactionGraph, Transaction, build_transaction_graph_from_dataframe
)
from algorithms.graph_mining import (
    GraphMiningAlgorithms, GraphMiningConfig, run_graph_mining_pipeline
)
from features.graph_features import GraphFeatureExtractor, FeatureConfig, build_node_feature_matrix
from models.fraud_detector import (
    FraudDetector, ModelConfig, train_fraud_detection_pipeline, AnomalyDetector
)
from data.data_generator import load_demo_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    def __init__(self, config_path: str = None):
        self.graph = TransactionGraph()
        self.miner = GraphMiningAlgorithms()
        self.feature_extractor = GraphFeatureExtractor()
        self.detector = FraudDetector()
        self.anomaly_detector = AnomalyDetector()
        self.config = self._load_config(config_path)
        self.training_history = []
        self.feature_names = []

    def _load_config(self, path: str = None) -> Dict:
        import yaml
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def load_data(
        self,
        data_path: str,
        sender_col: str = 'sender_id',
        receiver_col: str = 'receiver_id',
        amount_col: str = 'amount',
        timestamp_col: str = 'timestamp',
        transaction_id_col: str = 'transaction_id'
    ) -> 'FraudDetectionPipeline':
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        self.raw_data = df
        self.graph = build_transaction_graph_from_dataframe(
            df,
            sender_col=sender_col,
            receiver_col=receiver_col,
            amount_col=amount_col,
            timestamp_col=timestamp_col,
            transaction_id_col=transaction_id_col
        )
        
        logger.info(f"Loaded {len(df)} transactions into graph with "
                    f"{self.graph.graph.number_of_nodes()} nodes and "
                    f"{self.graph.graph.number_of_edges()} edges")
        return self

    def generate_synthetic_data(
        self,
        n_accounts: int = 1000,
        n_transactions: int = 10000,
        fraud_ratio: float = 0.05
    ) -> 'FraudDetectionPipeline':
        logger.info("Generating synthetic transaction data")
        transactions, relationships = load_demo_data()
        self.raw_data = transactions
        self.graph = build_transaction_graph_from_dataframe(transactions)
        return self

    def extract_graph_features(self) -> 'FraudDetectionPipeline':
        logger.info("Running graph mining algorithms")
        self.mining_results = run_graph_mining_pipeline(self.graph.graph)
        
        logger.info("Extracting node features")
        self.feature_df, self.feature_names = build_node_feature_matrix(
            self.graph.graph,
            self.feature_extractor,
            self.mining_results
        )
        
        logger.info(f"Extracted {len(self.feature_names)} features for {len(self.feature_df)} nodes")
        return self

    def prepare_labels(self, label_source: str = 'is_fraud') -> 'FraudDetectionPipeline':
        logger.info("Preparing labels")
        
        if hasattr(self, 'raw_data') and label_source in self.raw_data.columns:
            node_labels = {}
            for _, row in self.raw_data.iterrows():
                sender, receiver, label = row['sender_id'], row['receiver_id'], row[label_source]
                node_labels[sender] = max(node_labels.get(sender, 0), label)
                node_labels[receiver] = max(node_labels.get(receiver, 0), label)
            
            self.labels = pd.Series({
                node: node_labels.get(node, 0)
                for node in self.feature_df.index
            })
        else:
            self.labels = pd.Series(0, index=self.feature_df.index)
        
        fraud_count = self.labels.sum()
        logger.info(f"Labels: {fraud_count} fraud, {len(self.labels) - fraud_count} normal")
        return self

    def train_model(
        self,
        test_size: float = 0.2,
        use_ensemble: bool = False
    ) -> 'FraudDetectionPipeline':
        logger.info("Training fraud detection model")
        
        X = self.feature_df
        y = self.labels
        
        if y.sum() == 0:
            logger.warning("No fraud labels found. Using anomaly detection instead.")
            self.anomaly_detector.fit(X)
            self._is_anomaly = True
        else:
            self.detector, self.train_metrics = train_fraud_detection_pipeline(
                X, y,
                test_size=test_size,
                use_ensemble=use_ensemble
            )
            self._is_anomaly = False
        
        self.training_history.append({
            'timestamp': datetime.now(),
            'features': len(self.feature_names),
            'nodes': len(self.feature_df),
            'fraud_ratio': float(y.mean()),
            'test_size': test_size,
            'use_ensemble': use_ensemble
        })
        
        return self

    def predict(self, transaction: Transaction) -> Dict[str, Any]:
        sender_features = self.feature_extractor.extract_all_features(
            self.graph.graph,
            self.mining_results.get('centralities'),
            self.mining_results.get('communities')
        )
        
        relevant_nodes = [transaction.sender_id, transaction.receiver_id]
        node_features = sender_features[
            sender_features.index.isin(relevant_nodes)
        ]
        
        if node_features.empty:
            return {
                'fraud_probability': 0.0,
                'fraud_prediction': False,
                'risk_level': 'low'
            }
        
        avg_features = node_features.mean().to_frame().T
        
        for col in self.feature_names:
            if col not in avg_features.columns:
                avg_features[col] = 0
        avg_features = avg_features[self.feature_names]
        
        if self._is_anomaly:
            score = self.anomaly_detector.anomaly_score(avg_features)
            fraud_prob = float(score[0])
            fraud_pred = fraud_prob >= 0.8
        else:
            result = self.detector.predict_with_score(avg_features)
            fraud_prob = float(result['fraud_probability'].iloc[0])
            fraud_pred = bool(result['fraud_prediction'].iloc[0])
        
        if fraud_prob >= 0.7:
            risk = 'high'
        elif fraud_prob >= 0.3:
            risk = 'medium'
        else:
            risk = 'low'
        
        return {
            'fraud_probability': fraud_prob,
            'fraud_prediction': fraud_pred,
            'risk_level': risk
        }

    def get_suspicious_patterns(self) -> List[Dict]:
        patterns = self.mining_results.get('patterns', [])
        
        communities = self.mining_results.get('communities', {}).get('louvain', {})
        community_features = self.miner.compute_community_features(
            self.graph.graph, communities
        )
        
        suspicious_communities = []
        for comm_id, features in community_features.items():
            if features['density'] > 0.7 or features['total_weight'] > 100000:
                suspicious_communities.append({
                    'community_id': comm_id,
                    **features
                })
        
        return {
            'patterns': patterns,
            'suspicious_communities': suspicious_communities
        }

    def get_summary(self) -> Dict:
        if self.graph.graph.number_of_nodes() == 0:
            return {"status": "no_data"}
        
        summary = {
            'graph': {
                'nodes': self.graph.graph.number_of_nodes(),
                'edges': self.graph.graph.number_of_edges(),
                'density': nx.density(self.graph.graph),
                'transactions': len(self.graph.transactions)
            },
            'features': {
                'count': len(self.feature_names),
                'names': self.feature_names[:10]
            },
            'model': {
                'trained': self.detector._is_fitted,
                'type': self.detector.config.model_type,
                'anomaly_mode': getattr(self, '_is_anomaly', False)
            },
            'training_history': self.training_history
        }
        
        if self.detector.feature_importance is not None:
            summary['top_features'] = self.detector.get_feature_importance(5).to_dict('records')
        
        return summary

    def save_model(self, path: str):
        self.detector.save(path)
        logger.info(f"Pipeline model saved to {path}")

    def load_model(self, path: str):
        self.detector.load(path)
        logger.info(f"Pipeline model loaded from {path}")
        return self

    def run_full_pipeline(
        self,
        data_path: str = None,
        generate_data: bool = False,
        label_source: str = 'is_fraud',
        test_size: float = 0.2,
        use_ensemble: bool = False
    ) -> Dict:
        if data_path:
            self.load_data(data_path)
        elif generate_data:
            self.generate_synthetic_data()
        else:
            raise ValueError("Either data_path or generate_data must be specified")
        
        self.extract_graph_features()
        self.prepare_labels(label_source)
        self.train_model(test_size, use_ensemble)
        
        suspicious = self.get_suspicious_patterns()
        
        return {
            'summary': self.get_summary(),
            'metrics': getattr(self, 'train_metrics', None),
            'suspicious_patterns': suspicious,
            'feature_importance': self.detector.get_feature_importance(10).to_dict('records')
            if self.detector.feature_importance is not None else []
        }


def run_demo():
    pipeline = FraudDetectionPipeline()
    
    result = pipeline.run_full_pipeline(
        generate_data=True,
        label_source='is_fraud',
        test_size=0.2,
        use_ensemble=True
    )
    
    print("\n=== Fraud Detection Pipeline Results ===")
    print(f"Graph: {result['summary']['graph']['nodes']} nodes, {result['summary']['graph']['edges']} edges")
    
    if result['metrics']:
        print(f"\nModel Performance:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    patterns = result.get('suspicious_patterns', {})
    print(f"\nSuspicious Patterns Found: {len(patterns.get('patterns', []))}")
    for pattern in patterns.get('patterns', [])[:5]:
        print(f"  - {pattern.get('description', 'Unknown pattern')}")
    
    top_features = result.get('feature_importance', [])
    if top_features:
        print(f"\nTop Features:")
        for feat in top_features[:5]:
            print(f"  {feat.get('feature', '?')}: {feat.get('importance', 0):.4f}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = run_demo()
