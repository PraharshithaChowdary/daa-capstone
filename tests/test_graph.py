import pytest
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.transaction_graph import TransactionGraph, Transaction, build_transaction_graph_from_dataframe
from algorithms.graph_mining import GraphMiningAlgorithms, GraphMiningConfig, run_graph_mining_pipeline
from features.graph_features import GraphFeatureExtractor, FeatureConfig
from models.fraud_detector import FraudDetector, ModelConfig, train_fraud_detection_pipeline


class TestTransactionGraph:
    def test_empty_graph(self):
        graph = TransactionGraph()
        stats = graph.get_statistics()
        assert stats['num_nodes'] == 0
        assert stats['num_edges'] == 0
        assert stats['num_transactions'] == 0

    def test_add_transaction(self):
        graph = TransactionGraph()
        tx = Transaction(
            transaction_id='tx1',
            sender_id='ACC_001',
            receiver_id='ACC_002',
            amount=100.0,
            timestamp=datetime.now()
        )
        graph.add_transaction(tx)
        assert graph.graph.number_of_nodes() == 2
        assert graph.graph.number_of_edges() == 1
        assert len(graph.transactions) == 1

    def test_edge_aggregation(self):
        graph = TransactionGraph()
        tx1 = Transaction('tx1', 'ACC_001', 'ACC_002', 100.0, datetime.now())
        tx2 = Transaction('tx2', 'ACC_001', 'ACC_002', 200.0, datetime.now())
        graph.add_transaction(tx1)
        graph.add_transaction(tx2)
        edge_data = graph.graph['ACC_001']['ACC_002']
        assert edge_data['weight'] == 300.0
        assert edge_data['transaction_count'] == 2

    def test_neighbor_discovery(self):
        graph = TransactionGraph()
        tx1 = Transaction('tx1', 'ACC_001', 'ACC_002', 100.0, datetime.now())
        tx2 = Transaction('tx2', 'ACC_002', 'ACC_003', 200.0, datetime.now())
        graph.add_transaction(tx1)
        graph.add_transaction(tx2)
        neighbors = graph.get_neighbors('ACC_001', hops=2)
        assert 'ACC_003' in neighbors
        assert 'ACC_002' in neighbors

    def test_time_window(self):
        graph = TransactionGraph()
        tx1 = Transaction('tx1', 'ACC_001', 'ACC_002', 100.0, datetime(2024, 1, 1))
        tx2 = Transaction('tx2', 'ACC_002', 'ACC_003', 200.0, datetime(2024, 6, 1))
        graph.add_transaction(tx1)
        graph.add_transaction(tx2)
        subgraph = graph.get_time_window_subgraph(datetime(2023, 12, 1), datetime(2024, 2, 1))
        assert subgraph.graph.number_of_nodes() == 2
        assert subgraph.graph.number_of_edges() == 1


class TestGraphMiningAlgorithms:
    def test_pagerank(self):
        G = nx.DiGraph()
        G.add_edge('A', 'B', weight=1)
        G.add_edge('B', 'C', weight=1)
        G.add_edge('C', 'A', weight=1)
        miner = GraphMiningAlgorithms()
        pr = miner.compute_pagerank(G)
        assert len(pr) == 3
        assert all(0 <= v <= 1 for v in pr.values())

    def test_community_detection(self):
        G = nx.karate_club_graph()
        miner = GraphMiningAlgorithms(GraphMiningConfig(min_community_size=2))
        partition = miner.detect_communities_louvain(G)
        assert len(partition) == G.number_of_nodes()
        assert len(set(v for v in partition.values() if v >= 0)) >= 2

    def test_suspicious_patterns(self):
        G = nx.complete_graph(10)
        miner = GraphMiningAlgorithms()
        partition = miner.detect_communities_louvain(G, resolution=1.0)
        patterns = miner.find_suspicious_patterns(G, partition)
        assert len(patterns) > 0

    def test_centrality_computation(self):
        G = nx.path_graph(5)
        miner = GraphMiningAlgorithms()
        centralities = miner.compute_all_centralities(G)
        assert 'pagerank' in centralities
        assert 'degree_centrality' in centralities
        assert 'betweenness_centrality' in centralities


class TestFeatureExtractor:
    def test_node_features(self):
        G = nx.karate_club_graph()
        extractor = GraphFeatureExtractor()
        features = extractor.extract_all_features(G, {})
        assert len(features) == G.number_of_nodes()
        assert len(features.columns) > 0

    def test_feature_normalization(self):
        G = nx.complete_graph(5)
        extractor = GraphFeatureExtractor(FeatureConfig(normalize_features=True))
        features = extractor.extract_all_features(G, {})
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            assert abs(features[numeric_cols].mean()).max() < 5

    def test_subgraph_features(self):
        G = nx.barabasi_albert_graph(20, 3)
        extractor = GraphFeatureExtractor()
        features = extractor.extract_subgraph_features(G, '0')
        assert 'subgraph_nodes' in features
        assert 'subgraph_density' in features


class TestFraudDetector:
    def test_model_training(self):
        n_samples = 100
        n_features = 10
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'f{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.binomial(1, 0.2, n_samples))
        
        detector = FraudDetector(ModelConfig(model_type='xgboost', n_estimators=10))
        detector.fit(X, y)
        assert detector._is_fitted
        assert detector.feature_importance is not None

    def test_prediction(self):
        n_samples = 100
        n_features = 10
        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'f{i}' for i in range(n_features)]
        )
        y_train = pd.Series(np.random.binomial(1, 0.2, n_samples))
        
        detector = FraudDetector(ModelConfig(model_type='xgboost', n_estimators=10))
        detector.fit(X_train, y_train)
        
        X_test = pd.DataFrame(
            np.random.randn(10, n_features),
            columns=[f'f{i}' for i in range(n_features)]
        )
        preds = detector.predict(X_test)
        probas = detector.predict_proba(X_test)
        assert len(preds) == 10
        assert len(probas) == 10
        assert all(0 <= p <= 1 for p in probas)

    def test_optimal_threshold(self):
        n_samples = 200
        n_features = 5
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'f{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.binomial(1, 0.2, n_samples))
        
        detector = FraudDetector(ModelConfig(model_type='logistic_regression'))
        detector.fit(X, y)
        threshold = detector.find_optimal_threshold(X, y, metric='f1')
        assert 0 < threshold < 1

    def test_model_save_load(self, tmp_path):
        n_samples = 50
        n_features = 5
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'f{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.binomial(1, 0.2, n_samples))
        
        detector = FraudDetector(ModelConfig(model_type='xgboost', n_estimators=10))
        detector.fit(X, y)
        
        model_path = os.path.join(tmp_path, 'test_model.pkl')
        detector.save(model_path)
        assert os.path.exists(model_path)
        
        loaded = FraudDetector()
        loaded.load(model_path)
        assert loaded._is_fitted
        assert loaded.best_threshold == detector.best_threshold


class TestPipelineIntegration:
    def test_end_to_end(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from pipeline import FraudDetectionPipeline
        
        pipeline = FraudDetectionPipeline()
        pipeline.generate_synthetic_data()
        pipeline.extract_graph_features()
        pipeline.prepare_labels()
        pipeline.train_model()
        
        summary = pipeline.get_summary()
        assert summary['graph']['nodes'] > 0
        assert summary['graph']['edges'] > 0
        assert summary['model']['trained']

    def test_suspicious_pattern_detection(self):
        from pipeline import FraudDetectionPipeline
        
        pipeline = FraudDetectionPipeline()
        pipeline.generate_synthetic_data()
        pipeline.extract_graph_features()
        
        patterns = pipeline.get_suspicious_patterns()
        assert 'patterns' in patterns
        assert 'suspicious_communities' in patterns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
