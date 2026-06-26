import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, RobustScaler
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    compute_centralities: bool = True
    compute_community_features: bool = True
    compute_subgraph_features: bool = True
    compute_temporal_features: bool = True
    compute_behavioral_features: bool = True
    compute_structural_holes: bool = True
    compute_motif_features: bool = True
    normalize_features: bool = True
    n_jobs: int = -1
    k_hop_neighborhood: int = 2


class GraphFeatureExtractor:
    def __init__(
        self,
        config: FeatureConfig = None,
        scaler: RobustScaler = None
    ):
        self.config = config or FeatureConfig()
        self.scaler = scaler or RobustScaler()
        self._is_fitted = False
        self.feature_names = []

    def extract_node_features(
        self,
        graph: nx.Graph,
        node_ids: List[str] = None,
        centralities: Dict[str, Dict] = None,
        communities: Dict[str, Dict] = None
    ) -> pd.DataFrame:
        if node_ids is None:
            node_ids = list(graph.nodes())
        
        features = {}
        
        for node in node_ids:
            node_features = {}
            
            if centralities:
                for name, values in centralities.items():
                    if node in values:
                        node_features[f'centrality_{name}'] = values[node]
                    else:
                        node_features[f'centrality_{name}'] = 0.0
            
            if communities and 'louvain' in communities:
                node_features['community_id'] = communities['louvain'].get(node, -1)
            
            features[node] = node_features
        
        df = pd.DataFrame.from_dict(features, orient='index')
        df.index.name = 'node_id'
        
        self.feature_names = list(df.columns)
        return df

    def extract_subgraph_features(
        self,
        graph: nx.Graph,
        node_id: str,
        k_hops: int = 2
    ) -> Dict[str, float]:
        subgraph = self._get_k_hop_subgraph(graph, node_id, k_hops)
        
        features = {}
        
        features['subgraph_nodes'] = subgraph.number_of_nodes()
        features['subgraph_edges'] = subgraph.number_of_edges()
        features['subgraph_density'] = nx.density(subgraph)
        
        if subgraph.number_of_nodes() > 1:
            features['subgraph_clustering'] = np.mean(
                list(nx.clustering(subgraph).values())
            )
        else:
            features['subgraph_clustering'] = 0.0
        
        if graph.is_directed():
            in_deg = subgraph.in_degree(node_id, weight='weight')
            out_deg = subgraph.out_degree(node_id, weight='weight')
            features['ego_in_strength'] = in_deg
            features['ego_out_strength'] = out_deg
            features['ego_in_out_ratio'] = in_deg / max(out_deg, 1)
        else:
            features['ego_strength'] = subgraph.degree(node_id, weight='weight')
        
        try:
            neighbors = list(subgraph.neighbors(node_id))
            if neighbors:
                features['neighbor_avg_degree'] = np.mean([
                    subgraph.degree(n) for n in neighbors
                ])
            else:
                features['neighbor_avg_degree'] = 0.0
        except Exception:
            features['neighbor_avg_degree'] = 0.0
        
        return features

    def extract_all_subgraph_features(
        self,
        graph: nx.Graph,
        node_ids: List[str] = None
    ) -> pd.DataFrame:
        if node_ids is None:
            node_ids = list(graph.nodes())
        
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self.extract_subgraph_features)(graph, node, self.config.k_hop_neighborhood)
            for node in node_ids
        )
        
        df = pd.DataFrame(results, index=node_ids)
        return df

    def extract_structural_holes(self, graph: nx.Graph) -> Dict[str, Dict]:
        try:
            if graph.is_directed():
                graph = graph.to_undirected()
            
            holes = {}
            for node in graph.nodes():
                neighbors = set(graph.neighbors(node))
                n = len(neighbors)
                
                if n < 2:
                    holes[node] = {'constraint': 1.0, 'effective_size': 1.0, 'efficiency': 0.0}
                    continue
                
                try:
                    constraint = nx.algorithms.structuralholes.constraint(graph, node)
                    effective_size = n - (2 * sum(
                        1 for v in graph.nodes() if v in neighbors
                        for w in neighbors if w in graph.neighbors(v)
                    ) / n) if n > 0 else 1
                    
                    holes[node] = {
                        'constraint': constraint,
                        'effective_size': effective_size if effective_size > 0 else 1,
                        'efficiency': effective_size / n if n > 0 else 0
                    }
                except Exception:
                    holes[node] = {'constraint': 0.5, 'effective_size': 1.0, 'efficiency': 0.0}
            
            return holes
        except Exception as e:
            logger.error(f"Structural holes extraction failed: {e}")
            return {}

    def extract_motif_features(self, graph: nx.Graph) -> Dict[str, Dict]:
        try:
            is_directed = graph.is_directed()
            features = {}
            
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node)) if hasattr(graph, 'neighbors') else []
                n_neighbors = len(neighbors)
                
                mutual_pairs = 0
                for i, u in enumerate(neighbors):
                    for v in neighbors[i+1:]:
                        if is_directed:
                            if (graph.has_edge(u, v) and graph.has_edge(v, u)):
                                mutual_pairs += 1
                        else:
                            if graph.has_edge(u, v):
                                mutual_pairs += 1
                
                if n_neighbors >= 2:
                    stars = n_neighbors
                    triangles_motif = mutual_pairs
                    chains = (n_neighbors * (n_neighbors - 1) / 2) - mutual_pairs
                else:
                    stars = n_neighbors
                    triangles_motif = 0
                    chains = 0
                
                features[node] = {
                    'motif_stars': stars,
                    'motif_triangles': triangles_motif,
                    'motif_chains': chains,
                    'motif_ratio': triangles_motif / max(stars + chains, 1)
                }
            
            return features
        except Exception as e:
            logger.error(f"Motif extraction failed: {e}")
            return {}

    def extract_all_features(
        self,
        graph: nx.Graph,
        centralities: Dict[str, Dict] = None,
        communities: Dict[str, Dict] = None
    ) -> pd.DataFrame:
        node_ids = list(graph.nodes())
        all_features = []
        
        logger.info(f"Extracting features for {len(node_ids)} nodes")
        
        node_features = self.extract_node_features(graph, node_ids, centralities, communities)
        all_features.append(node_features)
        
        if self.config.compute_subgraph_features and len(node_ids) <= 10000:
            logger.info("Extracting subgraph features")
            subgraph_features = self.extract_all_subgraph_features(graph, node_ids)
            all_features.append(subgraph_features)
        
        if self.config.compute_structural_holes:
            logger.info("Extracting structural holes")
            holes = self.extract_structural_holes(graph)
            holes_df = pd.DataFrame.from_dict(holes, orient='index')
            all_features.append(holes_df)
        
        if self.config.compute_motif_features:
            logger.info("Extracting motif features")
            motifs = self.extract_motif_features(graph)
            motifs_df = pd.DataFrame.from_dict(motifs, orient='index')
            all_features.append(motifs_df)
        
        combined = pd.concat(all_features, axis=1)
        combined = combined.fillna(0)
        
        if self.config.normalize_features:
            combined = self._normalize_features(combined)
        
        self.feature_names = list(combined.columns)
        self._is_fitted = True
        
        return combined

    def _get_k_hop_subgraph(self, graph: nx.Graph, node: str, k: int) -> nx.Graph:
        nodes = {node}
        current_level = {node}
        
        for _ in range(k):
            next_level = set()
            for n in current_level:
                try:
                    next_level.update(graph.neighbors(n))
                except Exception:
                    pass
            nodes.update(next_level)
            current_level = next_level
        
        return graph.subgraph(nodes).copy()

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        if self._is_fitted:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df

    def get_feature_importance_summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None and hasattr(self, '_last_features'):
            df = self._last_features
        
        if df is None:
            return pd.DataFrame()
        
        summary = pd.DataFrame({
            'feature': df.columns,
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max(),
            'non_zero': (df != 0).sum(),
            'zero_ratio': (df == 0).sum() / len(df)
        })
        
        return summary.sort_values('std', ascending=False)


def build_node_feature_matrix(
    graph: nx.Graph,
    extractor: GraphFeatureExtractor = None,
    miner_results: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, List[str]]:
    if extractor is None:
        extractor = GraphFeatureExtractor()
    
    centralities = miner_results.get('centralities', {}) if miner_results else {}
    communities = miner_results.get('communities', {}) if miner_results else {}
    
    features = extractor.extract_all_features(
        graph=graph,
        centralities=centralities,
        communities=communities
    )
    
    return features, extractor.feature_names
