import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import community as community_louvain
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    k_clique_communities,
    asyn_lpa_communities
)
from networkx.algorithms.centrality import (
    betweenness_centrality,
    closeness_centrality,
    eigenvector_centrality,
    degree_centrality
)
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.cluster import (
    clustering,
    triangles,
    square_clustering
)
from networkx.algorithms.core import core_number
from networkx.algorithms.link_analysis import hits
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class GraphMiningConfig:
    pagerank_damping: float = 0.85
    pagerank_max_iter: int = 100
    pagerank_tol: float = 1e-6
    louvain_resolution: float = 1.0
    louvain_random_state: int = 42
    label_prop_max_iter: int = 30
    k_core_k: int = 3
    triangle_counting: bool = True
    min_community_size: int = 3


class GraphMiningAlgorithms:
    def __init__(self, config: GraphMiningConfig = None):
        self.config = config or GraphMiningConfig()
        self.results = {}

    def compute_pagerank(self, graph: nx.Graph, weight: str = 'weight') -> Dict[str, float]:
        try:
            pr = pagerank(
                graph,
                alpha=self.config.pagerank_damping,
                max_iter=self.config.pagerank_max_iter,
                tol=self.config.pagerank_tol,
                weight=weight
            )
            self.results['pagerank'] = pr
            return pr
        except Exception as e:
            logger.error(f"PageRank computation failed: {e}")
            return {}

    def compute_betweenness_centrality(
        self, 
        graph: nx.Graph, 
        weight: str = 'weight',
        k: Optional[int] = None
    ) -> Dict[str, float]:
        try:
            if k and graph.number_of_nodes() > k:
                bc = betweenness_centrality(graph, weight=weight, k=k, seed=self.config.louvain_random_state)
            else:
                bc = betweenness_centrality(graph, weight=weight)
            self.results['betweenness_centrality'] = bc
            return bc
        except Exception as e:
            logger.error(f"Betweenness centrality computation failed: {e}")
            return {}

    def compute_closeness_centrality(
        self, 
        graph: nx.Graph, 
        weight: str = 'weight'
    ) -> Dict[str, float]:
        try:
            cc = closeness_centrality(graph, distance=weight)
            self.results['closeness_centrality'] = cc
            return cc
        except Exception as e:
            logger.error(f"Closeness centrality computation failed: {e}")
            return {}

    def compute_eigenvector_centrality(
        self, 
        graph: nx.Graph, 
        weight: str = 'weight',
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        try:
            ec = eigenvector_centrality(
                graph, 
                weight=weight, 
                max_iter=max_iter, 
                tol=tol
            )
            self.results['eigenvector_centrality'] = ec
            return ec
        except Exception as e:
            logger.error(f"Eigenvector centrality computation failed: {e}")
            return {}

    def compute_degree_centrality(self, graph: nx.Graph) -> Dict[str, float]:
        try:
            dc = degree_centrality(graph)
            self.results['degree_centrality'] = dc
            return dc
        except Exception as e:
            logger.error(f"Degree centrality computation failed: {e}")
            return {}

    def compute_clustering_coefficient(
        self, 
        graph: nx.Graph, 
        weight: str = 'weight'
    ) -> Dict[str, float]:
        try:
            cc = clustering(graph, weight=weight)
            self.results['clustering_coefficient'] = cc
            return cc
        except Exception as e:
            logger.error(f"Clustering coefficient computation failed: {e}")
            return {}

    def compute_triangles(self, graph: nx.Graph) -> Dict[str, int]:
        try:
            if self.config.triangle_counting:
                g = graph.to_undirected() if graph.is_directed() else graph
                tri = triangles(g)
                self.results['triangles'] = tri
                return tri
            return {}
        except Exception as e:
            logger.error(f"Triangle counting failed: {e}")
            return {}

    def compute_k_core(self, graph: nx.Graph) -> Dict[str, int]:
        try:
            kc = core_number(graph)
            self.results['k_core'] = kc
            return kc
        except Exception as e:
            logger.error(f"K-core computation failed: {e}")
            return {}

    def compute_hits(self, graph: nx.Graph, max_iter: int = 100, tol: float = 1e-6) -> Tuple[Dict, Dict]:
        try:
            hubs, authorities = hits(graph, max_iter=max_iter, tol=tol)
            self.results['hubs'] = hubs
            self.results['authorities'] = authorities
            return hubs, authorities
        except Exception as e:
            logger.error(f"HITS computation failed: {e}")
            return {}, {}

    def detect_communities_louvain(
        self, 
        graph: nx.Graph, 
        weight: str = 'weight',
        resolution: float = None
    ) -> Dict[str, int]:
        try:
            if resolution is None:
                resolution = self.config.louvain_resolution
            
            if graph.is_directed():
                graph = graph.to_undirected()
            
            partition = community_louvain.best_partition(
                graph,
                weight=weight,
                resolution=resolution,
                random_state=self.config.louvain_random_state
            )
            
            community_sizes = defaultdict(int)
            for node, comm_id in partition.items():
                community_sizes[comm_id] += 1
            
            filtered_partition = {}
            for node, comm_id in partition.items():
                if community_sizes[comm_id] >= self.config.min_community_size:
                    filtered_partition[node] = comm_id
                else:
                    filtered_partition[node] = -1
            
            self.results['communities_louvain'] = filtered_partition
            self.results['community_sizes'] = dict(community_sizes)
            return filtered_partition
        except Exception as e:
            logger.error(f"Louvain community detection failed: {e}")
            return {}

    def detect_communities_label_propagation(
        self, 
        graph: nx.Graph
    ) -> Dict[str, int]:
        try:
            if graph.is_directed():
                graph = graph.to_undirected()
            
            communities = list(label_propagation_communities(graph))
            
            partition = {}
            for comm_id, community in enumerate(communities):
                if len(community) >= self.config.min_community_size:
                    for node in community:
                        partition[node] = comm_id
                else:
                    for node in community:
                        partition[node] = -1
            
            self.results['communities_label_propagation'] = partition
            return partition
        except Exception as e:
            logger.error(f"Label propagation community detection failed: {e}")
            return {}

    def detect_communities_greedy_modularity(
        self, 
        graph: nx.Graph, 
        weight: str = 'weight'
    ) -> Dict[str, int]:
        try:
            if graph.is_directed():
                graph = graph.to_undirected()
            
            communities = list(greedy_modularity_communities(graph, weight=weight))
            
            partition = {}
            for comm_id, community in enumerate(communities):
                if len(community) >= self.config.min_community_size:
                    for node in community:
                        partition[node] = comm_id
                else:
                    for node in community:
                        partition[node] = -1
            
            self.results['communities_greedy_modularity'] = partition
            return partition
        except Exception as e:
            logger.error(f"Greedy modularity community detection failed: {e}")
            return {}

    def detect_communities_asyn_lpa(
        self, 
        graph: nx.Graph, 
        weight: str = 'weight'
    ) -> Dict[str, int]:
        try:
            if graph.is_directed():
                graph = graph.to_undirected()
            
            communities = list(asyn_lpa_communities(graph, weight=weight))
            
            partition = {}
            for comm_id, community in enumerate(communities):
                if len(community) >= self.config.min_community_size:
                    for node in community:
                        partition[node] = comm_id
                else:
                    for node in community:
                        partition[node] = -1
            
            self.results['communities_asyn_lpa'] = partition
            return partition
        except Exception as e:
            logger.error(f"Asynchronous LPA community detection failed: {e}")
            return {}

    def compute_all_centralities(self, graph: nx.Graph, weight: str = 'weight') -> Dict[str, Dict]:
        results = {}
        results['pagerank'] = self.compute_pagerank(graph, weight)
        results['betweenness_centrality'] = self.compute_betweenness_centrality(graph, weight)
        results['closeness_centrality'] = self.compute_closeness_centrality(graph, weight)
        results['eigenvector_centrality'] = self.compute_eigenvector_centrality(graph, weight)
        results['degree_centrality'] = self.compute_degree_centrality(graph)
        results['clustering_coefficient'] = self.compute_clustering_coefficient(graph, weight)
        results['triangles'] = self.compute_triangles(graph)
        results['k_core'] = self.compute_k_core(graph)
        results['hubs'], results['authorities'] = self.compute_hits(graph)
        return results

    def detect_all_communities(self, graph: nx.Graph, weight: str = 'weight') -> Dict[str, Dict]:
        results = {}
        results['louvain'] = self.detect_communities_louvain(graph, weight)
        results['label_propagation'] = self.detect_communities_label_propagation(graph)
        results['greedy_modularity'] = self.detect_communities_greedy_modularity(graph, weight)
        results['asyn_lpa'] = self.detect_communities_asyn_lpa(graph, weight)
        return results

    def compute_all_features(self, graph: nx.Graph, weight: str = 'weight') -> Dict[str, Dict]:
        centralities = self.compute_all_centralities(graph, weight)
        communities = self.detect_all_communities(graph, weight)
        return {**centralities, **communities}

    def get_community_subgraphs(self, graph: nx.Graph, partition: Dict[str, int]) -> Dict[int, nx.Graph]:
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            if comm_id >= 0:
                communities[comm_id].append(node)
        
        subgraphs = {}
        for comm_id, nodes in communities.items():
            if len(nodes) >= self.config.min_community_size:
                subgraphs[comm_id] = graph.subgraph(nodes).copy()
        return subgraphs

    def compute_community_features(
        self, 
        graph: nx.Graph, 
        partition: Dict[str, int]
    ) -> Dict[int, Dict]:
        subgraphs = self.get_community_subgraphs(graph, partition)
        features = {}
        
        for comm_id, subgraph in subgraphs.items():
            features[comm_id] = {
                'size': subgraph.number_of_nodes(),
                'edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph),
                'avg_clustering': np.mean(list(clustering(subgraph).values())) if subgraph.number_of_nodes() > 0 else 0,
                'avg_degree': np.mean([d for _, d in subgraph.degree()]) if subgraph.number_of_nodes() > 0 else 0,
                'total_weight': sum(d.get('weight', 1) for _, _, d in subgraph.edges(data=True)),
                'is_connected': nx.is_connected(subgraph.to_undirected()) if subgraph.number_of_nodes() > 0 else False
            }
        return features

    def find_suspicious_patterns(
        self, 
        graph: nx.Graph, 
        partition: Dict[str, int] = None
    ) -> List[Dict]:
        patterns = []
        
        if partition is None:
            partition = self.detect_communities_louvain(graph)
        
        community_features = self.compute_community_features(graph, partition)
        
        for comm_id, features in community_features.items():
            if features['density'] > 0.8 and features['size'] >= 3:
                patterns.append({
                    'pattern_type': 'dense_community',
                    'community_id': comm_id,
                    'size': features['size'],
                    'density': features['density'],
                    'description': 'Highly dense community - potential collusion ring'
                })
            
            if features['total_weight'] > 100000 and features['size'] <= 5:
                patterns.append({
                    'pattern_type': 'high_value_small_group',
                    'community_id': comm_id,
                    'size': features['size'],
                    'total_weight': features['total_weight'],
                    'description': 'High transaction volume in small group - potential money laundering'
                })
        
        triangles = self.compute_triangles(graph)
        for node, count in triangles.items():
            if count >= 10:
                patterns.append({
                    'pattern_type': 'high_triangle_count',
                    'node': node,
                    'triangle_count': count,
                    'description': 'Node involved in many triangles - potential circular transactions'
                })
        
        k_core = self.compute_k_core(graph)
        max_k = max(k_core.values()) if k_core else 0
        if max_k >= 5:
            core_nodes = [n for n, k in k_core.items() if k == max_k]
            patterns.append({
                'pattern_type': 'high_k_core',
                'k_value': max_k,
                'nodes': core_nodes[:10],
                'description': f'Found {max_k}-core with {len(core_nodes)} nodes - tightly connected group'
            })
        
        return patterns


def run_graph_mining_pipeline(
    graph: nx.Graph,
    config: GraphMiningConfig = None
) -> Dict[str, Any]:
    miner = GraphMiningAlgorithms(config)
    
    centralities = miner.compute_all_centralities(graph)
    communities = miner.detect_all_communities(graph)
    patterns = miner.find_suspicious_patterns(graph, communities.get('louvain', {}))
    
    return {
        'centralities': centralities,
        'communities': communities,
        'patterns': patterns,
        'community_features': miner.compute_community_features(graph, communities.get('louvain', {}))
    }
