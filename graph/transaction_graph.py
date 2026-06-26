import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float
    timestamp: datetime
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[str] = None
    is_fraud: Optional[bool] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AccountNode:
    account_id: str
    node_type: str
    attributes: Dict = field(default_factory=dict)
    features: Dict = field(default_factory=dict)
    community_id: Optional[int] = None
    pagerank_score: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    degree_centrality: float = 0.0
    triangle_count: int = 0
    k_core_number: int = 0


class TransactionGraph:
    def __init__(self, directed: bool = True):
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.accounts: Dict[str, AccountNode] = {}
        self.transactions: List[Transaction] = []
        self.account_to_transactions: Dict[str, List[str]] = defaultdict(list)

    def add_account(self, account_id: str, node_type: str = "account", **attributes):
        if account_id not in self.accounts:
            self.accounts[account_id] = AccountNode(account_id=account_id, node_type=node_type, attributes=attributes)
            self.graph.add_node(account_id, **attributes)
        return self.accounts[account_id]

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)
        
        sender = self.add_account(transaction.sender_id, "sender")
        receiver = self.add_account(transaction.receiver_id, "receiver")
        
        self.account_to_transactions[transaction.sender_id].append(transaction.transaction_id)
        self.account_to_transactions[transaction.receiver_id].append(transaction.transaction_id)
        
        if self.graph.has_edge(transaction.sender_id, transaction.receiver_id):
            edge_data = self.graph[transaction.sender_id][transaction.receiver_id]
            edge_data['weight'] += transaction.amount
            edge_data['transaction_count'] += 1
            edge_data['transactions'].append(transaction.transaction_id)
            edge_data['timestamps'].append(transaction.timestamp)
            edge_data['amounts'].append(transaction.amount)
        else:
            self.graph.add_edge(
                transaction.sender_id,
                transaction.receiver_id,
                weight=transaction.amount,
                transaction_count=1,
                transactions=[transaction.transaction_id],
                timestamps=[transaction.timestamp],
                amounts=[transaction.amount]
            )

    def add_transactions_batch(self, transactions: List[Transaction]):
        for tx in transactions:
            self.add_transaction(tx)

    def get_subgraph(self, node_ids: List[str]) -> 'TransactionGraph':
        subgraph = TransactionGraph(directed=self.graph.is_directed())
        for node_id in node_ids:
            if node_id in self.accounts:
                subgraph.accounts[node_id] = self.accounts[node_id]
        subgraph.graph = self.graph.subgraph(node_ids).copy()
        return subgraph

    def get_neighbors(self, account_id: str, hops: int = 1) -> Set[str]:
        if account_id not in self.graph:
            return set()
        neighbors = set()
        current_level = {account_id}
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                if node in self.graph:
                    next_level.update(self.graph.successors(node))
                    next_level.update(self.graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level
        neighbors.discard(account_id)
        return neighbors

    def get_account_transactions(self, account_id: str) -> List[Transaction]:
        tx_ids = self.account_to_transactions.get(account_id, [])
        return [tx for tx in self.transactions if tx.transaction_id in tx_ids]

    def get_time_window_subgraph(self, start_time: datetime, end_time: datetime) -> 'TransactionGraph':
        subgraph = TransactionGraph(directed=self.graph.is_directed())
        relevant_transactions = [
            tx for tx in self.transactions
            if start_time <= tx.timestamp <= end_time
        ]
        subgraph.add_transactions_batch(relevant_transactions)
        return subgraph

    def get_statistics(self) -> Dict:
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_transactions': len(self.transactions),
            'density': nx.density(self.graph),
            'is_directed': self.graph.is_directed(),
            'num_accounts': len(self.accounts)
        }

    def to_networkx(self) -> nx.Graph:
        return self.graph.copy()

    def to_pandas_edgelist(self) -> pd.DataFrame:
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0),
                'transaction_count': data.get('transaction_count', 0),
            })
        return pd.DataFrame(edges)

    def to_pandas_nodelist(self) -> pd.DataFrame:
        nodes = []
        for node_id, node in self.accounts.items():
            nodes.append({
                'account_id': node_id,
                'node_type': node.node_type,
                **node.attributes,
                **node.features
            })
        return pd.DataFrame(nodes)


class TemporalTransactionGraph(TransactionGraph):
    def __init__(self, time_window_hours: int = 24):
        super().__init__()
        self.time_window_hours = time_window_hours
        self.time_slices: Dict[str, TransactionGraph] = {}

    def add_transaction(self, transaction: Transaction):
        super().add_transaction(transaction)
        
        time_key = transaction.timestamp.strftime(f"%Y-%m-%d-%H")
        if time_key not in self.time_slices:
            self.time_slices[time_key] = TransactionGraph()
        self.time_slices[time_key].add_transaction(transaction)

    def get_time_slice(self, timestamp: datetime) -> TransactionGraph:
        time_key = timestamp.strftime(f"%Y-%m-%d-%H")
        return self.time_slices.get(time_key, TransactionGraph())

    def get_recent_subgraph(self, hours: int = None) -> TransactionGraph:
        if hours is None:
            hours = self.time_window_hours
        if not self.transactions:
            return TransactionGraph()
        
        latest_time = max(tx.timestamp for tx in self.transactions)
        start_time = latest_time - timedelta(hours=hours)
        return self.get_time_window_subgraph(start_time, latest_time)


def build_transaction_graph_from_dataframe(
    df: pd.DataFrame,
    sender_col: str = 'sender_id',
    receiver_col: str = 'receiver_id',
    amount_col: str = 'amount',
    timestamp_col: str = 'timestamp',
    transaction_id_col: str = 'transaction_id',
    **kwargs
) -> TransactionGraph:
    graph = TransactionGraph()
    
    for _, row in df.iterrows():
        tx = Transaction(
            transaction_id=str(row[transaction_id_col]),
            sender_id=str(row[sender_col]),
            receiver_id=str(row[receiver_col]),
            amount=float(row[amount_col]),
            timestamp=pd.to_datetime(row[timestamp_col]),
            merchant_id=str(row.get('merchant_id', '')) if 'merchant_id' in row else None,
            device_id=str(row.get('device_id', '')) if 'device_id' in row else None,
            location=str(row.get('location', '')) if 'location' in row else None,
            is_fraud=bool(row.get('is_fraud', False)) if 'is_fraud' in row else None,
            metadata={k: v for k, v in row.items() 
                     if k not in [sender_col, receiver_col, amount_col, 
                                 timestamp_col, transaction_id_col, 'merchant_id', 
                                 'device_id', 'location', 'is_fraud']}
        )
        graph.add_transaction(tx)
    
    return graph


def build_temporal_graph_from_dataframe(
    df: pd.DataFrame,
    time_window_hours: int = 24,
    **kwargs
) -> TemporalTransactionGraph:
    graph = TemporalTransactionGraph(time_window_hours=time_window_hours)
    return build_transaction_graph_from_dataframe(df, **kwargs)
