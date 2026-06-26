import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)


class FraudDataGenerator:
    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        random.seed(random_state)

    def generate_transaction_data(
        self,
        n_accounts: int = 1000,
        n_transactions: int = 10000,
        fraud_ratio: float = 0.05,
        n_communities: int = 10,
        time_span_days: int = 30
    ) -> pd.DataFrame:
        accounts = [f"ACC_{i:06d}" for i in range(n_accounts)]
        merchants = [f"MER_{i:04d}" for i in range(100)]
        devices = [f"DEV_{i:04d}" for i in range(200)]
        locations = [f"LOC_{i:03d}" for i in range(50)]
        
        communities = self._assign_communities(accounts, n_communities)
        
        start_date = datetime.now() - timedelta(days=time_span_days)
        
        transactions = []
        for i in range(n_transactions):
            is_fraud = self.rng.random() < fraud_ratio
            
            if is_fraud:
                sender, receiver, amount = self._generate_fraud_transaction(
                    accounts, merchants, communities, n_communities
                )
            else:
                sender, receiver, amount = self._generate_normal_transaction(
                    accounts, merchants, communities, n_communities
                )
            
            timestamp = start_date + timedelta(
                seconds=int(self.rng.integers(0, time_span_days * 86400))
            )
            
            transactions.append({
                'transaction_id': f'TX_{i:07d}',
                'sender_id': sender,
                'receiver_id': receiver,
                'amount': round(amount, 2),
                'timestamp': timestamp,
                'merchant_id': self.rng.choice(merchants),
                'device_id': self.rng.choice(devices),
                'location': self.rng.choice(locations),
                'is_fraud': int(is_fraud),
                'account_community': communities.get(sender, -1),
                'merchant_community': communities.get(receiver, -1)
            })
        
        df = pd.DataFrame(transactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated {n_transactions} transactions ({fraud_ratio*100:.1f}% fraud)")
        logger.info(f"  Accounts: {n_accounts}, Communities: {n_communities}")
        
        return df

    def _assign_communities(
        self,
        accounts: List[str],
        n_communities: int
    ) -> dict:
        communities = {}
        for i, account in enumerate(accounts):
            communities[account] = i % n_communities
        return communities

    def _generate_normal_transaction(
        self,
        accounts: List[str],
        merchants: List[str],
        communities: dict,
        n_communities: int
    ) -> Tuple[str, str, float]:
        sender = self.rng.choice(accounts)
        
        if self.rng.random() < 0.7:
            same_comm = [a for a in accounts if communities.get(a) == communities.get(sender) and a != sender]
            receiver = self.rng.choice(same_comm) if same_comm else self.rng.choice([a for a in accounts if a != sender])
        else:
            receiver = self.rng.choice([a for a in accounts if a != sender])
        
        amount = self.rng.lognormal(mean=4.0, sigma=1.0)
        amount = min(max(amount, 1.0), 50000.0)
        
        return sender, receiver, amount

    def _generate_fraud_transaction(
        self,
        accounts: List[str],
        merchants: List[str],
        communities: dict,
        n_communities: int
    ) -> Tuple[str, str, float]:
        fraud_amount = self.rng.choice([
            self.rng.lognormal(mean=6.0, sigma=0.5),
            self.rng.uniform(10000, 100000),
            self.rng.uniform(50, 200)
        ])
        fraud_amount = min(fraud_amount, 100000.0)
        
        fraud_type = self.rng.integers(0, 4)
        
        if fraud_type == 0:
            sender = self.rng.choice(accounts)
            diff_comm = [a for a in accounts if communities.get(a) != communities.get(sender) and a != sender]
            receiver = self.rng.choice(diff_comm) if diff_comm else self.rng.choice([a for a in accounts if a != sender])
        
        elif fraud_type == 1:
            fraud_ring = self.rng.choice(accounts, size=5, replace=False)
            sender = fraud_ring[0]
            receiver = fraud_ring[1]
            fraud_amount = self.rng.uniform(100, 5000)
        
        elif fraud_type == 2:
            new_accounts = [f"FRAUD_ACC_{i:04d}" for i in range(50)]
            sender = self.rng.choice(new_accounts)
            receiver = self.rng.choice(accounts)
            fraud_amount = self.rng.lognormal(mean=5.0, sigma=0.3)
        
        else:
            sender = self.rng.choice(accounts)
            receiver = sender
            while receiver == sender:
                receiver = self.rng.choice(accounts)
            fraud_amount = self.rng.uniform(500, 10000)
            
            if self.rng.random() < 0.3:
                reversal_tx = {
                    'transaction_id': f'FRAUD_REV_{self.rng.integers(1000):04d}',
                    'sender_id': receiver,
                    'receiver_id': sender,
                    'amount': fraud_amount * 0.95,
                    'is_fraud': 1
                }
        
        return sender, receiver, fraud_amount

    def generate_network_data(
        self,
        n_accounts: int = 1000,
        n_relationships: int = 5000,
        fraud_rate: float = 0.05
    ) -> pd.DataFrame:
        accounts = [f"ACC_{i:06d}" for i in range(n_accounts)]
        
        relationships = []
        for i in range(n_relationships):
            is_fraud = self.rng.random() < fraud_rate
            
            if is_fraud:
                source = self.rng.choice(accounts[:int(n_accounts*0.2)])
                target = self.rng.choice(accounts[int(n_accounts*0.8):])
                relationship_type = self.rng.choice(['joint_account', 'shared_device', 'family', 'business'])
            else:
                source = self.rng.choice(accounts)
                target = self.rng.choice([a for a in accounts if a != source])
                relationship_type = self.rng.choice(['friend', 'colleague', 'family', 'business'])
            
            relationships.append({
                'source_account': source,
                'target_account': target,
                'relationship_type': relationship_type,
                'strength': round(self.rng.uniform(0.1, 1.0), 2),
                'is_suspicious': int(is_fraud)
            })
        
        return pd.DataFrame(relationships)


def load_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    generator = FraudDataGenerator()
    transactions = generator.generate_transaction_data(
        n_accounts=500,
        n_transactions=5000,
        fraud_ratio=0.05,
        n_communities=5
    )
    relationships = generator.generate_network_data(n_accounts=500, n_relationships=2000)
    return transactions, relationships


if __name__ == "__main__":
    transactions, relationships = load_demo_data()
    transactions.to_csv('data/transactions.csv', index=False)
    relationships.to_csv('data/relationships.csv', index=False)
    print(f"Generated {len(transactions)} transactions and {len(relationships)} relationships")
    print(f"Fraud rate: {transactions['is_fraud'].mean():.2%}")
