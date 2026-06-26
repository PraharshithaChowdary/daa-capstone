import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import pickle
import json
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_type: str = 'xgboost'
    threshold: float = 0.5
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    scale_pos_weight: float = 1.0
    early_stopping_rounds: int = 20
    use_gpu: bool = False
    n_jobs: int = -1


class FraudDetector:
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.training_history = []
        self._is_fitted = False
        self.best_threshold = self.config.threshold

    def _build_model(self):
        if self.config.model_type == 'xgboost':
            params = {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'random_state': self.config.random_state,
                'scale_pos_weight': self.config.scale_pos_weight,
                'use_label_encoder': False,
                'eval_metric': 'aucpr',
                'n_jobs': self.config.n_jobs
            }
            if self.config.use_gpu:
                params['tree_method'] = 'gpu_hist'
            return xgb.XGBClassifier(**params)
        
        elif self.config.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                class_weight='balanced'
            )
        
        elif self.config.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                random_state=self.config.random_state
            )
        
        elif self.config.model_type == 'logistic_regression':
            return LogisticRegression(
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                max_iter=1000
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,
        feature_names: List[str] = None
    ):
        logger.info(f"Training {self.config.model_type} model")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if validation_data:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            
            if self.config.model_type == 'xgboost':
                self.model = self._build_model()
                self.model.fit(
                    X_scaled, y,
                    eval_set=[(X_scaled, y), (X_val_scaled, y_val)],
                    verbose=False
                )
            else:
                self.model = self._build_model()
                self.model.fit(X_scaled, y)
        else:
            self.model = self._build_model()
            self.model.fit(X_scaled, y)
        
        self._is_fitted = True
        
        if hasattr(self.model, 'feature_importances_'):
            if feature_names:
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                self.feature_importance = pd.DataFrame({
                    'feature': [f'f{i}' for i in range(len(self.model.feature_importances_))],
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        self.training_history.append({
            'timestamp': datetime.now(),
            'samples': len(X),
            'features': X.shape[1],
            'model_type': self.config.model_type
        })
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return (self.predict_proba(X_scaled) >= self.best_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)[:, 1]
        return self.model.decision_function(X_scaled)

    def predict_with_score(self, X: pd.DataFrame) -> pd.DataFrame:
        scores = self.predict_proba(X)
        predictions = (scores >= self.best_threshold).astype(int)
        
        return pd.DataFrame({
            'fraud_probability': scores,
            'fraud_prediction': predictions,
            'risk_level': pd.cut(
                scores,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['low', 'medium', 'high']
            )
        })

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = None
    ) -> Dict[str, float]:
        if threshold:
            self.best_threshold = threshold
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'avg_precision': average_precision_score(y, y_proba)
        }
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics

    def find_optimal_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'f1'
    ) -> float:
        y_proba = self.predict_proba(X)
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
        
        if metric == 'f1':
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores[:-1])
        elif metric == 'recall':
            best_idx = np.argmax(recalls[:-1])
        elif metric == 'precision':
            best_idx = np.argmax(precisions[:-1])
        else:
            best_idx = np.argmax(f1_scores[:-1])
        
        self.best_threshold = thresholds[best_idx]
        return self.best_threshold

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5
    ) -> Dict[str, List[float]]:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_state)
        
        metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1_score': [], 'roc_auc': [], 'avg_precision': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.fit(X_train, y_train)
            fold_metrics = self.evaluate(X_val, y_val)
            
            for metric_name, value in fold_metrics.items():
                if metric_name in metrics:
                    metrics[metric_name].append(value)
        
        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.feature_importance is None:
            return pd.DataFrame()
        return self.feature_importance.head(top_n)

    def save(self, path: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'best_threshold': self.best_threshold,
            'is_fitted': self._is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.best_threshold = model_data['best_threshold']
        self._is_fitted = model_data['is_fitted']
        logger.info(f"Model loaded from {path}")
        return self


class EnsembleFraudDetector(FraudDetector):
    def __init__(self, config: ModelConfig = None):
        super().__init__(config)
        self.models = {}
        self.weights = {}

    def add_model(self, name: str, detector: FraudDetector, weight: float = 1.0):
        self.models[name] = detector
        self.weights[name] = weight

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None
    ):
        for name, detector in self.models.items():
            logger.info(f"Training ensemble model: {name}")
            detector.fit(X, y, validation_data)
        self.scaler.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.best_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("No models in ensemble")
        
        weighted_probas = np.zeros(len(X))
        total_weight = sum(self.weights.values())
        
        for name, detector in self.models.items():
            probas = detector.predict_proba(X)
            weighted_probas += probas * (self.weights[name] / total_weight)
        
        return weighted_probas

    def predict_with_score(self, X: pd.DataFrame) -> pd.DataFrame:
        scores = self.predict_proba(X)
        predictions = (scores >= self.best_threshold).astype(int)
        
        model_predictions = {}
        for name, detector in self.models.items():
            model_predictions[f'{name}_prediction'] = detector.predict(X)
        
        result = pd.DataFrame({
            'fraud_probability': scores,
            'fraud_prediction': predictions,
            'risk_level': pd.cut(
                scores,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['low', 'medium', 'high']
            ),
            **model_predictions
        })
        
        result['ensemble_consensus'] = result[[
            c for c in result.columns if c.endswith('_prediction')
        ]].mean(axis=1)
        
        return result


class AnomalyDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            max_samples='auto',
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)


def train_fraud_detection_pipeline(
    features: pd.DataFrame,
    labels: pd.Series,
    model_config: ModelConfig = None,
    test_size: float = 0.2,
    use_ensemble: bool = False
) -> Tuple[FraudDetector, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    if use_ensemble:
        detector = EnsembleFraudDetector(model_config)
        xgb_detector = FraudDetector(ModelConfig(model_type='xgboost'))
        rf_detector = FraudDetector(ModelConfig(model_type='random_forest'))
        lr_detector = FraudDetector(ModelConfig(model_type='logistic_regression'))
        
        detector.add_model('xgboost', xgb_detector, weight=0.5)
        detector.add_model('random_forest', rf_detector, weight=0.3)
        detector.add_model('logistic_regression', lr_detector, weight=0.2)
    else:
        detector = FraudDetector(model_config)
    
    detector.fit(
        X_train, y_train,
        validation_data=(X_test, y_test)
    )
    
    optimal_threshold = detector.find_optimal_threshold(X_test, y_test)
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    
    metrics = detector.evaluate(X_test, y_test)
    
    logger.info(f"Test metrics: {metrics}")
    
    return detector, metrics
