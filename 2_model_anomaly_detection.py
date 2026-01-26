# -*- coding: utf-8 -*-
"""
Mô-đun Phát hiện Bất thường
Xử lý phát hiện bất thường / ngoại lệ trong kế toán và các tín hiệu rủi ro gian lận tiềm ẩn.
Ghi chú:
- Được khung trong "bất thường/ngoại lệ" chứ không phải gian lận chắc chắn; dấu tăng/giảm có thể là lỗi dữ liệu.
- Mức nhiễm mặc định được giữ khiêm tốn (10%) để giảm cảnh báo giả; điều chỉnh theo độ chịu đựng rủi ro.
- Trong sản xuất, cân nhắc yêu cầu nhiều cờ đỏ trước khi gắn nhãn rủi ro gian lận cao.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
import importlib.util

# Suppress sklearn robust covariance warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Determinant has increased')
warnings.filterwarnings('ignore', category=UserWarning, message='The covariance matrix associated to your dataset is not full rank')

# Import AuditLogger from 6_logging_config.py
spec = importlib.util.spec_from_file_location("logging_config", "6_logging_config.py")
logging_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_module)
AuditLogger = logging_module.AuditLogger

logger = AuditLogger.get_logger(__name__)

class AnomalyDetectionModel:
    """
    Mô hình Phát hiện Bất thường
    Xác định các mẫu bất thường và tiềm ẩn gian lận trong dữ liệu ngân hàng
    """
    
    def __init__(self):
        self.model_name = "Anomaly Detection Model"
        self.models = {}
        self.scaler = StandardScaler()
        self.anomalies = []
        self.anomaly_scores = {}
        self.feature_names = []
        self.feature_importance = {}  # Track which features drive anomalies
        self.peer_benchmarks = {}  # Store peer comparison data for explanations
        # Peer clustering (KMeans) for fair comparisons
        self.peer_cluster_model = None
        self.peer_cluster_scaler = StandardScaler()
        self.peer_cluster_features = []  # feature names used for clustering
        self.peer_cluster_mapping = {}   # bank_id -> cluster label
        self.cluster_peer_benchmarks = {}  # cluster_id -> {feature: stats}
        # Per-model feature transforms (e.g., PCA for EllipticEnvelope)
        self.model_transforms = {}
        
    def train_models(self, training_data: pd.DataFrame, 
                    contamination: float = 0.1,
                    **kwargs) -> Dict:
        """
        Train multiple anomaly detection models
        
        Args:
            training_data: Historical data from multiple banks
            contamination: Expected proportion of outliers (tune per risk appetite)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing training results
        """
        # Prepare features for anomaly detection
        features = self._prepare_anomaly_features(training_data)
        
        if features.empty:
            return {'status': 'failed', 'reason': 'No valid features'}
        
        # Remove duplicate columns BEFORE storing feature names and fitting scaler
        features = features.loc[:, ~features.columns.duplicated(keep='first')]
        
        # Store feature names (now deduplicated)
        self.feature_names = features.columns.tolist()
        
        # Build peer clusters (size/business-model oriented) before benchmarks
        try:
            self._train_kmeans_peer_clustering(training_data)
        except Exception as e:
            logger.warning(f"Peer clustering skipped: {str(e)}")

        # Calculate overall peer benchmarks for regulatory explanations
        self._calculate_peer_benchmarks(features)

        # Calculate cluster-specific peer benchmarks if clustering available
        try:
            self._calculate_cluster_peer_benchmarks(features, training_data)
        except Exception as e:
            logger.warning(f"Cluster benchmarks skipped: {str(e)}")

        # Clean data: replace infinity with NaN, then handle NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column median
        for col in features.columns:
            if features[col].isna().any():
                features[col].fillna(features[col].median(), inplace=True)
        
        # Double-check for any remaining NaN values
        features = features.fillna(0)
        
        # Scale features - use .values to avoid sklearn feature name validation
        X_scaled = self.scaler.fit_transform(features.values)
        
        training_results = {}
        
        # Train Isolation Forest
        training_results['isolation_forest'] = self._train_isolation_forest(
            X_scaled, contamination
        )
        
        # Train DBSCAN for clustering-based anomaly detection
        training_results['dbscan'] = self._train_dbscan(X_scaled)
        
        # Train One-Class SVM
        training_results['one_class_svm'] = self._train_one_class_svm(
            X_scaled, contamination
        )
        
        # Train Local Outlier Factor
        training_results['local_outlier_factor'] = self._train_local_outlier_factor(
            X_scaled, contamination
        )
        
        # Train Elliptic Envelope
        training_results['elliptic_envelope'] = self._train_elliptic_envelope(
            X_scaled, contamination
        )
        
        return training_results

    def _prepare_peer_cluster_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare aggregated bank-level features for peer clustering.

        Prefers size and business-model indicators to group similar banks.
        If 'bank_id' exists, aggregates per bank; else clusters per-row fallback.
        """
        # Candidate features for clustering (pick those present and numeric)
        candidate_cols = [
            'total_assets', 'total_loans', 'total_deposits',
            'loan_to_deposit_ratio', 'net_interest_margin',
            'liquidity_ratio', 'capital_adequacy_ratio',
            'sector_concentration_hhi', 'obs_to_assets_ratio',
            'derivatives_to_assets_ratio', 'external_rating_score'
        ]
        cols = [c for c in candidate_cols if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
        if not cols:
            return pd.DataFrame()

        # Aggregate to bank-level if possible
        if 'bank_id' in data.columns:
            agg_df = data.groupby('bank_id')[cols].median().reset_index()
        else:
            # Fallback: use rows directly (no aggregation)
            agg_df = data[cols].copy()
            agg_df['bank_id'] = data.get('bank_id', pd.RangeIndex(len(agg_df)))

        # Clean values
        agg_df[cols] = agg_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return agg_df[['bank_id'] + cols]

    def _select_optimal_k(self, X: np.ndarray, k_min: int, k_max: int) -> int:
        """
        Dynamically select k using silhouette and Calinski-Harabasz scores.
        Prioritize silhouette; use CH as tiebreaker.
        """
        best_k = k_min
        best_sil = -1.0
        best_ch = -1.0
        for k in range(k_min, k_max + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = km.fit_predict(X)
                # Require at least 2 distinct clusters
                if len(set(labels)) < 2:
                    continue
                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                # Better silhouette, or tie with better CH
                if sil > best_sil or (abs(sil - best_sil) < 1e-6 and ch > best_ch):
                    best_k = k
                    best_sil = sil
                    best_ch = ch
            except Exception:
                continue
        return max(best_k, k_min)

    def _train_kmeans_peer_clustering(self, training_data: pd.DataFrame) -> Dict:
        """
        Train KMeans-based peer clustering at bank level with dynamic k.
        Stores mapping bank_id -> cluster label and cluster model/scaler.
        """
        agg_df = self._prepare_peer_cluster_features(training_data)
        if agg_df.empty:
            raise ValueError("No suitable features for peer clustering")

        self.peer_cluster_features = [c for c in agg_df.columns if c != 'bank_id']
        X = agg_df[self.peer_cluster_features].values
        
        # Clean data: replace infinity with NaN, then handle NaN values
        X = np.where(np.isfinite(X), X, np.nan)
        X_df = pd.DataFrame(X, columns=self.peer_cluster_features)
        for col in X_df.columns:
            if X_df[col].isna().any():
                X_df[col].fillna(X_df[col].median(), inplace=True)
        X_df = X_df.fillna(0)
        X = X_df.values
        
        X_scaled = self.peer_cluster_scaler.fit_transform(X)

        n_banks = X_scaled.shape[0]
        # Choose k in [2, min(8, n_banks-1)]
        k_min, k_max = 2, max(2, min(8, n_banks - 1))
        if k_max < 2:
            raise ValueError("Insufficient banks for clustering")

        k_opt = self._select_optimal_k(X_scaled, k_min, k_max)
        km = KMeans(n_clusters=k_opt, random_state=42, n_init='auto')
        labels = km.fit_predict(X_scaled)

        # Persist
        self.peer_cluster_model = km
        # Map bank_id -> label
        bank_ids = agg_df['bank_id'].tolist()
        self.peer_cluster_mapping = {str(b): int(lbl) for b, lbl in zip(bank_ids, labels)}

        return {
            'trained': True,
            'k': k_opt,
            'n_banks': n_banks
        }

    def _calculate_cluster_peer_benchmarks(self, features_df: pd.DataFrame, training_df: pd.DataFrame) -> None:
        """
        Compute peer benchmarks per cluster using anomaly feature set.
        """
        if not self.peer_cluster_mapping:
            return
        # Align rows to clusters using bank_id
        if 'bank_id' not in training_df.columns:
            return
        bank_ids_series = training_df['bank_id'].astype(str)
        cluster_ids = bank_ids_series.map(self.peer_cluster_mapping)
        # For each cluster, compute stats across features_df rows
        cluster_benchmarks = {}
        for cluster in sorted(set([c for c in cluster_ids.dropna().unique()])):
            mask = cluster_ids == cluster
            sub = features_df.loc[mask]
            bm = {}
            for col in sub.columns:
                if pd.api.types.is_numeric_dtype(sub[col]):
                    values = sub[col].dropna()
                    if len(values) > 0:
                        bm[col] = {
                            'mean': float(values.mean()),
                            'median': float(values.median()),
                            'std': float(values.std()),
                            'p25': float(values.quantile(0.25)),
                            'p75': float(values.quantile(0.75)),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
            cluster_benchmarks[int(cluster)] = bm
        self.cluster_peer_benchmarks = cluster_benchmarks
    
    def _prepare_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection
        
        Args:
            data: Input dataframe
            
        Returns:
            DataFrame with selected features
        """
        # Collect columns to concatenate instead of assigning one at a time
        cols_to_add = []
        
        # Financial ratios
        ratio_cols = [col for col in data.columns if any(x in col for x in 
                     ['ratio', 'roa', 'roe', 'margin', 'coverage'])]
        cols_to_add.extend(ratio_cols)
        
        # Growth metrics
        growth_cols = [col for col in data.columns if '_growth' in col]
        cols_to_add.extend(growth_cols)
        
        # Transaction volumes and patterns
        volume_cols = [col for col in data.columns if any(x in col for x in
                      ['total_', 'volume', 'count', 'number_of'])]
        cols_to_add.extend([col for col in volume_cols if col in data.columns])
        
        # Build features DataFrame all at once using concat (avoids fragmentation)
        features = data[cols_to_add].copy() if cols_to_add else pd.DataFrame()

        # Add expert-driven composite red flags if source columns exist
        # 1) Rapid asset growth with declining income margin
        if all(col in data.columns for col in ['total_assets_growth', 'net_interest_margin']):
            features['flag_asset_surge_income_drop'] = (
                (data['total_assets_growth'] > 0.25).astype(float) *
                (data['net_interest_margin'] < data['net_interest_margin'].median()).astype(float)
            )
        # 2) Asset growth with falling ROA
        if all(col in data.columns for col in ['total_assets_growth', 'roa']):
            features['flag_asset_surge_roa_drop'] = (
                (data['total_assets_growth'] > 0.20).astype(float) *
                (data['roa'] < data['roa'].median()).astype(float)
            )
        # 3) Asset growth with falling ROE
        if all(col in data.columns for col in ['total_assets_growth', 'roe']):
            features['flag_asset_surge_roe_drop'] = (
                (data['total_assets_growth'] > 0.20).astype(float) *
                (data['roe'] < data['roe'].median()).astype(float)
            )
        # Placeholder for operational risk signals if available (e.g., employee_turnover, op_losses)
        if all(col in data.columns for col in ['employee_turnover_rate']):
            features['flag_high_turnover'] = (data['employee_turnover_rate'] > 0.15).astype(float)
        
        # Fill missing values
        features = features.fillna(0)
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _train_isolation_forest(self, X: np.ndarray, contamination: float) -> Dict:
        """
        Train Isolation Forest model
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of outliers
            
        Returns:
            Training results
        """
        try:
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            model.fit(X)
            self.models['isolation_forest'] = model
            
            # Get anomaly predictions for training data
            predictions = model.predict(X)
            anomaly_count = (predictions == -1).sum()
            
            return {
                'trained': True,
                'contamination': contamination,
                'anomalies_found': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(X))
            }
        except Exception as e:
            return {'trained': False, 'error': str(e)}
    
    def _train_dbscan(self, X: np.ndarray) -> Dict:
        """
        Train DBSCAN clustering for density-based anomaly detection
        
        DBSCAN identifies core, border, and noise (outlier) points.
        Points labeled -1 are considered anomalies (noise points in low-density regions).
        
        Args:
            X: Feature matrix
            
        Returns:
            Training results including clusters found and anomaly count
        """
        try:
            # Adaptive eps and min_samples based on data dimensionality
            # eps: controls neighborhood radius (0.5 is reasonable for standardized data)
            # min_samples: minimum points in neighborhood (typically 2*dim, capped at 10)
            n_features = X.shape[1]
            min_samples = min(10, max(5, 2 * n_features))
            
            model = DBSCAN(
                eps=0.5,  # Radius of neighborhood (tunable)
                min_samples=min_samples,  # Minimum points to form core point
                metric='euclidean',
                n_jobs=-1  # Parallel processing
            )
            labels = model.fit_predict(X)
            self.models['dbscan'] = model
            
            # Points with label -1 are noise/anomalies
            anomaly_count = (labels == -1).sum()
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            return {
                'trained': True,
                'clusters_found': int(n_clusters),
                'anomalies_found': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(X)) if len(X) > 0 else 0,
                'core_points': int((labels != -1).sum()),
                'eps': 0.5,
                'min_samples': min_samples
            }
        except Exception as e:
            return {'trained': False, 'error': str(e)}
    
    def _train_one_class_svm(self, X: np.ndarray, contamination: float) -> Dict:
        """
        Train One-Class SVM model
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of outliers
            
        Returns:
            Training results
        """
        try:
            # Limit sample size for SVM if too large
            if len(X) > 1000:
                rng = np.random.RandomState(42)  # Seeded random generator for reproducibility
                sample_idx = rng.choice(len(X), 1000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='auto',
                random_state=42  # Added for reproducibility
            )
            model.fit(X_sample)
            self.models['one_class_svm'] = model
            
            predictions = model.predict(X_sample)
            anomaly_count = (predictions == -1).sum()
            
            return {
                'trained': True,
                'samples_used': len(X_sample),
                'anomalies_found': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(X_sample))
            }
        except Exception as e:
            return {'trained': False, 'error': str(e)}
    
    def _train_local_outlier_factor(self, X: np.ndarray, 
                                   contamination: float) -> Dict:
        """
        Train Local Outlier Factor model for local density-based anomaly detection
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of outliers
            
        Returns:
            Training results including anomaly count and rate
        """
        try:
            # LocalOutlierFactor with novelty=True for out-of-sample prediction
            model = LocalOutlierFactor(
                contamination=contamination,
                novelty=True,  # Allow predict on new data
                n_neighbors=min(20, max(5, int(0.1 * len(X)))),  # Adaptive neighbors
                random_state=42  # For reproducibility
            )
            model.fit(X)
            self.models['local_outlier_factor'] = model
            
            # Get predictions for training data
            predictions = model.predict(X)
            anomaly_count = (predictions == -1).sum()
            
            return {
                'trained': True,
                'contamination': contamination,
                'anomalies_found': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(X))
            }
        except Exception as e:
            return {'trained': False, 'error': str(e)}
    
    def _train_elliptic_envelope(self, X: np.ndarray, 
                               contamination: float) -> Dict:
        """
        Train Elliptic Envelope model
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of outliers
            
        Returns:
            Training results
        """
        try:
            # If covariance matrix is not full rank, reduce dimensionality
            X_input = X
            try:
                rank = np.linalg.matrix_rank(X)
                if rank < X.shape[1]:
                    # Reduce to full-rank via PCA and persist transform
                    pca = PCA(n_components=rank, random_state=42)
                    X_input = pca.fit_transform(X)
                    self.model_transforms['elliptic_envelope'] = pca
                    logger.info("Applied PCA to ensure full-rank covariance for EllipticEnvelope")
                else:
                    self.model_transforms['elliptic_envelope'] = None
            except Exception:
                # If PCA fails, proceed without transform
                self.model_transforms['elliptic_envelope'] = None

            # Fit with conservative support_fraction to reduce robust covariance warnings
            model = EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                support_fraction=0.8
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model.fit(X_input)
                # If determinant warning occurs, refit with higher support_fraction
                det_warns = [msg for msg in w
                             if issubclass(msg.category, RuntimeWarning)
                             and 'Determinant has increased' in str(msg.message)]
                if det_warns:
                    logger.info("Refitting EllipticEnvelope with support_fraction=0.9 due to determinant warning")
                    model = EllipticEnvelope(
                        contamination=contamination,
                        random_state=42,
                        support_fraction=0.9
                    )
                    model.fit(X_input)
            self.models['elliptic_envelope'] = model
            
            predictions = model.predict(X_input)
            anomaly_count = (predictions == -1).sum()
            
            return {
                'trained': True,
                'anomalies_found': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(X_input))
            }
        except Exception as e:
            return {'trained': False, 'error': str(e)}
    
    def detect_anomalies(self, input_data: pd.DataFrame, 
                        voting_threshold: float = 0.6,
                        min_red_flags: int = 1) -> Dict:
        """
        Detect accounting anomalies using ensemble voting across multiple models.
        Combines probabilistic models (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope)
        with density-based clustering (DBSCAN) for robust anomaly detection.
        
        Args:
            input_data: Data to check for anomalies
            voting_threshold: Proportion of models that must agree (0-1, default 0.6 = 60%)
            min_red_flags: Minimum number of expert red flags required to escalate
            
        Returns:
            Dictionary containing anomaly detection results with ensemble scores
        """
        # Prepare features
        features = self._prepare_anomaly_features(input_data)
        
        if features.empty:
            return {'status': 'failed', 'reason': 'No valid features'}
        
        # Remove duplicate columns (keep first occurrence)
        features = features.loc[:, ~features.columns.duplicated(keep='first')]
        
        # Align feature columns with those used during training
        if hasattr(self, 'feature_names') and self.feature_names:
            features = features.reindex(columns=self.feature_names, fill_value=0)

        # Clean data: replace infinity with NaN, then handle NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        for col in features.columns:
            if features[col].isna().any():
                features[col].fillna(features[col].median(), inplace=True)
        features = features.fillna(0)

        # Count expert red flags (columns starting with flag_)
        flag_cols = [c for c in features.columns if c.startswith('flag_')]
        flag_counts = features[flag_cols].sum(axis=1) if flag_cols else pd.Series(0, index=features.index)
        # If no flag columns exist, do not block anomalies on red-flag threshold
        effective_min_red_flags = 0 if not flag_cols else min_red_flags

        # Scale features - convert to numpy to avoid sklearn feature name validation issues
        X_scaled = self.scaler.transform(features.values)
        
        # Collect predictions from all models
        model_predictions = {}
        anomaly_scores_array = []
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'dbscan':
                    # DBSCAN: special handling - use fit_predict on training data concept
                    # For new data, we use decision_function approximation
                    dbscan_labels = model.fit_predict(X_scaled)
                    predictions = np.where(dbscan_labels == -1, -1, 1)  # -1 for noise, 1 for core/border
                    model_predictions['dbscan'] = predictions
                    binary_predictions = (predictions == 1).astype(int)
                    anomaly_scores_array.append(1 - binary_predictions)
                    
                elif model_name == 'local_outlier_factor':
                    # LOF with novelty enabled
                    predictions = model.predict(X_scaled)
                    model_predictions['local_outlier_factor'] = predictions
                    binary_predictions = (predictions == 1).astype(int)
                    anomaly_scores_array.append(1 - binary_predictions)
                    
                else:
                    # Standard models with predict method
                    X_for_predict = X_scaled
                    # Apply any stored transform (e.g., PCA for EllipticEnvelope)
                    transform = self.model_transforms.get(model_name)
                    if transform is not None:
                        try:
                            X_for_predict = transform.transform(X_scaled)
                        except Exception as e:
                            logger.warning(f"Transform for {model_name} failed: {str(e)}")
                            X_for_predict = X_scaled
                    predictions = model.predict(X_for_predict)
                    model_predictions[model_name] = predictions
                    
                    # Convert to binary (1 = normal, 0 = anomaly)
                    binary_predictions = (predictions == 1).astype(int)
                    anomaly_scores_array.append(1 - binary_predictions)
                
            except Exception as e:
                logger.warning(f"Error in {model_name}: {str(e)}")
        
        # Ensemble voting across all models
        if anomaly_scores_array:
            anomaly_scores_matrix = np.column_stack(anomaly_scores_array)
            ensemble_scores = anomaly_scores_matrix.mean(axis=1)
            n_models = len(anomaly_scores_array)
            
            # Identify anomalies based on voting threshold AND minimum red flags
            is_anomaly = (ensemble_scores >= voting_threshold) & (flag_counts >= effective_min_red_flags)
            
            # Store results
            self.anomaly_scores = {
                'ensemble_scores': ensemble_scores.tolist(),
                'is_anomaly': is_anomaly.tolist(),
                'model_predictions': model_predictions,
                'n_models_voted': n_models
            }
            
            # Create detailed anomaly report with explanations
            anomaly_indices = np.where(is_anomaly)[0]
            self.anomalies = []

            # Determine peer cluster for this bank (if available)
            bank_cluster = None
            try:
                if 'bank_id' in input_data.columns and self.peer_cluster_mapping:
                    bids = input_data['bank_id'].astype(str).unique().tolist()
                    if bids:
                        bank_cluster = self.peer_cluster_mapping.get(str(bids[0]))
                elif self.peer_cluster_model is not None and self.peer_cluster_features:
                    # Fallback: infer cluster from available features in input_data (median)
                    available = [c for c in self.peer_cluster_features if c in input_data.columns]
                    if available:
                        vec = input_data[available].median().values.reshape(1, -1)
                        # Clean data before scaling
                        vec = np.where(np.isfinite(vec), vec, np.nan)
                        vec_df = pd.DataFrame(vec, columns=available)
                        for col in vec_df.columns:
                            if vec_df[col].isna().any():
                                vec_df[col].fillna(vec_df[col].median(), inplace=True)
                        vec_df = vec_df.fillna(0)
                        vec = vec_df.values
                        vec_scaled = self.peer_cluster_scaler.transform(vec)
                        bank_cluster = int(self.peer_cluster_model.predict(vec_scaled)[0])
            except Exception:
                bank_cluster = None
            
            for idx in anomaly_indices:
                # Generate explanation for this specific anomaly
                explanation = self._generate_anomaly_explanation(
                    idx, 
                    input_data, 
                    ensemble_scores[idx],
                    model_predictions,
                    anomaly_scores_matrix[idx] if idx < len(anomaly_scores_matrix) else None,
                    bank_cluster
                )
                
                anomaly_record = {
                    'index': int(idx),
                    'anomaly_score': float(ensemble_scores[idx]),
                    'severity': self._classify_anomaly_severity(ensemble_scores[idx]),
                    'red_flag_count': int(flag_counts.iloc[idx]) if idx < len(flag_counts) else 0,
                    'models_voting_anomaly': sum([1 for score in anomaly_scores_matrix[idx] if score > 0.5]),
                    'data': input_data.iloc[idx].to_dict() if idx < len(input_data) else {},
                    'explanation': explanation,  # NEW: Regulatory transparency
                    'contributing_factors': explanation.get('top_factors', []),
                    'regulatory_narrative': explanation.get('narrative', '')
                }
                self.anomalies.append(anomaly_record)
            
            return {
                'total_records': len(input_data),
                'anomalies_detected': len(anomaly_indices),
                'anomaly_rate': len(anomaly_indices) / len(input_data),
                'anomalies': self.anomalies,
                'ensemble_scores': ensemble_scores.tolist(),
                'flag_counts': flag_counts.tolist(),
                'models_used': list(model_predictions.keys()),
                'voting_threshold': voting_threshold,
                'min_red_flags': min_red_flags
            }
        else:
            return {'status': 'failed', 'reason': 'No models available for prediction'}
    
    def _calculate_peer_benchmarks(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate peer benchmarks for each feature to enable regulatory comparisons
        
        Args:
            data: Training dataset with multiple banks
            
        Returns:
            Dictionary of feature benchmarks (mean, std, percentiles)
        """
        benchmarks = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].dropna()
                if len(values) > 0:
                    benchmarks[col] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'p25': float(values.quantile(0.25)),
                        'p75': float(values.quantile(0.75)),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
        
        self.peer_benchmarks = benchmarks
        return benchmarks
    
    def _generate_anomaly_explanation(self, 
                                      index: int, 
                                      data: pd.DataFrame, 
                                      anomaly_score: float,
                                      model_predictions: Dict,
                                      individual_scores: np.ndarray = None,
                                      bank_cluster: Optional[int] = None) -> Dict:
        """
        Generate regulatory-compliant explanation for why a data point was flagged as anomalous
        
        Args:
            index: Index of the anomalous record
            data: Full dataset
            anomaly_score: Overall anomaly score
            model_predictions: Predictions from each model
            individual_scores: Individual anomaly scores from each model
            
        Returns:
            Detailed explanation with contributing factors
        """
        explanation = {
            'top_factors': [],
            'narrative': '',
            'model_consensus': {},
            'deviation_details': {},
            'peer_group': None
        }
        
        if index >= len(data):
            return explanation
        
        record = data.iloc[index]
        
        # Identify which models flagged this record
        models_flagging = []
        if individual_scores is not None:
            for i, (model_name, _) in enumerate(model_predictions.items()):
                if i < len(individual_scores) and individual_scores[i] > 0.5:
                    models_flagging.append(model_name)
        
        explanation['model_consensus'] = {
            'models_flagging': models_flagging,
            'consensus_strength': len(models_flagging) / max(len(model_predictions), 1)
        }
        
        # Choose benchmarks: cluster-specific if available, else overall
        benchmarks_source = self.peer_benchmarks
        if bank_cluster is not None and bank_cluster in self.cluster_peer_benchmarks:
            explanation['peer_group'] = int(bank_cluster)
            bm_candidate = self.cluster_peer_benchmarks.get(int(bank_cluster), {})
            # Use cluster benchmarks when available; fallback per-feature to overall
            def get_benchmark(feat):
                return bm_candidate.get(feat, self.peer_benchmarks.get(feat))
        else:
            def get_benchmark(feat):
                return self.peer_benchmarks.get(feat)

        # Calculate deviations from chosen peer benchmarks for each feature
        deviations = []
        
        for feature in self.feature_names:
            bm = get_benchmark(feature)
            if feature in record.index and bm is not None:
                value = record[feature]
                benchmark = bm
                
                # Skip if value is NaN
                if pd.isna(value):
                    continue
                
                # Calculate percentage deviation from peer mean
                if benchmark['std'] > 0:
                    z_score = (value - benchmark['mean']) / benchmark['std']
                    pct_deviation = ((value - benchmark['mean']) / abs(benchmark['mean']) * 100) if benchmark['mean'] != 0 else 0
                    
                    # Flag significant deviations (>2 standard deviations)
                    if abs(z_score) > 2:
                        deviations.append({
                            'feature': feature,
                            'value': float(value),
                            'peer_mean': benchmark['mean'],
                            'peer_median': benchmark['median'],
                            'z_score': float(z_score),
                            'pct_deviation': float(pct_deviation),
                            'severity': 'extreme' if abs(z_score) > 3 else 'notable'
                        })
        
        # Sort deviations by absolute z-score to identify top contributors
        deviations.sort(key=lambda x: abs(x['z_score']), reverse=True)
        explanation['top_factors'] = deviations[:5]  # Top 5 contributing factors
        explanation['deviation_details'] = {d['feature']: d for d in deviations}
        
        # Generate regulatory narrative
        narrative_parts = []
        
        if models_flagging:
            model_list = ', '.join(models_flagging)
            narrative_parts.append(f"Models flagging anomaly: {model_list} (consensus: {explanation['model_consensus']['consensus_strength']:.1%}).")
        
        if deviations:
            top_3 = deviations[:3]
            deviation_desc = []
            
            for dev in top_3:
                feature_display = dev['feature'].replace('_', ' ').title()
                direction = "above" if dev['pct_deviation'] > 0 else "below"
                
                deviation_desc.append(
                    f"{feature_display}: {dev['value']:.4f} vs peer average {dev['peer_mean']:.4f} "
                    f"({direction} by {abs(dev['pct_deviation']):.1f}%, z-score: {dev['z_score']:.2f})"
                )
            
            narrative_parts.append(
                f"Key outliers identified: {'; '.join(deviation_desc)}."
            )
        
        if not narrative_parts:
            narrative_parts.append("Anomaly detected based on ensemble model analysis, but specific factors could not be determined.")
        
        explanation['narrative'] = ' '.join(narrative_parts)
        
        return explanation
    
    def _classify_anomaly_severity(self, score: float) -> str:
        """
        Classify anomaly severity based on score
        
        Args:
            score: Anomaly score (0-1)
            
        Returns:
            Severity classification
        """
        if score >= 0.9:
            return "CRITICAL"
        elif score >= 0.7:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def detect_fraud_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Detect heuristic fraud/red-flag patterns (not definitive fraud detection)
        
        Args:
            data: Transaction or financial data
            
        Returns:
            Dictionary of detected fraud patterns
        """
        fraud_patterns = []
        
        # Pattern 1: Unusual transaction amounts
        if 'transaction_amount' in data.columns:
            amounts = data['transaction_amount']
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            
            unusual_amounts = amounts[(amounts > mean_amount + 3 * std_amount) | 
                                     (amounts < mean_amount - 3 * std_amount)]
            
            if len(unusual_amounts) > 0:
                fraud_patterns.append({
                    'pattern': 'unusual_transaction_amounts',
                    'count': len(unusual_amounts),
                    'severity': 'MEDIUM',
                    'description': 'Transactions with amounts far from normal range'
                })
        
        # Pattern 2: Rapid succession of transactions
        if 'timestamp' in data.columns:
            data_sorted = data.sort_values('timestamp')
            time_diffs = data_sorted['timestamp'].diff()
            
            rapid_transactions = time_diffs[time_diffs < pd.Timedelta(minutes=1)]
            
            if len(rapid_transactions) > 5:
                fraud_patterns.append({
                    'pattern': 'rapid_transaction_sequence',
                    'count': len(rapid_transactions),
                    'severity': 'HIGH',
                    'description': 'Multiple transactions in very short time period'
                })
        
        # Pattern 3: Unusual growth rates
        growth_cols = [col for col in data.columns if '_growth' in col]
        for col in growth_cols:
            if col in data.columns:
                growth_values = data[col].abs()
                if (growth_values > 1.0).any():  # >100% growth
                    fraud_patterns.append({
                        'pattern': f'extreme_growth_{col}',
                        'count': (growth_values > 1.0).sum(),
                        'severity': 'HIGH',
                        'description': f'Extreme growth rate in {col}'
                    })
        
        return {
            'patterns_detected': len(fraud_patterns),
            'patterns': fraud_patterns
        }
    
    def analyze_anomaly_trends(self, time_series_data: pd.DataFrame,
                              time_column: str = 'period') -> Dict:
        """
        Analyze anomaly trends over time using vectorized operations
        
        Args:
            time_series_data: Time series data with period information
            time_column: Name of time column (default: 'period')
            
        Returns:
            Dictionary containing trend analysis with period-wise anomaly rates
        """
        if time_column not in time_series_data.columns:
            return {'status': 'failed', 'reason': f'Time column {time_column} not found'}
        
        # Detect anomalies for all periods in one vectorized call
        result = self.detect_anomalies(time_series_data)
        
        if not result or 'is_anomaly' not in result:
            return {'status': 'no_data'}
        
        # Assign anomaly flags to rows and aggregate by period
        time_series_data_copy = time_series_data.copy()
        time_series_data_copy['is_anomaly'] = result.get('is_anomaly', [False] * len(time_series_data))
        
        # Vectorized aggregation by period
        grouped = time_series_data_copy.groupby(time_column).agg(
            count=('is_anomaly', 'sum'),
            total=('is_anomaly', 'size'),
            rate=('is_anomaly', 'mean')
        ).reset_index()
        grouped = grouped.rename(columns={time_column: 'period'})
        grouped['count'] = grouped['count'].astype(int)
        
        anomaly_counts = grouped.to_dict('records')
        
        # Analyze trend
        if anomaly_counts:
            counts_df = pd.DataFrame(anomaly_counts)
            
            trend_analysis = {
                'periods_analyzed': len(anomaly_counts),
                'total_anomalies': int(counts_df['count'].sum()),
                'avg_anomaly_rate': float(counts_df['rate'].mean()),
                'max_anomaly_period': counts_df.loc[counts_df['count'].idxmax()]['period'],
                'trend': 'increasing' if len(counts_df) > 1 and counts_df['count'].iloc[-1] > counts_df['count'].iloc[0] else 'decreasing',
                'period_details': anomaly_counts
            }
            
            return trend_analysis
        
        return {'status': 'no_data'}
    
    def get_anomaly_report(self) -> Dict:
        """
        Generate comprehensive anomaly detection report
        
        Returns:
            Dictionary containing full report
        """
        return {
            'model_name': self.model_name,
            'models_used': list(self.models.keys()),
            'total_anomalies': len(self.anomalies),
            'anomalies': self.anomalies,
            'anomaly_scores': self.anomaly_scores,
            'timestamp': pd.Timestamp.now().isoformat()
        }
