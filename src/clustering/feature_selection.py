"""
Feature Selection for Watershed Fire Regime Clustering

This module handles feature selection and preprocessing for clustering analysis,
including:
1. Feature importance analysis
2. Correlation analysis
3. Dimensionality reduction
4. Feature scaling
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Feature selection and preprocessing for watershed fire regime clustering.
    
    This class handles:
    1. Feature importance analysis
    2. Correlation analysis
    3. Dimensionality reduction
    4. Feature scaling
    """
    
    def __init__(self, 
                 n_components: Optional[int] = None,
                 correlation_threshold: float = 0.7,
                 use_robust_scaling: bool = True):
        """
        Initialize feature selector.
        
        Args:
            n_components: Number of PCA components to use (None for auto)
            correlation_threshold: Threshold for removing correlated features
            use_robust_scaling: Whether to use robust scaling (True) or standard scaling (False)
        """
        self.n_components = n_components
        self.correlation_threshold = correlation_threshold
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.imputer = SimpleImputer(strategy='mean')
        
        # Cache for processed data
        self.feature_importance = None
        self.correlation_matrix = None
        self.selected_features = None
        self.label_encoders = {}
        
        logger.info("Feature Selector initialized")
        logger.info(f"  - Correlation threshold: {correlation_threshold}")
        logger.info(f"  - Scaling method: {'Robust' if use_robust_scaling else 'Standard'}")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data by handling categorical variables and missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Handle categorical variables
        for col in df.columns:
            if df[col].dtype == 'object':
                # Create and fit label encoder
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded categorical variable: {col}")
        
        # Handle missing values
        if df_processed.isna().any().any():
            logger.info("Handling missing values")
            # Get columns with missing values
            cols_with_na = df_processed.columns[df_processed.isna().any()].tolist()
            logger.info(f"Columns with missing values: {cols_with_na}")
            
            # Impute missing values
            df_processed = pd.DataFrame(
                self.imputer.fit_transform(df_processed),
                columns=df_processed.columns,
                index=df_processed.index
            )
        
        return df_processed
    
    def analyze_features(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze features for clustering suitability.
        
        Args:
            metrics_df: DataFrame with fire metrics
            
        Returns:
            Dict: Feature analysis results
        """
        logger.info("Analyzing features for clustering")
        
        # Preprocess data
        processed_df = self._preprocess_data(metrics_df)
        
        # Calculate feature importance
        importance = self._calculate_feature_importance(processed_df)
        
        # Calculate correlations
        correlations = self._calculate_correlations(processed_df)
        
        # Identify highly correlated features
        correlated_features = self._identify_correlated_features(correlations)
        
        # Store results
        self.feature_importance = importance
        self.correlation_matrix = correlations
        
        return {
            'feature_importance': importance,
            'correlations': correlations,
            'correlated_features': correlated_features
        }
    
    def select_features(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and preprocess features for clustering.
        
        Args:
            metrics_df: DataFrame with fire metrics
            
        Returns:
            pd.DataFrame: Selected and preprocessed features
        """
        logger.info("Selecting features for clustering")
        
        # Preprocess data
        processed_df = self._preprocess_data(metrics_df)
        
        # Remove highly correlated features
        df_uncorrelated = self._remove_correlated_features(processed_df)
        
        # Scale features
        df_scaled = self._scale_features(df_uncorrelated)
        
        # Apply PCA if specified
        if self.n_components is not None:
            df_pca = self._apply_pca(df_scaled)
            self.selected_features = df_pca
            return df_pca
        
        self.selected_features = df_scaled
        return df_scaled
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate feature importance using F-scores."""
        # Use a dummy target for feature selection
        dummy_target = np.zeros(len(df))
        
        # Calculate F-scores
        selector = SelectKBest(f_classif, k='all')
        selector.fit(df, dummy_target)
        
        # Create importance series
        importance = pd.Series(
            selector.scores_,
            index=df.columns
        ).sort_values(ascending=False)
        
        return importance
    
    def _calculate_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix."""
        return df.corr()
    
    def _identify_correlated_features(self, corr_matrix: pd.DataFrame) -> List[Tuple[str, str]]:
        """Identify pairs of highly correlated features."""
        correlated_pairs = []
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find highly correlated pairs
        for col in upper.columns:
            high_corr = upper[col][abs(upper[col]) > self.correlation_threshold]
            for idx in high_corr.index:
                correlated_pairs.append((col, idx))
        
        return correlated_pairs
    
    def _remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        if self.correlation_matrix is None:
            self.correlation_matrix = self._calculate_correlations(df)
        
        # Get correlated pairs
        correlated_pairs = self._identify_correlated_features(self.correlation_matrix)
        
        # Remove one feature from each correlated pair
        features_to_remove = set()
        for feat1, feat2 in correlated_pairs:
            # Keep the feature with higher importance
            if self.feature_importance is not None:
                if self.feature_importance[feat1] < self.feature_importance[feat2]:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            else:
                # If no importance scores, remove the second feature
                features_to_remove.add(feat2)
        
        # Remove features
        df_uncorrelated = df.drop(columns=list(features_to_remove))
        
        logger.info(f"Removed {len(features_to_remove)} correlated features")
        return df_uncorrelated
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the selected scaler."""
        # Scale features
        scaled_data = self.scaler.fit_transform(df)
        
        # Create DataFrame with scaled data
        df_scaled = pd.DataFrame(
            scaled_data,
            columns=df.columns,
            index=df.index
        )
        
        return df_scaled
    
    def _apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA to reduce dimensionality."""
        # Fit PCA
        pca_data = self.pca.fit_transform(df)
        
        # Create DataFrame with PCA components
        df_pca = pd.DataFrame(
            pca_data,
            columns=[f'PC{i+1}' for i in range(pca_data.shape[1])],
            index=df.index
        )
        
        # Log explained variance
        explained_variance = self.pca.explained_variance_ratio_
        logger.info("PCA explained variance:")
        for i, var in enumerate(explained_variance):
            logger.info(f"  PC{i+1}: {var:.3f}")
        
        return df_pca
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection process."""
        if self.feature_importance is None or self.correlation_matrix is None:
            return {
                'status': 'No feature analysis performed yet',
                'selected_features': list(self.selected_features.columns) if self.selected_features is not None else None
            }
        
        return {
            'feature_importance': self.feature_importance.to_dict(),
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'selected_features': list(self.selected_features.columns) if self.selected_features is not None else None,
            'n_original_features': len(self.correlation_matrix),
            'n_selected_features': len(self.selected_features.columns) if self.selected_features is not None else None
        } 