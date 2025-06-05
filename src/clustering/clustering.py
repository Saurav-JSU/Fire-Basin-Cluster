"""
Clustering Analysis for Watershed Fire Regimes

This module implements clustering algorithms for analyzing watershed fire regimes,
including:
1. K-means clustering
2. Hierarchical clustering
3. DBSCAN clustering
4. Cluster validation and evaluation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatershedClusterer:
    """
    Clustering analysis for watershed fire regimes.
    
    This class handles:
    1. Multiple clustering algorithms
    2. Cluster validation
    3. Cluster analysis
    """
    
    def __init__(self, 
                 algorithm: str = 'kmeans',
                 n_clusters: Optional[int] = None,
                 max_clusters: int = 10,
                 random_state: int = 42):
        """
        Initialize clusterer.
        
        Args:
            algorithm: Clustering algorithm to use ('kmeans', 'hierarchical', or 'dbscan')
            n_clusters: Number of clusters to use (None for auto-determination)
            max_clusters: Maximum number of clusters to consider for auto-determination
            random_state: Random seed for reproducibility
        """
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        
        # Initialize clusterer
        self._initialize_clusterer()
        
        # Cache for results
        self.cluster_labels = None
        self.cluster_metrics = None
        
        logger.info(f"Clusterer initialized with {algorithm} algorithm")
    
    def _initialize_clusterer(self):
        """Initialize the clustering algorithm."""
        if self.algorithm == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.algorithm == 'hierarchical':
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters
            )
        elif self.algorithm == 'dbscan':
            self.clusterer = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def find_optimal_clusters(self, data: pd.DataFrame) -> int:
        """
        Find optimal number of clusters using multiple metrics.
        
        Args:
            data: Input data for clustering
            
        Returns:
            int: Optimal number of clusters
        """
        if len(data) <= 2:
            logger.warning("Not enough samples for cluster optimization. Using 1 cluster.")
            return 1
            
        # Limit max_clusters based on sample size
        max_k = min(self.max_clusters, len(data) - 1)
        if max_k < 2:
            logger.warning("Not enough samples for multiple clusters. Using 1 cluster.")
            return 1
            
        logger.info(f"Finding optimal number of clusters (max: {max_k})")
        
        # Calculate metrics for different numbers of clusters
        metrics = []
        for k in range(2, max_k + 1):
            # Fit clustering
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(data)
            
            # Calculate metrics
            silhouette = silhouette_score(data, labels)
            calinski = calinski_harabasz_score(data, labels)
            davies = davies_bouldin_score(data, labels)
            
            metrics.append({
                'n_clusters': k,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies
            })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Find optimal k based on silhouette score
        optimal_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'n_clusters']
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def fit(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit clustering to data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            np.ndarray: Cluster labels
        """
        # Determine number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_clusters(data)
        
        # Adjust n_clusters if necessary
        if self.n_clusters > len(data):
            logger.warning(f"Reducing number of clusters from {self.n_clusters} to {len(data)} due to sample size")
            self.n_clusters = len(data)
            self._initialize_clusterer()
        
        logger.info(f"Fitting {self.algorithm} clustering")
        
        # Fit clustering
        self.cluster_labels = self.clusterer.fit_predict(data)
        
        # Calculate cluster metrics
        self.cluster_metrics = self._calculate_cluster_metrics(data)
        
        return self.cluster_labels
    
    def _calculate_cluster_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        if len(np.unique(self.cluster_labels)) < 2:
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0
            }
            
        return {
            'silhouette_score': silhouette_score(data, self.cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(data, self.cluster_labels),
            'davies_bouldin_score': davies_bouldin_score(data, self.cluster_labels)
        }
    
    def get_cluster_characteristics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get characteristics of each cluster.
        
        Args:
            data: Input data used for clustering
            
        Returns:
            pd.DataFrame: Cluster characteristics
        """
        if self.cluster_labels is None:
            raise ValueError("Must fit clustering before getting characteristics")
        
        # Add cluster labels to data
        df = data.copy()
        df['cluster'] = self.cluster_labels
        
        # Calculate cluster characteristics
        characteristics = []
        for cluster in np.unique(self.cluster_labels):
            cluster_data = df[df['cluster'] == cluster]
            
            # Calculate statistics for each feature
            stats = {}
            for col in data.columns:
                stats[f'{col}_mean'] = cluster_data[col].mean()
                stats[f'{col}_std'] = cluster_data[col].std()
            
            # Add cluster info
            stats['cluster'] = cluster
            stats['size'] = len(cluster_data)
            
            characteristics.append(stats)
        
        return pd.DataFrame(characteristics)
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of clustering results."""
        if self.cluster_labels is None:
            return {
                'status': 'No clustering performed yet',
                'algorithm': self.algorithm,
                'n_clusters': self.n_clusters
            }
        
        return {
            'algorithm': self.algorithm,
            'n_clusters': self.n_clusters,
            'cluster_sizes': pd.Series(self.cluster_labels).value_counts().to_dict(),
            'metrics': self.cluster_metrics
        } 