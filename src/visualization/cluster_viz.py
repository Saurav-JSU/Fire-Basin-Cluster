"""
Visualization Tools for Watershed Fire Regime Clustering

This module provides visualization tools for analyzing and presenting clustering results,
including:
1. Feature importance plots
2. Cluster distribution plots
3. Cluster characteristic plots
4. Geographic visualization of clusters
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
import geopandas as gpd
from sklearn.decomposition import PCA
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterVisualizer:
    """
    Visualization tools for watershed fire regime clustering.
    
    This class provides methods for:
    1. Plotting feature importance
    2. Visualizing cluster distributions
    3. Analyzing cluster characteristics
    4. Creating geographic visualizations
    """
    
    def __init__(self, 
                 feature_importance: Optional[pd.Series] = None,
                 cluster_labels: Optional[np.ndarray] = None,
                 watersheds: Optional[gpd.GeoDataFrame] = None):
        """
        Initialize visualizer.
        
        Args:
            feature_importance: Series of feature importance scores
            cluster_labels: Array of cluster labels
            watersheds: GeoDataFrame of watershed boundaries
        """
        self.feature_importance = feature_importance
        self.cluster_labels = cluster_labels
        self.watersheds = watersheds
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        logger.info("Cluster Visualizer initialized")
    
    def plot_feature_importance(self, 
                              top_n: Optional[int] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Args:
            top_n: Number of top features to plot (None for all)
            figsize: Figure size
            
        Returns:
            plt.Figure: Feature importance plot
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance data available")
        
        # Get top features
        if top_n is not None:
            importance = self.feature_importance.nlargest(top_n)
        else:
            importance = self.feature_importance
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        importance.plot(kind='barh', ax=ax)
        
        # Customize plot
        ax.set_title('Feature Importance for Clustering')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        
        # Rotate y-axis labels for better readability
        plt.xticks(rotation=45)
        
        return fig
    
    def plot_cluster_distribution(self,
                                data: pd.DataFrame,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot distribution of features by cluster.
        
        Args:
            data: Feature data with cluster labels
            figsize: Figure size
            
        Returns:
            plt.Figure: Cluster distribution plot
        """
        if self.cluster_labels is None:
            raise ValueError("No cluster labels available")
        
        # Add cluster labels to data
        plot_data = data.copy()
        plot_data['cluster'] = self.cluster_labels
        
        # Create figure with subplots
        n_features = len(data.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(data.columns):
            sns.boxplot(data=plot_data, x='cluster', y=feature, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Value')
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_characteristics(self,
                                   cluster_stats: pd.DataFrame,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot characteristics of each cluster.
        
        Args:
            cluster_stats: DataFrame of cluster statistics
            figsize: Figure size
            
        Returns:
            plt.Figure: Cluster characteristics plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mean values for each feature by cluster
        cluster_stats.xs('mean', level=1, axis=1).plot(kind='bar', ax=ax)
        
        # Customize plot
        ax.set_title('Cluster Characteristics')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Feature Value')
        plt.xticks(rotation=45)
        
        # Add legend
        ax.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_geographic_clusters(self,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot geographic distribution of clusters.
        
        Args:
            figsize: Figure size
            
        Returns:
            plt.Figure: Geographic cluster plot
        """
        if self.watersheds is None or self.cluster_labels is None:
            raise ValueError("Both watersheds and cluster labels are required")
        
        # Add cluster labels to watersheds
        plot_data = self.watersheds.copy()
        plot_data['cluster'] = self.cluster_labels
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot watersheds colored by cluster
        plot_data.plot(column='cluster', 
                      categorical=True,
                      legend=True,
                      ax=ax)
        
        # Customize plot
        ax.set_title('Geographic Distribution of Fire Regime Clusters')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        return fig
    
    def plot_pca_visualization(self,
                             data: pd.DataFrame,
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create PCA visualization of clusters.
        
        Args:
            data: Feature data
            figsize: Figure size
            
        Returns:
            plt.Figure: PCA visualization plot
        """
        if self.cluster_labels is None:
            raise ValueError("No cluster labels available")
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points colored by cluster
        scatter = ax.scatter(pca_data[:, 0], 
                           pca_data[:, 1],
                           c=self.cluster_labels,
                           cmap='viridis')
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(),
                          title="Clusters")
        ax.add_artist(legend1)
        
        # Customize plot
        ax.set_title('PCA Visualization of Clusters')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        return fig 