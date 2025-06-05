"""
Main script for running watershed fire regime clustering analysis.

This script:
1. Loads and preprocesses fire metrics data
2. Performs feature selection
3. Runs clustering analysis
4. Generates visualizations
5. Saves results
"""
import os
import sys
import logging
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.clustering.feature_selection import FeatureSelector
from src.clustering.clustering import WatershedClusterer
from src.visualization.cluster_viz import ClusterVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(metrics_file: str, watersheds_file: str) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Load fire metrics and watershed data.
    
    Args:
        metrics_file: Path to fire metrics CSV file
        watersheds_file: Path to watershed boundaries GeoJSON file
        
    Returns:
        Tuple of (metrics DataFrame, watersheds GeoDataFrame)
    """
    logger.info("Loading data")
    
    # Load fire metrics
    metrics_df = pd.read_csv(metrics_file, index_col=0)
    logger.info(f"Loaded {len(metrics_df)} watershed metrics")
    
    # Load watershed boundaries
    watersheds = gpd.read_file(watersheds_file)
    logger.info(f"Loaded {len(watersheds)} watershed boundaries")
    
    return metrics_df, watersheds

def run_clustering_analysis(metrics_df: pd.DataFrame,
                          watersheds: gpd.GeoDataFrame,
                          output_dir: str,
                          n_components: Optional[int] = None,
                          correlation_threshold: float = 0.7,
                          algorithm: str = 'kmeans',
                          n_clusters: Optional[int] = None) -> Dict:
    """
    Run complete clustering analysis.
    
    Args:
        metrics_df: DataFrame of fire metrics
        watersheds: GeoDataFrame of watershed boundaries
        output_dir: Directory to save results
        n_components: Number of PCA components to use
        correlation_threshold: Threshold for removing correlated features
        algorithm: Clustering algorithm to use
        n_clusters: Number of clusters to use
        
    Returns:
        Dict: Analysis results and metadata
    """
    logger.info("Starting clustering analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature selector
    feature_selector = FeatureSelector(
        n_components=n_components,
        correlation_threshold=correlation_threshold
    )
    
    # Analyze features
    feature_analysis = feature_selector.analyze_features(metrics_df)
    logger.info("Feature analysis complete")
    
    # Select features
    selected_features = feature_selector.select_features(metrics_df)
    logger.info(f"Selected {len(selected_features.columns)} features")
    
    # Initialize clusterer
    clusterer = WatershedClusterer(
        algorithm=algorithm,
        n_clusters=n_clusters
    )
    
    # Run clustering
    clusterer.fit(selected_features)
    logger.info("Clustering complete")
    
    # Get cluster characteristics
    cluster_stats = clusterer.get_cluster_characteristics(metrics_df)
    
    # Initialize visualizer
    visualizer = ClusterVisualizer(
        feature_importance=feature_analysis['feature_importance'],
        cluster_labels=clusterer.cluster_labels,
        watersheds=watersheds
    )
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Feature importance plot
    importance_fig = visualizer.plot_feature_importance()
    importance_fig.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    # Cluster distribution plot
    dist_fig = visualizer.plot_cluster_distribution(selected_features)
    dist_fig.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
    
    # Cluster characteristics plot
    char_fig = visualizer.plot_cluster_characteristics(cluster_stats)
    char_fig.savefig(os.path.join(output_dir, 'cluster_characteristics.png'))
    
    # Geographic cluster plot
    geo_fig = visualizer.plot_geographic_clusters()
    geo_fig.savefig(os.path.join(output_dir, 'geographic_clusters.png'))
    
    # PCA visualization
    pca_fig = visualizer.plot_pca_visualization(selected_features)
    pca_fig.savefig(os.path.join(output_dir, 'pca_visualization.png'))
    
    # Save results
    logger.info("Saving results")
    
    # Add cluster labels to watersheds
    watersheds['cluster'] = clusterer.cluster_labels
    watersheds.to_file(os.path.join(output_dir, 'clustered_watersheds.geojson'))
    
    # Save cluster statistics
    cluster_stats.to_csv(os.path.join(output_dir, 'cluster_statistics.csv'))
    
    # Save analysis metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_watersheds': len(metrics_df),
        'n_features': len(metrics_df.columns),
        'n_selected_features': len(selected_features.columns),
        'feature_analysis': feature_analysis,
        'cluster_summary': clusterer.get_cluster_summary()
    }
    
    with open(os.path.join(output_dir, 'analysis_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main():
    """Main function to run clustering analysis."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run watershed fire regime clustering analysis')
    parser.add_argument('--metrics', required=True, help='Path to fire metrics CSV file')
    parser.add_argument('--watersheds', required=True, help='Path to watershed boundaries GeoJSON file')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--n-components', type=int, help='Number of PCA components to use')
    parser.add_argument('--correlation-threshold', type=float, default=0.7,
                       help='Threshold for removing correlated features')
    parser.add_argument('--algorithm', default='kmeans',
                       choices=['kmeans', 'dbscan'],
                       help='Clustering algorithm to use')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters to use')
    
    args = parser.parse_args()
    
    # Load data
    metrics_df, watersheds = load_data(args.metrics, args.watersheds)
    
    # Run analysis
    metadata = run_clustering_analysis(
        metrics_df=metrics_df,
        watersheds=watersheds,
        output_dir=args.output,
        n_components=args.n_components,
        correlation_threshold=args.correlation_threshold,
        algorithm=args.algorithm,
        n_clusters=args.n_clusters
    )
    
    logger.info("Analysis complete")
    logger.info(f"Results saved to {args.output}")

if __name__ == '__main__':
    main() 