"""
Configuration settings for the wildfire watershed clustering project.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Google Earth Engine Configuration
GEE_CONFIG = {
    "project_id": "ee-jsuhydrolabenb",  # Default GEE project ID
    "service_account_key": None,  # Path to service account key if using service account
    "max_pixels": 1e9,  # Maximum pixels for GEE exports
    "scale": 1000,  # Default scale in meters for raster operations
    "crs": "EPSG:4326",  # Default coordinate reference system
}

# Dataset Configuration
DATASETS = {
    "huc12": {
        "asset_id": "USGS/WBD/2017/HUC12",
        "date": "2017-04-22",
        "type": "FeatureCollection"
    },
    "firms": {
        "asset_id": "FIRMS", 
        "start_date": "2000-11-01",
        "end_date": "2025-06-03",
        "type": "ImageCollection"
    }
}

# Study Area Configuration - Western United States
STUDY_AREA = {
    "name": "Western_United_States",
    "bounds": {
        "west": -125.0,    # Western boundary (Pacific Coast)
        "east": -102.0,    # Eastern boundary (approximately Colorado/New Mexico eastern border)
        "south": 31.0,     # Southern boundary (Mexico border)
        "north": 49.0      # Northern boundary (Canada border)
    },
    "states": [
        "Washington", "Oregon", "California", "Idaho", "Nevada", "Utah", 
        "Arizona", "Montana", "Wyoming", "Colorado", "New Mexico"
    ]
}

# Fire Data Processing Configuration
FIRE_CONFIG = {
    "confidence_threshold": 80,  # Minimum fire confidence percentage
    "spatial_threshold_degrees": 0.01,  # Spatial clustering threshold (degrees)
    "temporal_threshold_days": 5,  # Temporal clustering threshold (days)
    "fire_end_threshold_days": 16,  # Days without activity to consider fire ended
    "min_fire_size_ha": 1.0,  # Minimum fire size in hectares
}

# Fire Characteristics Configuration
FIRE_METRICS = {
    "temporal_metrics": [
        "fire_frequency",           # Number of fires per time period
        "mean_fire_return_interval", # Mean time between fires
        "median_fire_return_interval", # Median time between fires  
        "fire_return_interval_std", # Standard deviation of FRI
        "max_fire_free_period",     # Longest period without fire
        "time_since_last_fire",     # Years since most recent fire
        "fire_season_peak",         # Dominant fire month/season
        "fire_season_length",       # Length of fire season
    ],
    "spatial_metrics": [
        "total_burned_fraction",    # Fraction of watershed ever burned
        "mean_fire_size",          # Average fire size in watershed
        "median_fire_size",        # Median fire size
        "fire_size_cv",            # Coefficient of variation of fire sizes
        "largest_fire_fraction",   # Fraction burned by largest fire
        "repeat_burn_fraction",    # Fraction burned multiple times
        "burn_severity_proxy",     # Based on FRP or vegetation recovery
    ],
    "intensity_metrics": [
        "mean_frp",                # Mean Fire Radiative Power
        "max_frp",                 # Maximum FRP observed
        "fire_duration_mean",      # Average fire duration
        "fire_duration_max",       # Maximum fire duration
        "peak_fire_month",         # Month with most fire activity
    ]
}

# Clustering Configuration
CLUSTERING_CONFIG = {
    "algorithms": {
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean"
        },
        "gmm": {
            "n_components": 8,
            "covariance_type": "full",
            "random_state": 42
        },
        "hierarchical": {
            "n_clusters": 8,
            "linkage": "ward",
            "affinity": "euclidean"
        },
        "kmeans": {
            "n_clusters": 8,
            "random_state": 42,
            "n_init": 10
        }
    },
    "feature_scaling": "StandardScaler",  # StandardScaler, MinMaxScaler, RobustScaler
    "n_clusters_range": range(3, 15),     # Range for optimal cluster selection
    "validation_metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"]
}

# Parallel Processing Configuration
PROCESSING_CONFIG = {
    "n_jobs": min(96, os.cpu_count()),  # Use all available cores (max 96 for your system)
    "chunk_size": 1000,                 # Chunk size for parallel processing
    "memory_limit": "90GB",             # Memory limit for dask
    "use_gpu": False,                   # Set to True if implementing GPU acceleration
    "batch_size": 100,                  # Batch size for GEE requests
}

# Visualization Configuration
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "color_palette": "viridis",
    "map_tiles": "OpenStreetMap",
    "export_formats": ["png", "pdf", "svg"]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "wildfire_clustering.log"
}

# Create logs directory
LOGGING_CONFIG["file"].parent.mkdir(parents=True, exist_ok=True)

def get_study_area_bounds() -> Dict[str, float]:
    """Get study area bounding box."""
    return STUDY_AREA["bounds"]

def get_fire_confidence_threshold() -> int:
    """Get minimum fire confidence threshold."""
    return FIRE_CONFIG["confidence_threshold"]

def get_gee_scale() -> int:
    """Get default Google Earth Engine scale."""
    return GEE_CONFIG["scale"]

def update_config(section: str, **kwargs) -> None:
    """Update configuration parameters dynamically."""
    if section == "fire":
        FIRE_CONFIG.update(kwargs)
    elif section == "clustering":
        CLUSTERING_CONFIG.update(kwargs)
    elif section == "processing":
        PROCESSING_CONFIG.update(kwargs)
    else:
        raise ValueError(f"Unknown configuration section: {section}")