"""Configuration module for wildfire watershed clustering."""

from .settings import (
    GEE_CONFIG, DATASETS, STUDY_AREA, FIRE_CONFIG, 
    FIRE_METRICS, CLUSTERING_CONFIG, PROCESSING_CONFIG,
    get_study_area_bounds, get_fire_confidence_threshold, get_gee_scale
)

__all__ = [
    'GEE_CONFIG', 'DATASETS', 'STUDY_AREA', 'FIRE_CONFIG',
    'FIRE_METRICS', 'CLUSTERING_CONFIG', 'PROCESSING_CONFIG',
    'get_study_area_bounds', 'get_fire_confidence_threshold', 'get_gee_scale'
]

# src/__init__.py
"""Main source code package for wildfire watershed clustering."""

__version__ = "0.1.0"
__author__ = "Wildfire Watershed Clustering Team"
__description__ = "Clustering HUC12 watersheds based on wildfire characteristics"