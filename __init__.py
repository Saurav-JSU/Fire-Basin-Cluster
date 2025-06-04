# config/__init__.py
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

# src/data/__init__.py
"""Data loading and preprocessing modules."""

from .gee_loader import GEEDataLoader
from .preprocessor import FIRMSPreprocessor

__all__ = ['GEEDataLoader', 'FIRMSPreprocessor']

# src/features/__init__.py
"""Feature engineering and fire metrics calculation modules."""

from .fire_metrics import WatershedFireMetrics
from .temporal_analysis import TemporalFireAnalyzer

__all__ = ['WatershedFireMetrics', 'TemporalFireAnalyzer']

# src/clustering/__init__.py
"""Clustering algorithms and validation modules."""

__all__ = []

# src/visualization/__init__.py
"""Visualization and mapping modules."""

__all__ = []

# tests/__init__.py
"""Test suite for wildfire watershed clustering project."""

__all__ = []