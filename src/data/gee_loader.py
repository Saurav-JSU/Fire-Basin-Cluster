"""
Google Earth Engine data loader for HUC12 watersheds and FIRMS fire data.
"""
import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys

# Robust import of config settings
try:
    from config.settings import (
        GEE_CONFIG, DATASETS, STUDY_AREA, FIRE_CONFIG, 
        RAW_DATA_DIR, get_study_area_bounds
    )
except ImportError:
    # Alternative import method if config module not in path
    try:
        import importlib.util
        config_file = Path(__file__).parent.parent.parent / "config" / "settings.py"
        if config_file.exists():
            spec = importlib.util.spec_from_file_location("config.settings", config_file)
            settings_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(settings_module)
            
            GEE_CONFIG = settings_module.GEE_CONFIG
            DATASETS = settings_module.DATASETS
            STUDY_AREA = settings_module.STUDY_AREA
            FIRE_CONFIG = settings_module.FIRE_CONFIG
            RAW_DATA_DIR = settings_module.RAW_DATA_DIR
            get_study_area_bounds = settings_module.get_study_area_bounds
        else:
            raise ImportError("Config file not found")
    except Exception as e:
        # Fallback configuration if all imports fail
        print(f"Warning: Could not import config settings ({e}). Using fallback configuration.")
        
        # Fallback configuration
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        GEE_CONFIG = {
            "project_id": "jsuhydrolabenb",
            "service_account_key": None,
            "max_pixels": 1e9,
            "scale": 1000,
            "crs": "EPSG:4326",
        }
        
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
        
        STUDY_AREA = {
            "name": "Western_United_States",
            "bounds": {
                "west": -125.0,
                "east": -102.0,
                "south": 31.0,
                "north": 49.0
            }
        }
        
        FIRE_CONFIG = {
            "confidence_threshold": 80,
            "spatial_threshold_degrees": 0.01,
            "temporal_threshold_days": 5,
            "fire_end_threshold_days": 16,
            "min_fire_size_ha": 1.0,
        }
        
        def get_study_area_bounds():
            return STUDY_AREA["bounds"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GEEDataLoader:
    """
    Google Earth Engine data loader for wildfire watershed clustering project.
    
    Handles authentication, data loading, and initial preprocessing for:
    - HUC12 watershed boundaries 
    - FIRMS fire detection data
    """
    
    def __init__(self, project_id: Optional[str] = None, 
                 service_account_key: Optional[str] = None):
        """
        Initialize GEE data loader.
        
        Args:
            project_id: Google Cloud project ID for Earth Engine
            service_account_key: Path to service account key file (optional)
        """
        self.project_id = project_id or GEE_CONFIG.get("project_id")
        self.service_account_key = service_account_key or GEE_CONFIG.get("service_account_key")
        self.authenticated = False
        
        # Study area geometry
        self.study_area_bounds = get_study_area_bounds()
        self.study_area_geometry = None
        
        # Data cache
        self.huc12_data = None
        self.firms_data = None
        
    def authenticate(self) -> bool:
        """
        Authenticate with Google Earth Engine.
        
        Returns:
            bool: True if authentication successful
        """
        try:
            if self.service_account_key:
                # Service account authentication
                logger.info("Authenticating with service account...")
                credentials = ee.ServiceAccountCredentials(
                    None, self.service_account_key
                )
                ee.Initialize(credentials, project=self.project_id)
            else:
                # Interactive authentication
                logger.info("Authenticating interactively...")
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
            
            self.authenticated = True
            logger.info("Google Earth Engine authentication successful!")
            return True
            
        except Exception as e:
            logger.error(f"GEE authentication failed: {e}")
            self.authenticated = False
            return False
    
    def _create_study_area_geometry(self) -> ee.Geometry:
        """
        Create study area geometry from bounding box.
        
        Returns:
            ee.Geometry: Study area polygon
        """
        if self.study_area_geometry is None:
            bounds = self.study_area_bounds
            coords = [
                [bounds["west"], bounds["south"]],
                [bounds["west"], bounds["north"]],
                [bounds["east"], bounds["north"]], 
                [bounds["east"], bounds["south"]],
                [bounds["west"], bounds["south"]]
            ]
            self.study_area_geometry = ee.Geometry.Polygon([coords])
            
        return self.study_area_geometry
    
    def load_huc12_watersheds(self, save_local: bool = True) -> ee.FeatureCollection:
        """
        Load HUC12 watershed boundaries for the study area.
        
        Args:
            save_local: Whether to save data locally for faster access
            
        Returns:
            ee.FeatureCollection: HUC12 watersheds in study area
        """
        if not self.authenticated:
            raise RuntimeError("Must authenticate with GEE first")
        
        logger.info("Loading HUC12 watershed boundaries...")
        
        try:
            # Load HUC12 dataset
            huc12 = ee.FeatureCollection(DATASETS["huc12"]["asset_id"])
            
            # Filter to study area
            study_area = self._create_study_area_geometry()
            huc12_filtered = huc12.filterBounds(study_area)
            
            # Get watershed count
            count = huc12_filtered.size()
            logger.info(f"Found {count.getInfo()} HUC12 watersheds in study area")
            
            # Cache the data
            self.huc12_data = huc12_filtered
            
            # Save locally if requested
            if save_local:
                self._save_huc12_local(huc12_filtered)
            
            return huc12_filtered
            
        except Exception as e:
            logger.error(f"Error loading HUC12 data: {e}")
            raise
    
    def load_firms_data(self, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       confidence_threshold: Optional[int] = None) -> ee.ImageCollection:
        """
        Load FIRMS fire data for the study area and time period.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format) 
            confidence_threshold: Minimum fire confidence (default from config)
            
        Returns:
            ee.ImageCollection: Filtered FIRMS fire data
        """
        if not self.authenticated:
            raise RuntimeError("Must authenticate with GEE first")
        
        # Set defaults from config
        start_date = start_date or DATASETS["firms"]["start_date"]
        end_date = end_date or DATASETS["firms"]["end_date"]
        confidence_threshold = confidence_threshold or FIRE_CONFIG["confidence_threshold"]
        
        logger.info(f"Loading FIRMS data from {start_date} to {end_date}...")
        logger.info(f"Applying confidence threshold: >{confidence_threshold}%")
        
        try:
            # Load FIRMS dataset
            firms = ee.ImageCollection(DATASETS["firms"]["asset_id"])
            
            # Filter by date range
            firms_filtered = firms.filterDate(start_date, end_date)
            
            # Filter by study area
            study_area = self._create_study_area_geometry()
            firms_filtered = firms_filtered.filterBounds(study_area)
            
            # Apply confidence filter
            firms_filtered = firms_filtered.filter(
                ee.Filter.gt('confidence', confidence_threshold)
            )
            
            # Get image count
            count = firms_filtered.size()
            logger.info(f"Found {count.getInfo()} FIRMS images meeting criteria")
            
            # Cache the data
            self.firms_data = firms_filtered
            
            return firms_filtered
            
        except Exception as e:
            logger.error(f"Error loading FIRMS data: {e}")
            raise
    
    def _save_huc12_local(self, huc12_fc: ee.FeatureCollection) -> None:
        """
        Save HUC12 data locally as GeoJSON.
        
        Args:
            huc12_fc: HUC12 FeatureCollection to save
        """
        try:
            output_path = RAW_DATA_DIR / "huc12_western_us.geojson"
            logger.info(f"Saving HUC12 data to {output_path}...")
            
            # Convert to GeoDataFrame and save
            gdf = geemap.ee_to_gdf(huc12_fc)
            gdf.to_file(output_path, driver="GeoJSON")
            
            logger.info(f"Saved {len(gdf)} HUC12 watersheds locally")
            
        except Exception as e:
            logger.error(f"Error saving HUC12 data locally: {e}")
    
    def get_watershed_fire_data(self, huc12_id: str, 
                              buffer_meters: int = 0) -> Dict:
        """
        Get fire data for a specific HUC12 watershed.
        
        Args:
            huc12_id: HUC12 identifier
            buffer_meters: Buffer around watershed boundary
            
        Returns:
            Dict: Fire data summary for the watershed
        """
        if not self.authenticated or self.huc12_data is None or self.firms_data is None:
            raise RuntimeError("Must load HUC12 and FIRMS data first")
        
        try:
            # Get specific watershed
            watershed = self.huc12_data.filter(ee.Filter.eq('huc12', huc12_id))
            
            if buffer_meters > 0:
                watershed = watershed.geometry().buffer(buffer_meters)
            else:
                watershed = watershed.geometry()
            
            # Filter FIRMS data to watershed
            watershed_fires = self.firms_data.filterBounds(watershed)
            
            # Get basic statistics
            fire_count = watershed_fires.size()
            
            return {
                "huc12_id": huc12_id,
                "fire_count": fire_count.getInfo(),
                "watershed_geometry": watershed,
                "fires": watershed_fires
            }
            
        except Exception as e:
            logger.error(f"Error getting watershed fire data for {huc12_id}: {e}")
            raise
    
    def export_watershed_sample(self, n_watersheds: int = 10, 
                              output_path: Optional[Path] = None) -> gpd.GeoDataFrame:
        """
        Export a sample of watersheds with basic fire statistics for testing.
        
        Args:
            n_watersheds: Number of watersheds to sample
            output_path: Path to save sample data
            
        Returns:
            gpd.GeoDataFrame: Sample watershed data with fire statistics
        """
        if not self.authenticated or self.huc12_data is None:
            raise RuntimeError("Must load HUC12 data first")
        
        logger.info(f"Creating sample of {n_watersheds} watersheds...")
        
        try:
            # Get sample of watersheds
            sample_watersheds = self.huc12_data.limit(n_watersheds)
            
            # Convert to GeoDataFrame
            sample_gdf = geemap.ee_to_gdf(sample_watersheds)
            
            # Add basic fire statistics (simplified for testing)
            if self.firms_data is not None:
                sample_gdf['fire_count_estimate'] = 0  # Placeholder - will implement properly later
            
            # Save if path provided
            if output_path:
                output_path = Path(output_path)
                sample_gdf.to_file(output_path, driver="GeoJSON")
                logger.info(f"Saved sample to {output_path}")
            
            return sample_gdf
            
        except Exception as e:
            logger.error(f"Error creating watershed sample: {e}")
            raise
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about loaded datasets.
        
        Returns:
            Dict: Dataset information summary
        """
        info = {
            "authenticated": self.authenticated,
            "study_area": self.study_area_bounds,
            "datasets": {
                "huc12": {
                    "loaded": self.huc12_data is not None,
                    "count": None
                },
                "firms": {
                    "loaded": self.firms_data is not None,
                    "count": None,
                    "date_range": DATASETS["firms"]
                }
            }
        }
        
        try:
            if self.huc12_data is not None:
                info["datasets"]["huc12"]["count"] = self.huc12_data.size().getInfo()
            
            if self.firms_data is not None:
                info["datasets"]["firms"]["count"] = self.firms_data.size().getInfo()
                
        except Exception as e:
            logger.warning(f"Could not get dataset counts: {e}")
        
        return info

def main():
    """
    Example usage of GEE data loader.
    """
    # Initialize loader
    loader = GEEDataLoader()
    
    # Authenticate
    if not loader.authenticate():
        logger.error("Authentication failed!")
        return
    
    # Load data
    try:
        # Load HUC12 watersheds
        huc12_data = loader.load_huc12_watersheds()
        logger.info("HUC12 data loaded successfully")
        
        # Load FIRMS data (subset for testing)
        firms_data = loader.load_firms_data(
            start_date="2020-01-01", 
            end_date="2020-12-31"
        )
        logger.info("FIRMS data loaded successfully") 
        
        # Get dataset info
        info = loader.get_dataset_info()
        logger.info(f"Dataset info: {info}")
        
        # Export sample for testing
        sample = loader.export_watershed_sample(
            n_watersheds=5,
            output_path=RAW_DATA_DIR / "sample_watersheds.geojson"
        )
        logger.info(f"Sample created with {len(sample)} watersheds")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()