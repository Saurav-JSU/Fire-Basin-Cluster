"""
FIRMS fire data preprocessing and fire event identification.

This module handles:
1. FIRMS fire detection point filtering and validation
2. Spatial-temporal clustering to identify individual fire events
3. Fire event characterization (duration, area, intensity)
4. Data export and quality control

Uses real FIRMS data from Google Earth Engine - no simulated data.
"""
import ee
import geemap
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import json
from tqdm import tqdm
import warnings

# Robust import of config settings
try:
    from config.settings import (
        FIRE_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
        PROCESSING_CONFIG, get_fire_confidence_threshold
    )
except ImportError:
    # Fallback configuration
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    FIRE_CONFIG = {
        "confidence_threshold": 80,
        "spatial_threshold_degrees": 0.01,
        "temporal_threshold_days": 5,
        "fire_end_threshold_days": 16,
        "min_fire_size_ha": 1.0,
    }
    
    PROCESSING_CONFIG = {
        "n_jobs": -1,
        "chunk_size": 1000,
        "batch_size": 100,
    }
    
    def get_fire_confidence_threshold():
        return FIRE_CONFIG["confidence_threshold"]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FIRMSPreprocessor:
    """
    FIRMS fire data preprocessor for spatial-temporal fire event identification.
    
    This class converts individual FIRMS fire detection points into coherent 
    fire events using spatial-temporal clustering algorithms based on research
    from global fire database studies.
    """
    
    def __init__(self, spatial_threshold: Optional[float] = None,
                 temporal_threshold: Optional[int] = None,
                 confidence_threshold: Optional[int] = None):
        """
        Initialize FIRMS preprocessor.
        
        Args:
            spatial_threshold: Spatial clustering threshold in degrees (default: 0.01°)
            temporal_threshold: Temporal clustering threshold in days (default: 5)
            confidence_threshold: Minimum fire confidence percentage (default: 80)
        """
        self.spatial_threshold = spatial_threshold or FIRE_CONFIG["spatial_threshold_degrees"]
        self.temporal_threshold = temporal_threshold or FIRE_CONFIG["temporal_threshold_days"]
        self.confidence_threshold = confidence_threshold or get_fire_confidence_threshold()
        
        logger.info(f"FIRMS Preprocessor initialized:")
        logger.info(f"  - Spatial threshold: {self.spatial_threshold}°")
        logger.info(f"  - Temporal threshold: {self.temporal_threshold} days")
        logger.info(f"  - Confidence threshold: {self.confidence_threshold}%")
        
        # Cache for processed data
        self.fire_detections = None
        self.fire_events = None
        
    def extract_firms_points_from_gee(self, firms_collection: ee.ImageCollection,
                                    geometry: ee.Geometry,
                                    max_pixels: int = 50000) -> pd.DataFrame:
        """
        Extract FIRMS fire detection points from Google Earth Engine ImageCollection.
        
        Args:
            firms_collection: Filtered FIRMS ImageCollection from GEE
            geometry: Study area geometry (e.g., HUC12 watershed)
            max_pixels: Maximum number of pixels to sample
            
        Returns:
            pd.DataFrame: Fire detection points with metadata
        """
        logger.info("Extracting FIRMS detection points from Google Earth Engine")
        
        try:
            # Get collection size
            collection_size = firms_collection.size().getInfo()
            logger.info(f"Processing {collection_size} FIRMS images")
            
            if collection_size == 0:
                logger.warning("No FIRMS images found in collection")
                return pd.DataFrame()
            
            # Function to extract fire points from each image
            def extract_fire_points(image):
                """Extract fire detection points from a single FIRMS image."""
                
                # Get image date and metadata
                date = image.date()
                
                # ✅ Use correct FIRMS bands: ['T21', 'confidence', 'line_number']
                # T21 = brightness temperature, confidence = detection confidence
                
                # Apply confidence threshold at PIXEL level
                confidence_band = image.select('confidence')
                confidence_mask = confidence_band.gt(self.confidence_threshold)
                
                # Create fire detection mask - any pixel with valid confidence > threshold
                fire_mask = confidence_mask
                
                # Sample points from fire pixels
                fire_points = fire_mask.sample(
                    region=geometry,
                    scale=1000,  # 1km resolution
                    numPixels=max_pixels,
                    geometries=True
                )
                
                # Add metadata to each point
                def add_metadata(feature):
                    # Get coordinates
                    coords = feature.geometry().coordinates()
                    
                    # Sample all bands at this location
                    point_data = image.sample(
                        region=feature.geometry(),
                        scale=1000,
                        numPixels=1,
                        geometries=False
                    ).first()
                    
                    return feature.set({
                        'date': date.format('YYYY-MM-dd'),
                        'timestamp': date.millis(),
                        'longitude': ee.Number(coords.get(0)),
                        'latitude': ee.Number(coords.get(1)),
                        'year': date.get('year'),
                        'month': date.get('month'),
                        'day': date.get('day'),
                        'doy': date.getRelative('day', 'year').add(1),  # Day of year
                        'T21': point_data.get('T21'),  # Brightness temperature
                        'confidence': point_data.get('confidence'),  # Detection confidence
                        'line_number': point_data.get('line_number')  # FIRMS line number
                    })
                
                return fire_points.map(add_metadata)
            
            # Map over all images and flatten
            all_fire_points = firms_collection.map(extract_fire_points).flatten()
            
            # Get the count of extracted points
            points_count = all_fire_points.size().getInfo()
            logger.info(f"Extracted {points_count} fire detection points")
            
            if points_count == 0:
                logger.warning("No fire detection points found")
                return pd.DataFrame()
            
            # Convert to pandas DataFrame using geemap
            logger.info("Converting to pandas DataFrame...")
            df = geemap.ee_to_df(all_fire_points)
            
            logger.info(f"Successfully converted to DataFrame: {len(df)} records")
            
            return self._standardize_firms_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error extracting FIRMS points from GEE: {e}")
            raise

    def _standardize_firms_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize FIRMS DataFrame to consistent format.
        
        Args:
            df: Raw FIRMS DataFrame from GEE
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        logger.info("Standardizing FIRMS DataFrame format")
        
        # Ensure required columns exist
        required_columns = ['longitude', 'latitude', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Standardize column names and types
        df_std = df.copy()
        
        # Coordinate standardization
        df_std['longitude'] = pd.to_numeric(df_std['longitude'], errors='coerce')
        df_std['latitude'] = pd.to_numeric(df_std['latitude'], errors='coerce')
        
        # Date standardization
        df_std['date'] = pd.to_datetime(df_std['date'], errors='coerce')
        
        # Standardize FIRMS-specific columns
        if 'T21' in df_std.columns:
            df_std['T21'] = pd.to_numeric(df_std['T21'], errors='coerce')
            # Convert T21 (brightness temperature) to FRP proxy if needed
            # For compatibility with downstream code expecting 'frp'
            df_std['frp'] = df_std['T21']  # Use T21 as FRP proxy
        
        if 'confidence' in df_std.columns:
            df_std['confidence'] = pd.to_numeric(df_std['confidence'], errors='coerce')
        else:
            df_std['confidence'] = 90  # Default high confidence
        
        # Add default values for expected columns
        if 'daynight' not in df_std.columns:
            df_std['daynight'] = 'D'  # Default to day
        
        if 'satellite' not in df_std.columns:
            df_std['satellite'] = 'MODIS'  # FIRMS is MODIS-based
        
        # Data validation and cleaning
        initial_count = len(df_std)
        
        # Remove rows with invalid coordinates or dates
        df_std = df_std.dropna(subset=['longitude', 'latitude', 'date'])
        
        # Validate coordinate ranges
        df_std = df_std[
            (df_std['longitude'] >= -180) & (df_std['longitude'] <= 180) &
            (df_std['latitude'] >= -90) & (df_std['latitude'] <= 90)
        ]
        
        # Confidence was already filtered during extraction, but double-check
        if 'confidence' in df_std.columns:
            df_std = df_std[df_std['confidence'] >= self.confidence_threshold]
        
        final_count = len(df_std)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} invalid records ({removed_count/initial_count*100:.1f}%)")
        
        logger.info(f"Standardized DataFrame: {final_count} valid fire detections")
        
        # Cache the processed data
        self.fire_detections = df_std.copy()
        
        return df_std
    
    def load_firms_data_from_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load FIRMS data from local file (CSV, JSON, or GeoJSON).
        
        Args:
            file_path: Path to FIRMS data file
            
        Returns:
            pd.DataFrame: Fire detection points
        """
        file_path = Path(file_path)
        logger.info(f"Loading FIRMS data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"FIRMS data file not found: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            elif file_path.suffix.lower() == '.geojson':
                gdf = gpd.read_file(file_path)
                df = pd.DataFrame(gdf.drop(columns=['geometry']))
                # Extract coordinates
                df['longitude'] = gdf.geometry.x
                df['latitude'] = gdf.geometry.y
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(df)} fire detection records from file")
            return self._standardize_firms_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error loading FIRMS data from file: {e}")
            raise
    
    def identify_fire_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify individual fire events using spatial-temporal clustering.
        
        Uses DBSCAN clustering with spatial and temporal constraints based on
        research from global fire databases (San-Miguel-Ayanz et al., 2019).
        
        Args:
            df: Fire detection points DataFrame
            
        Returns:
            pd.DataFrame: Fire events with event IDs and characteristics
        """
        logger.info("Identifying fire events using spatial-temporal clustering")
        
        if len(df) == 0:
            logger.warning("No fire detections to cluster")
            return pd.DataFrame()
        
        # Prepare data for clustering
        df_work = df.copy()
        df_work['datetime'] = pd.to_datetime(df_work['date'])
        df_work = df_work.sort_values('datetime')
        
        # Convert dates to numeric (days since first detection)
        start_date = df_work['datetime'].min()
        df_work['days_since_start'] = (df_work['datetime'] - start_date).dt.days
        
        # Prepare features for clustering: [lon, lat, temporal_weight]
        # Scale temporal dimension to match spatial scale
        temporal_scale = self.spatial_threshold / self.temporal_threshold
        
        features = np.column_stack([
            df_work['longitude'].values,
            df_work['latitude'].values,
            df_work['days_since_start'].values * temporal_scale
        ])
        
        logger.info(f"Clustering {len(features)} detections with DBSCAN")
        logger.info(f"  - Spatial threshold: {self.spatial_threshold}°")
        logger.info(f"  - Temporal scale factor: {temporal_scale:.6f}")
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(
            eps=self.spatial_threshold,
            min_samples=1,  # Single detection can be a fire event
            metric='euclidean',
            n_jobs=PROCESSING_CONFIG.get('n_jobs', -1)
        )
        
        cluster_labels = dbscan.fit_predict(features)
        
        # Add cluster labels to dataframe
        df_work['event_id'] = cluster_labels
        
        # Remove noise points (label = -1)
        noise_points = np.sum(cluster_labels == -1)
        if noise_points > 0:
            logger.info(f"Removed {noise_points} noise points")
            df_work = df_work[df_work['event_id'] != -1]
        
        n_events = len(np.unique(df_work['event_id']))
        logger.info(f"Identified {n_events} fire events from {len(df_work)} detections")
        
        # Post-process events
        df_events = self._characterize_fire_events(df_work)
        
        # Cache the results
        self.fire_events = df_events.copy()
        
        return df_events
    
    def _characterize_fire_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate characteristics for each fire event.
        
        Args:
            df: Fire detections with event IDs
            
        Returns:
            pd.DataFrame: Event-level characteristics
        """
        logger.info("Characterizing fire events")
        
        event_stats = []
        
        for event_id in tqdm(df['event_id'].unique(), desc="Processing events"):
            event_data = df[df['event_id'] == event_id].copy()
            
            # Basic event information
            stats = {
                'event_id': event_id,
                'n_detections': len(event_data),
                'start_date': event_data['datetime'].min(),
                'end_date': event_data['datetime'].max(),
                'duration_days': (event_data['datetime'].max() - event_data['datetime'].min()).days + 1,
            }
            
            # Spatial characteristics
            stats.update({
                'centroid_lon': event_data['longitude'].mean(),
                'centroid_lat': event_data['latitude'].mean(),
                'lon_range': event_data['longitude'].max() - event_data['longitude'].min(),
                'lat_range': event_data['latitude'].max() - event_data['latitude'].min(),
                'bbox_area_deg2': (event_data['longitude'].max() - event_data['longitude'].min()) * 
                                  (event_data['latitude'].max() - event_data['latitude'].min()),
            })
            
            # Intensity characteristics (if FRP data available)
            if 'frp' in event_data.columns and not event_data['frp'].isna().all():
                frp_valid = event_data['frp'].dropna()
                if len(frp_valid) > 0:
                    stats.update({
                        'mean_frp': frp_valid.mean(),
                        'max_frp': frp_valid.max(),
                        'total_frp': frp_valid.sum(),
                        'frp_available': True
                    })
                else:
                    stats.update({
                        'mean_frp': np.nan,
                        'max_frp': np.nan,
                        'total_frp': np.nan,
                        'frp_available': False
                    })
            else:
                stats.update({
                    'mean_frp': np.nan,
                    'max_frp': np.nan,
                    'total_frp': np.nan,
                    'frp_available': False
                })
            
            # Confidence characteristics
            if 'confidence' in event_data.columns:
                stats.update({
                    'mean_confidence': event_data['confidence'].mean(),
                    'min_confidence': event_data['confidence'].min(),
                    'max_confidence': event_data['confidence'].max(),
                })
            
            # Temporal patterns
            event_data['hour'] = event_data['datetime'].dt.hour
            event_data['day_of_year'] = event_data['datetime'].dt.dayofyear
            
            stats.update({
                'peak_hour': event_data.groupby('hour').size().idxmax() if len(event_data) > 1 else event_data['hour'].iloc[0],
                'peak_day_of_year': event_data['day_of_year'].iloc[0],  # Use first detection's day of year
                'season': self._get_season(event_data['day_of_year'].iloc[0])
            })
            
            event_stats.append(stats)
        
        events_df = pd.DataFrame(event_stats)
        
        # Add derived metrics
        events_df['detection_density'] = events_df['n_detections'] / events_df['duration_days']
        
        # Convert spatial extent to kilometers (rough approximation)
        # 1 degree ≈ 111 km at equator
        events_df['spatial_extent_km'] = np.sqrt(events_df['bbox_area_deg2']) * 111
        
        # Quality flags
        events_df['is_single_detection'] = events_df['n_detections'] == 1
        events_df['is_short_duration'] = events_df['duration_days'] <= 1
        events_df['is_large_spatial'] = events_df['spatial_extent_km'] > 50  # >50km extent
        
        logger.info(f"Event characteristics calculated for {len(events_df)} events")
        
        # Log summary statistics
        if len(events_df) > 0:
            logger.info(f"Event summary:")
            logger.info(f"  - Mean duration: {events_df['duration_days'].mean():.1f} days")
            logger.info(f"  - Mean detections per event: {events_df['n_detections'].mean():.1f}")
            logger.info(f"  - Single detection events: {events_df['is_single_detection'].sum()} ({events_df['is_single_detection'].mean()*100:.1f}%)")
            if not events_df['frp_available'].isna().all():
                logger.info(f"  - Events with FRP data: {events_df['frp_available'].sum()}")
        
        return events_df
    
    def _get_season(self, day_of_year: int) -> str:
        """
        Get season from day of year (Northern Hemisphere).
        
        Args:
            day_of_year: Day of year (1-365/366)
            
        Returns:
            str: Season name
        """
        if day_of_year < 80 or day_of_year >= 355:  # Dec 21 - Mar 20
            return 'Winter'
        elif day_of_year < 172:  # Mar 21 - Jun 20
            return 'Spring'
        elif day_of_year < 266:  # Jun 21 - Sep 22
            return 'Summer'
        else:  # Sep 23 - Dec 20
            return 'Fall'
    
    def filter_events(self, events_df: pd.DataFrame, 
                     min_detections: int = 1,
                     min_duration_days: int = 1,
                     max_spatial_extent_km: float = 500,
                     remove_single_detections: bool = False) -> pd.DataFrame:
        """
        Filter fire events based on quality criteria.
        
        Args:
            events_df: Fire events DataFrame
            min_detections: Minimum number of detections per event
            min_duration_days: Minimum duration in days
            max_spatial_extent_km: Maximum spatial extent in km
            remove_single_detections: Whether to remove single detection events
            
        Returns:
            pd.DataFrame: Filtered events
        """
        logger.info("Filtering fire events based on quality criteria")
        
        initial_count = len(events_df)
        
        if initial_count == 0:
            logger.warning("No events to filter")
            return events_df
        
        # Apply filters
        filtered = events_df.copy()
        
        if min_detections > 1:
            filtered = filtered[filtered['n_detections'] >= min_detections]
            
        if min_duration_days > 1:
            filtered = filtered[filtered['duration_days'] >= min_duration_days]
            
        if max_spatial_extent_km < float('inf'):
            filtered = filtered[filtered['spatial_extent_km'] <= max_spatial_extent_km]
            
        if remove_single_detections:
            filtered = filtered[~filtered['is_single_detection']]
        
        final_count = len(filtered)
        removed_count = initial_count - final_count
        
        logger.info(f"Event filtering results:")
        logger.info(f"  - Initial events: {initial_count}")
        logger.info(f"  - Final events: {final_count}")
        logger.info(f"  - Removed: {removed_count} ({removed_count/initial_count*100:.1f}%)")
        
        return filtered
    
    def export_events(self, events_df: pd.DataFrame, 
                     output_path: Optional[Union[str, Path]] = None,
                     format: str = 'csv') -> Path:
        """
        Export fire events to file.
        
        Args:
            events_df: Fire events DataFrame to export
            output_path: Output file path (auto-generated if None)
            format: Export format ('csv', 'json', 'parquet')
            
        Returns:
            Path: Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_events_{timestamp}.{format}"
            output_path = PROCESSED_DATA_DIR / filename
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(events_df)} fire events to {output_path}")
        
        try:
            if format.lower() == 'csv':
                events_df.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                # Handle datetime objects for JSON export
                events_df_json = events_df.copy()
                for col in events_df_json.columns:
                    if events_df_json[col].dtype == 'datetime64[ns]':
                        events_df_json[col] = events_df_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                events_df_json.to_json(output_path, orient='records', indent=2)
            elif format.lower() == 'parquet':
                events_df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Export completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting events: {e}")
            raise
    
    def get_processing_summary(self) -> Dict:
        """
        Get summary of preprocessing results.
        
        Returns:
            Dict: Processing summary statistics
        """
        summary = {
            'configuration': {
                'spatial_threshold_degrees': self.spatial_threshold,
                'temporal_threshold_days': self.temporal_threshold,
                'confidence_threshold_percent': self.confidence_threshold,
            },
            'data_summary': {}
        }
        
        if self.fire_detections is not None:
            summary['data_summary']['fire_detections'] = {
                'total_detections': len(self.fire_detections),
                'date_range': {
                    'start': self.fire_detections['date'].min().isoformat() if hasattr(self.fire_detections['date'].min(), 'isoformat') else str(self.fire_detections['date'].min()),
                    'end': self.fire_detections['date'].max().isoformat() if hasattr(self.fire_detections['date'].max(), 'isoformat') else str(self.fire_detections['date'].max())
                },
                'confidence_stats': {
                    'mean': float(self.fire_detections['confidence'].mean()),
                    'min': float(self.fire_detections['confidence'].min()),
                    'max': float(self.fire_detections['confidence'].max())
                } if 'confidence' in self.fire_detections.columns else None
            }
        
        if self.fire_events is not None:
            summary['data_summary']['fire_events'] = {
                'total_events': len(self.fire_events),
                'duration_stats_days': {
                    'mean': float(self.fire_events['duration_days'].mean()),
                    'median': float(self.fire_events['duration_days'].median()),
                    'max': float(self.fire_events['duration_days'].max())
                },
                'spatial_extent_stats_km': {
                    'mean': float(self.fire_events['spatial_extent_km'].mean()),
                    'median': float(self.fire_events['spatial_extent_km'].median()),
                    'max': float(self.fire_events['spatial_extent_km'].max())
                },
                'single_detection_events': int(self.fire_events['is_single_detection'].sum()),
                'large_spatial_events': int(self.fire_events['is_large_spatial'].sum())
            }
        
        return summary