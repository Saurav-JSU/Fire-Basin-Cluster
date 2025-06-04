"""
Watershed Fire Metrics Calculator

This module calculates comprehensive fire regime characteristics for HUC12 watersheds
based on fire events identified from FIRMS data. These metrics serve as features
for watershed clustering analysis.

Key Metrics Calculated:
1. Fire Return Intervals (temporal patterns)
2. Burn Fractions (spatial coverage)  
3. Fire Seasonality (temporal distribution)
4. Fire Intensity Aggregations (FRP-based metrics)
5. Composite Fire Regime Indicators

Uses only real fire event data - no simulations.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
from tqdm import tqdm

# Robust import of config settings
try:
    from config.settings import (
        FIRE_METRICS, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        PROCESSING_CONFIG, get_study_area_bounds
    )
except ImportError:
    # Fallback configuration
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    
    FIRE_METRICS = {
        "temporal_metrics": [
            "fire_frequency", "mean_fire_return_interval", "median_fire_return_interval",
            "fire_return_interval_std", "max_fire_free_period", "time_since_last_fire",
            "fire_season_peak", "fire_season_length"
        ],
        "spatial_metrics": [
            "total_burned_fraction", "mean_fire_size", "median_fire_size",
            "fire_size_cv", "largest_fire_fraction", "repeat_burn_fraction"
        ],
        "intensity_metrics": [
            "mean_frp", "max_frp", "fire_duration_mean", "fire_duration_max",
            "peak_fire_month"
        ]
    }
    
    PROCESSING_CONFIG = {
        "n_jobs": -1,
        "chunk_size": 1000
    }
    
    def get_study_area_bounds():
        return {"west": -125.0, "east": -102.0, "south": 31.0, "north": 49.0}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatershedFireMetrics:
    """
    Calculate comprehensive fire regime metrics for HUC12 watersheds.
    
    This class processes fire events and watershed boundaries to compute
    quantitative fire characteristics that capture the fire regime of each
    watershed for clustering analysis.
    """
    
    def __init__(self, study_period_years: Optional[int] = None):
        """
        Initialize watershed fire metrics calculator.
        
        Args:
            study_period_years: Length of study period in years (for rate calculations)
        """
        self.study_period_years = study_period_years
        
        # Cache for processed data
        self.watersheds = None
        self.fire_events = None
        self.watershed_metrics = None
        
        logger.info("Watershed Fire Metrics Calculator initialized")
        if study_period_years:
            logger.info(f"Study period: {study_period_years} years")
    
    def load_watershed_boundaries(self, watershed_file: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Load HUC12 watershed boundaries.
        
        Args:
            watershed_file: Path to watershed boundaries file (GeoJSON, Shapefile, etc.)
            
        Returns:
            gpd.GeoDataFrame: Watershed boundaries with HUC12 IDs
        """
        watershed_file = Path(watershed_file)
        logger.info(f"Loading watershed boundaries from {watershed_file}")
        
        if not watershed_file.exists():
            raise FileNotFoundError(f"Watershed file not found: {watershed_file}")
        
        try:
            watersheds_gdf = gpd.read_file(watershed_file)
            logger.info(f"Loaded {len(watersheds_gdf)} watershed boundaries")
            
            # Validate required columns
            if 'huc12' not in watersheds_gdf.columns:
                # Try common alternative names
                huc_cols = [col for col in watersheds_gdf.columns 
                           if 'huc' in col.lower() or 'id' in col.lower()]
                if huc_cols:
                    watersheds_gdf = watersheds_gdf.rename(columns={huc_cols[0]: 'huc12'})
                    logger.info(f"Renamed column '{huc_cols[0]}' to 'huc12'")
                else:
                    raise ValueError("No HUC12 ID column found in watershed data")
            
            # Ensure CRS is geographic (needed for area calculations)
            if watersheds_gdf.crs is None:
                logger.warning("No CRS found, assuming EPSG:4326")
                watersheds_gdf = watersheds_gdf.set_crs("EPSG:4326")
            elif not watersheds_gdf.crs.is_geographic:
                logger.info(f"Converting from {watersheds_gdf.crs} to EPSG:4326")
                watersheds_gdf = watersheds_gdf.to_crs("EPSG:4326")
            
            # Calculate watershed areas (in km²)
            watersheds_gdf['area_km2'] = watersheds_gdf.to_crs("EPSG:3857").area / 1e6
            
            # Cache the data
            self.watersheds = watersheds_gdf.copy()
            
            logger.info(f"Watershed summary:")
            logger.info(f"  - Total watersheds: {len(watersheds_gdf)}")
            logger.info(f"  - Area range: {watersheds_gdf['area_km2'].min():.1f} - {watersheds_gdf['area_km2'].max():.1f} km²")
            logger.info(f"  - Mean area: {watersheds_gdf['area_km2'].mean():.1f} km²")
            
            return watersheds_gdf
            
        except Exception as e:
            logger.error(f"Error loading watershed boundaries: {e}")
            raise
    
    def load_fire_events(self, fire_events_file: Union[str, Path]) -> pd.DataFrame:
        """
        Load fire events data (from Step 2 preprocessing).
        
        Args:
            fire_events_file: Path to fire events file (CSV, JSON, Parquet)
            
        Returns:
            pd.DataFrame: Fire events with characteristics
        """
        fire_events_file = Path(fire_events_file)
        logger.info(f"Loading fire events from {fire_events_file}")
        
        if not fire_events_file.exists():
            raise FileNotFoundError(f"Fire events file not found: {fire_events_file}")
        
        try:
            # Load based on file extension
            if fire_events_file.suffix.lower() == '.csv':
                fire_events_df = pd.read_csv(fire_events_file)
            elif fire_events_file.suffix.lower() == '.json':
                fire_events_df = pd.read_json(fire_events_file)
            elif fire_events_file.suffix.lower() == '.parquet':
                fire_events_df = pd.read_parquet(fire_events_file)
            else:
                raise ValueError(f"Unsupported file format: {fire_events_file.suffix}")
            
            logger.info(f"Loaded {len(fire_events_df)} fire events")
            
            # Validate required columns
            required_cols = ['event_id', 'start_date', 'end_date', 'centroid_lon', 'centroid_lat']
            missing_cols = [col for col in required_cols if col not in fire_events_df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns in fire events: {missing_cols}")
            
            # Convert date columns
            for date_col in ['start_date', 'end_date']:
                if not pd.api.types.is_datetime64_any_dtype(fire_events_df[date_col]):
                    fire_events_df[date_col] = pd.to_datetime(fire_events_df[date_col])
            
            # Calculate derived temporal metrics
            fire_events_df['duration_days'] = (fire_events_df['end_date'] - fire_events_df['start_date']).dt.days + 1
            fire_events_df['year'] = fire_events_df['start_date'].dt.year
            fire_events_df['month'] = fire_events_df['start_date'].dt.month
            fire_events_df['day_of_year'] = fire_events_df['start_date'].dt.dayofyear
            
            # Add season information
            fire_events_df['season'] = fire_events_df['month'].map(self._month_to_season)
            
            # Cache the data
            self.fire_events = fire_events_df.copy()
            
            # Determine study period if not provided
            if self.study_period_years is None:
                year_range = fire_events_df['year'].max() - fire_events_df['year'].min() + 1
                self.study_period_years = year_range
                logger.info(f"Determined study period: {year_range} years ({fire_events_df['year'].min()}-{fire_events_df['year'].max()})")
            
            logger.info(f"Fire events summary:")
            logger.info(f"  - Date range: {fire_events_df['start_date'].min().date()} to {fire_events_df['end_date'].max().date()}")
            logger.info(f"  - Mean duration: {fire_events_df['duration_days'].mean():.1f} days")
            logger.info(f"  - Events per year: {len(fire_events_df) / self.study_period_years:.1f}")
            
            return fire_events_df
            
        except Exception as e:
            logger.error(f"Error loading fire events: {e}")
            raise
    
    def _month_to_season(self, month: int) -> str:
        """Convert month number to season (Northern Hemisphere)."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # month in [9, 10, 11]
            return 'Fall'
    
    def intersect_fires_with_watersheds(self, buffer_km: float = 0.0) -> pd.DataFrame:
        """
        Intersect fire events with watershed boundaries to determine which fires
        affect which watersheds.
        
        Args:
            buffer_km: Buffer around fire centroids in kilometers
            
        Returns:
            pd.DataFrame: Fire-watershed intersection data
        """
        if self.watersheds is None or self.fire_events is None:
            raise ValueError("Must load watersheds and fire events first")
        
        logger.info("Intersecting fire events with watershed boundaries")
        
        # Convert fire events to GeoDataFrame
        fire_gdf = gpd.GeoDataFrame(
            self.fire_events,
            geometry=gpd.points_from_xy(
                self.fire_events['centroid_lon'], 
                self.fire_events['centroid_lat']
            ),
            crs="EPSG:4326"
        )
        
        # Apply buffer if specified
        if buffer_km > 0:
            # Convert to projected CRS for accurate buffering
            fire_gdf_proj = fire_gdf.to_crs("EPSG:3857")
            fire_gdf_proj['geometry'] = fire_gdf_proj.geometry.buffer(buffer_km * 1000)  # Convert km to meters
            fire_gdf = fire_gdf_proj.to_crs("EPSG:4326")
            logger.info(f"Applied {buffer_km} km buffer to fire locations")
        
        # Spatial join
        logger.info("Performing spatial intersection...")
        fire_watershed_intersections = gpd.sjoin(
            fire_gdf, 
            self.watersheds[['huc12', 'area_km2', 'geometry']], 
            how='inner', 
            predicate='intersects'
        )
        
        # Clean up the result
        intersection_df = fire_watershed_intersections.drop(columns=['geometry', 'index_right'])
        
        logger.info(f"Intersection results:")
        logger.info(f"  - Fire events intersecting watersheds: {len(intersection_df)}")
        logger.info(f"  - Unique watersheds with fires: {intersection_df['huc12'].nunique()}")
        logger.info(f"  - Fires per watershed (mean): {len(intersection_df) / intersection_df['huc12'].nunique():.1f}")
        
        return intersection_df
    
    def calculate_fire_return_intervals(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fire return interval statistics for each watershed.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            
        Returns:
            pd.DataFrame: Fire return interval metrics per watershed
        """
        logger.info("Calculating fire return intervals")
        
        fri_metrics = []
        
        for huc12_id in tqdm(fire_watershed_df['huc12'].unique(), desc="Processing watersheds"):
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            watershed_fires = watershed_fires.sort_values('start_date')
            
            # Basic fire frequency metrics
            n_fires = len(watershed_fires)
            years_with_fire = watershed_fires['year'].nunique()
            
            metrics = {
                'huc12': huc12_id,
                'n_fires': n_fires,
                'years_with_fire': years_with_fire,
                'fire_frequency_per_year': n_fires / self.study_period_years,
                'fire_years_fraction': years_with_fire / self.study_period_years
            }
            
            if n_fires > 1:
                # Calculate return intervals
                fire_dates = watershed_fires['start_date'].dt.date.unique()
                fire_dates = sorted(fire_dates)
                
                intervals_days = []
                for i in range(1, len(fire_dates)):
                    interval = (fire_dates[i] - fire_dates[i-1]).days
                    intervals_days.append(interval)
                
                if intervals_days:
                    intervals_years = np.array(intervals_days) / 365.25
                    
                    metrics.update({
                        'mean_fire_return_interval_years': np.mean(intervals_years),
                        'median_fire_return_interval_years': np.median(intervals_years),
                        'std_fire_return_interval_years': np.std(intervals_years),
                        'min_fire_return_interval_years': np.min(intervals_years),
                        'max_fire_return_interval_years': np.max(intervals_years),
                        'cv_fire_return_interval': np.std(intervals_years) / np.mean(intervals_years) if np.mean(intervals_years) > 0 else 0
                    })
                else:
                    # Multiple fires on same date
                    metrics.update({
                        'mean_fire_return_interval_years': 0,
                        'median_fire_return_interval_years': 0,
                        'std_fire_return_interval_years': 0,
                        'min_fire_return_interval_years': 0,
                        'max_fire_return_interval_years': 0,
                        'cv_fire_return_interval': 0
                    })
            else:
                # Single fire or no fires
                metrics.update({
                    'mean_fire_return_interval_years': np.nan,
                    'median_fire_return_interval_years': np.nan,
                    'std_fire_return_interval_years': np.nan,
                    'min_fire_return_interval_years': np.nan,
                    'max_fire_return_interval_years': np.nan,
                    'cv_fire_return_interval': np.nan
                })
            
            # Time since last fire
            if n_fires > 0:
                last_fire_date = watershed_fires['start_date'].max()
                current_date = datetime.now()
                days_since_last_fire = (current_date - last_fire_date).days
                metrics['years_since_last_fire'] = days_since_last_fire / 365.25
            else:
                metrics['years_since_last_fire'] = np.nan
            
            # Fire-free periods
            all_years = set(range(
                watershed_fires['year'].min() if n_fires > 0 else datetime.now().year,
                watershed_fires['year'].max() + 1 if n_fires > 0 else datetime.now().year + 1
            ))
            fire_years = set(watershed_fires['year'].unique())
            fire_free_years = all_years - fire_years
            
            if fire_free_years:
                # Find longest consecutive fire-free period
                fire_free_years_sorted = sorted(fire_free_years)
                max_consecutive = 1
                current_consecutive = 1
                
                for i in range(1, len(fire_free_years_sorted)):
                    if fire_free_years_sorted[i] == fire_free_years_sorted[i-1] + 1:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 1
                
                metrics['max_fire_free_period_years'] = max_consecutive
            else:
                metrics['max_fire_free_period_years'] = 0
            
            fri_metrics.append(metrics)
        
        fri_df = pd.DataFrame(fri_metrics)
        
        logger.info(f"Fire return interval statistics:")
        if len(fri_df) > 0:
            logger.info(f"  - Watersheds with fires: {len(fri_df)}")
            logger.info(f"  - Mean fires per watershed: {fri_df['n_fires'].mean():.1f}")
            logger.info(f"  - Mean fire frequency: {fri_df['fire_frequency_per_year'].mean():.2f} fires/year")
        
        return fri_df
    
    def calculate_burn_fractions(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate burn fraction metrics for each watershed.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            
        Returns:
            pd.DataFrame: Burn fraction metrics per watershed
        """
        logger.info("Calculating burn fractions")
        
        burn_metrics = []
        
        for huc12_id in tqdm(fire_watershed_df['huc12'].unique(), desc="Processing burn fractions"):
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            
            # Get watershed area
            watershed_area_km2 = self.watersheds[self.watersheds['huc12'] == huc12_id]['area_km2'].iloc[0]
            
            metrics = {
                'huc12': huc12_id,
                'watershed_area_km2': watershed_area_km2,
                'n_fires': len(watershed_fires)
            }
            
            if len(watershed_fires) > 0:
                # Calculate fire size statistics (if spatial_extent_km available)
                if 'spatial_extent_km' in watershed_fires.columns:
                    fire_sizes = watershed_fires['spatial_extent_km'].dropna()
                    
                    if len(fire_sizes) > 0:
                        # Approximate burned area from spatial extent
                        # Note: This is rough - spatial_extent_km is diagonal, actual area is smaller
                        estimated_fire_areas = np.pi * (fire_sizes / 2) ** 2  # Assume circular fires
                        
                        total_fire_area = estimated_fire_areas.sum()
                        
                        metrics.update({
                            'total_estimated_burned_area_km2': total_fire_area,
                            'burned_fraction_estimate': min(total_fire_area / watershed_area_km2, 1.0),
                            'mean_fire_size_km2': estimated_fire_areas.mean(),
                            'median_fire_size_km2': estimated_fire_areas.median(),
                            'largest_fire_size_km2': estimated_fire_areas.max(),
                            'largest_fire_fraction': min(estimated_fire_areas.max() / watershed_area_km2, 1.0),
                            'fire_size_cv': estimated_fire_areas.std() / estimated_fire_areas.mean() if estimated_fire_areas.mean() > 0 else 0
                        })
                    else:
                        # No valid fire size data
                        metrics.update({
                            'total_estimated_burned_area_km2': np.nan,
                            'burned_fraction_estimate': np.nan,
                            'mean_fire_size_km2': np.nan,
                            'median_fire_size_km2': np.nan,
                            'largest_fire_size_km2': np.nan,
                            'largest_fire_fraction': np.nan,
                            'fire_size_cv': np.nan
                        })
                else:
                    # Use detection density as proxy for burn coverage
                    total_detections = watershed_fires['n_detections'].sum() if 'n_detections' in watershed_fires.columns else len(watershed_fires)
                    
                    # Rough approximation: each detection represents ~1 km²
                    estimated_burned_area = total_detections * 1.0  # km²
                    
                    metrics.update({
                        'total_estimated_burned_area_km2': estimated_burned_area,
                        'burned_fraction_estimate': min(estimated_burned_area / watershed_area_km2, 1.0),
                        'mean_fire_size_km2': estimated_burned_area / len(watershed_fires),
                        'median_fire_size_km2': np.nan,  # Can't calculate without individual fire sizes
                        'largest_fire_size_km2': np.nan,
                        'largest_fire_fraction': np.nan,
                        'fire_size_cv': np.nan
                    })
                
                # Repeat burn analysis (fires in different years at similar locations)
                if len(watershed_fires) > 1:
                    # Group fires by year and check for spatial overlap
                    yearly_fires = watershed_fires.groupby('year')
                    years_with_multiple_fires = sum(1 for year, fires in yearly_fires if len(fires) > 1)
                    
                    metrics.update({
                        'years_with_multiple_fires': years_with_multiple_fires,
                        'multiple_fire_years_fraction': years_with_multiple_fires / len(yearly_fires)
                    })
                else:
                    metrics.update({
                        'years_with_multiple_fires': 0,
                        'multiple_fire_years_fraction': 0
                    })
            else:
                # No fires in this watershed
                metrics.update({
                    'total_estimated_burned_area_km2': 0,
                    'burned_fraction_estimate': 0,
                    'mean_fire_size_km2': 0,
                    'median_fire_size_km2': 0,
                    'largest_fire_size_km2': 0,
                    'largest_fire_fraction': 0,
                    'fire_size_cv': 0,
                    'years_with_multiple_fires': 0,
                    'multiple_fire_years_fraction': 0
                })
            
            burn_metrics.append(metrics)
        
        burn_df = pd.DataFrame(burn_metrics)
        
        logger.info(f"Burn fraction statistics:")
        if len(burn_df) > 0:
            logger.info(f"  - Mean burned fraction: {burn_df['burned_fraction_estimate'].mean():.3f}")
            logger.info(f"  - Max burned fraction: {burn_df['burned_fraction_estimate'].max():.3f}")
            logger.info(f"  - Watersheds with >10% burned: {(burn_df['burned_fraction_estimate'] > 0.1).sum()}")
        
        return burn_df
    
    def calculate_fire_seasonality(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fire seasonality metrics for each watershed.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            
        Returns:
            pd.DataFrame: Seasonality metrics per watershed
        """
        logger.info("Calculating fire seasonality patterns")
        
        seasonality_metrics = []
        
        for huc12_id in tqdm(fire_watershed_df['huc12'].unique(), desc="Processing seasonality"):
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            
            metrics = {
                'huc12': huc12_id,
                'n_fires': len(watershed_fires)
            }
            
            if len(watershed_fires) > 0:
                # Monthly fire distribution
                monthly_counts = watershed_fires['month'].value_counts().sort_index()
                peak_month = monthly_counts.idxmax()
                
                # Season distribution
                seasonal_counts = watershed_fires['season'].value_counts()
                peak_season = seasonal_counts.idxmax()
                
                # Fire season length (months with fires)
                months_with_fires = len(monthly_counts[monthly_counts > 0])
                
                # Day of year statistics
                doy_stats = watershed_fires['day_of_year'].describe()
                
                metrics.update({
                    'peak_fire_month': peak_month,
                    'peak_fire_season': peak_season,
                    'fire_season_length_months': months_with_fires,
                    'fire_concentration_peak_month': monthly_counts.max() / len(watershed_fires),
                    'fire_concentration_peak_season': seasonal_counts.max() / len(watershed_fires),
                    'mean_fire_day_of_year': doy_stats['mean'],
                    'std_fire_day_of_year': doy_stats['std'],
                    'earliest_fire_day_of_year': doy_stats['min'],
                    'latest_fire_day_of_year': doy_stats['max']
                })
                
                # Seasonal distribution percentages
                total_fires = len(watershed_fires)
                for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                    season_count = seasonal_counts.get(season, 0)
                    metrics[f'{season.lower()}_fire_fraction'] = season_count / total_fires
                
                # Monthly distribution (for detailed analysis)
                for month in range(1, 13):
                    month_count = monthly_counts.get(month, 0)
                    metrics[f'month_{month:02d}_fire_fraction'] = month_count / total_fires
                
            else:
                # No fires - set all metrics to appropriate defaults
                metrics.update({
                    'peak_fire_month': np.nan,
                    'peak_fire_season': 'None',
                    'fire_season_length_months': 0,
                    'fire_concentration_peak_month': 0,
                    'fire_concentration_peak_season': 0,
                    'mean_fire_day_of_year': np.nan,
                    'std_fire_day_of_year': np.nan,
                    'earliest_fire_day_of_year': np.nan,
                    'latest_fire_day_of_year': np.nan
                })
                
                # Seasonal fractions (all zero)
                for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                    metrics[f'{season.lower()}_fire_fraction'] = 0
                
                # Monthly fractions (all zero)
                for month in range(1, 13):
                    metrics[f'month_{month:02d}_fire_fraction'] = 0
            
            seasonality_metrics.append(metrics)
        
        seasonality_df = pd.DataFrame(seasonality_metrics)
        
        logger.info(f"Fire seasonality statistics:")
        if len(seasonality_df) > 0:
            logger.info(f"  - Most common peak month: {seasonality_df['peak_fire_month'].mode().iloc[0] if not seasonality_df['peak_fire_month'].isna().all() else 'N/A'}")
            logger.info(f"  - Most common peak season: {seasonality_df['peak_fire_season'].mode().iloc[0]}")
            logger.info(f"  - Mean fire season length: {seasonality_df['fire_season_length_months'].mean():.1f} months")
        
        return seasonality_df
    
    def calculate_fire_intensity_metrics(self, fire_watershed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fire intensity and duration metrics for each watershed.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            
        Returns:
            pd.DataFrame: Intensity metrics per watershed
        """
        logger.info("Calculating fire intensity metrics")
        
        intensity_metrics = []
        
        for huc12_id in tqdm(fire_watershed_df['huc12'].unique(), desc="Processing intensity"):
            watershed_fires = fire_watershed_df[fire_watershed_df['huc12'] == huc12_id].copy()
            
            metrics = {
                'huc12': huc12_id,
                'n_fires': len(watershed_fires)
            }
            
            if len(watershed_fires) > 0:
                # Fire Radiative Power (FRP) metrics
                if 'mean_frp' in watershed_fires.columns:
                    frp_data = watershed_fires['mean_frp'].dropna()
                    
                    if len(frp_data) > 0:
                        metrics.update({
                            'mean_watershed_frp': frp_data.mean(),
                            'median_watershed_frp': frp_data.median(),
                            'max_watershed_frp': frp_data.max(),
                            'std_watershed_frp': frp_data.std(),
                            'cv_watershed_frp': frp_data.std() / frp_data.mean() if frp_data.mean() > 0 else 0,
                            'frp_data_available': True
                        })
                    else:
                        metrics.update({
                            'mean_watershed_frp': np.nan,
                            'median_watershed_frp': np.nan,
                            'max_watershed_frp': np.nan,
                            'std_watershed_frp': np.nan,
                            'cv_watershed_frp': np.nan,
                            'frp_data_available': False
                        })
                else:
                    metrics.update({
                        'mean_watershed_frp': np.nan,
                        'median_watershed_frp': np.nan,
                        'max_watershed_frp': np.nan,
                        'std_watershed_frp': np.nan,
                        'cv_watershed_frp': np.nan,
                        'frp_data_available': False
                    })
                
                # Fire duration metrics
                if 'duration_days' in watershed_fires.columns:
                    duration_data = watershed_fires['duration_days'].dropna()
                    
                    if len(duration_data) > 0:
                        metrics.update({
                            'mean_fire_duration_days': duration_data.mean(),
                            'median_fire_duration_days': duration_data.median(),
                            'max_fire_duration_days': duration_data.max(),
                            'std_fire_duration_days': duration_data.std(),
                            'single_day_fires_fraction': (duration_data == 1).sum() / len(duration_data),
                            'long_duration_fires_fraction': (duration_data > 7).sum() / len(duration_data)  # >1 week
                        })
                    else:
                        metrics.update({
                            'mean_fire_duration_days': np.nan,
                            'median_fire_duration_days': np.nan,
                            'max_fire_duration_days': np.nan,
                            'std_fire_duration_days': np.nan,
                            'single_day_fires_fraction': np.nan,
                            'long_duration_fires_fraction': np.nan
                        })
                else:
                    # Calculate from start/end dates
                    watershed_fires['calc_duration'] = (watershed_fires['end_date'] - watershed_fires['start_date']).dt.days + 1
                    duration_data = watershed_fires['calc_duration']
                    
                    metrics.update({
                        'mean_fire_duration_days': duration_data.mean(),
                        'median_fire_duration_days': duration_data.median(),
                        'max_fire_duration_days': duration_data.max(),
                        'std_fire_duration_days': duration_data.std(),
                        'single_day_fires_fraction': (duration_data == 1).sum() / len(duration_data),
                        'long_duration_fires_fraction': (duration_data > 7).sum() / len(duration_data)
                    })
                
                # Confidence metrics (if available)
                if 'mean_confidence' in watershed_fires.columns:
                    confidence_data = watershed_fires['mean_confidence'].dropna()
                    
                    if len(confidence_data) > 0:
                        metrics.update({
                            'mean_fire_confidence': confidence_data.mean(),
                            'min_fire_confidence': confidence_data.min(),
                            'std_fire_confidence': confidence_data.std()
                        })
                    else:
                        metrics.update({
                            'mean_fire_confidence': np.nan,
                            'min_fire_confidence': np.nan,
                            'std_fire_confidence': np.nan
                        })
                else:
                    metrics.update({
                        'mean_fire_confidence': np.nan,
                        'min_fire_confidence': np.nan,
                        'std_fire_confidence': np.nan
                    })
                
                # Fire detection density
                if 'n_detections' in watershed_fires.columns:
                    detection_data = watershed_fires['n_detections'].dropna()
                    
                    if len(detection_data) > 0:
                        metrics.update({
                            'total_fire_detections': detection_data.sum(),
                            'mean_detections_per_fire': detection_data.mean(),
                            'max_detections_per_fire': detection_data.max()
                        })
                    else:
                        metrics.update({
                            'total_fire_detections': 0,
                            'mean_detections_per_fire': 0,
                            'max_detections_per_fire': 0
                        })
                else:
                    metrics.update({
                        'total_fire_detections': len(watershed_fires),  # Assume 1 detection per fire
                        'mean_detections_per_fire': 1,
                        'max_detections_per_fire': 1
                    })
                
            else:
                # No fires - set all metrics to appropriate defaults
                metrics.update({
                    'mean_watershed_frp': np.nan,
                    'median_watershed_frp': np.nan,
                    'max_watershed_frp': np.nan,
                    'std_watershed_frp': np.nan,
                    'cv_watershed_frp': np.nan,
                    'frp_data_available': False,
                    'mean_fire_duration_days': np.nan,
                    'median_fire_duration_days': np.nan,
                    'max_fire_duration_days': np.nan,
                    'std_fire_duration_days': np.nan,
                    'single_day_fires_fraction': np.nan,
                    'long_duration_fires_fraction': np.nan,
                    'mean_fire_confidence': np.nan,
                    'min_fire_confidence': np.nan,
                    'std_fire_confidence': np.nan,
                    'total_fire_detections': 0,
                    'mean_detections_per_fire': 0,
                    'max_detections_per_fire': 0
                })
            
            intensity_metrics.append(metrics)
        
        intensity_df = pd.DataFrame(intensity_metrics)
        
        logger.info(f"Fire intensity statistics:")
        if len(intensity_df) > 0:
            logger.info(f"  - Mean fire duration: {intensity_df['mean_fire_duration_days'].mean():.1f} days")
            logger.info(f"  - Single-day fires (mean): {intensity_df['single_day_fires_fraction'].mean():.2f}")
            if not intensity_df['mean_watershed_frp'].isna().all():
                logger.info(f"  - Mean FRP: {intensity_df['mean_watershed_frp'].mean():.1f} MW")
        
        return intensity_df
    
    def calculate_composite_indices(self, fire_watershed_df: pd.DataFrame,
                                  fri_df: pd.DataFrame,
                                  burn_df: pd.DataFrame,
                                  seasonality_df: pd.DataFrame,
                                  intensity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite fire regime indices.
        
        Args:
            fire_watershed_df: Fire-watershed intersection data
            fri_df: Fire return interval metrics
            burn_df: Burn fraction metrics
            seasonality_df: Seasonality metrics
            intensity_df: Intensity metrics
            
        Returns:
            pd.DataFrame: Composite fire regime indices
        """
        logger.info("Calculating composite fire regime indices")
        
        # Merge all metrics
        all_metrics = fri_df.copy()
        
        for df in [burn_df, seasonality_df, intensity_df]:
            all_metrics = all_metrics.merge(df, on='huc12', how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            all_metrics = all_metrics.loc[:, ~all_metrics.columns.str.endswith('_dup')]
        
        composite_metrics = []
        
        for _, row in tqdm(all_metrics.iterrows(), total=len(all_metrics), desc="Computing indices"):
            huc12_id = row['huc12']
            
            metrics = {'huc12': huc12_id}
            
            # Fire Activity Index (0-1, higher = more fire activity)
            fire_freq = row.get('fire_frequency_per_year', 0)
            burned_frac = row.get('burned_fraction_estimate', 0)
            
            # Normalize by study area characteristics (simple approach)
            fire_activity_index = min((fire_freq * 10 + burned_frac) / 2, 1.0)
            
            metrics['fire_activity_index'] = fire_activity_index
            
            # Fire Regime Stability Index (0-1, higher = more predictable)
            fri_cv = row.get('cv_fire_return_interval', np.nan)
            seasonality_concentration = row.get('fire_concentration_peak_season', 0)
            
            if not np.isnan(fri_cv):
                # Lower CV = more stable
                stability_component1 = max(0, 1 - fri_cv)
            else:
                stability_component1 = 0.5  # Default for single/no fires
            
            # Higher seasonal concentration = more predictable
            stability_component2 = seasonality_concentration
            
            fire_stability_index = (stability_component1 + stability_component2) / 2
            metrics['fire_regime_stability_index'] = fire_stability_index
            
            # Fire Intensity Index (0-1, higher = more intense fires)
            mean_frp = row.get('mean_watershed_frp', np.nan)
            mean_duration = row.get('mean_fire_duration_days', np.nan)
            
            # Normalize FRP (typical range 0-500 MW)
            if not np.isnan(mean_frp):
                frp_component = min(mean_frp / 200, 1.0)
            else:
                frp_component = 0.5  # Default
            
            # Normalize duration (typical range 1-30 days)
            if not np.isnan(mean_duration):
                duration_component = min((mean_duration - 1) / 29, 1.0)
            else:
                duration_component = 0.5  # Default
            
            fire_intensity_index = (frp_component + duration_component) / 2
            metrics['fire_intensity_index'] = fire_intensity_index
            
            # Fire Regime Type Classification (categorical)
            # Based on frequency and burn patterns
            if fire_freq > 0.5:  # >0.5 fires per year
                if seasonality_concentration > 0.7:
                    regime_type = "High_Frequency_Seasonal"
                else:
                    regime_type = "High_Frequency_Variable"
            elif fire_freq > 0.1:  # 0.1-0.5 fires per year
                if burned_frac > 0.1:
                    regime_type = "Moderate_Frequency_High_Impact"
                else:
                    regime_type = "Moderate_Frequency_Low_Impact"
            elif fire_freq > 0:  # Some fires but infrequent
                regime_type = "Low_Frequency"
            else:
                regime_type = "Fire_Suppressed"
            
            metrics['fire_regime_type'] = regime_type
            
            # Fire Risk Category (for management applications)
            if fire_activity_index > 0.7:
                risk_category = "High"
            elif fire_activity_index > 0.3:
                risk_category = "Moderate"
            elif fire_activity_index > 0.1:
                risk_category = "Low"
            else:
                risk_category = "Minimal"
            
            metrics['fire_risk_category'] = risk_category
            
            composite_metrics.append(metrics)
        
        composite_df = pd.DataFrame(composite_metrics)
        
        logger.info(f"Composite fire regime statistics:")
        if len(composite_df) > 0:
            logger.info(f"  - Mean fire activity index: {composite_df['fire_activity_index'].mean():.3f}")
            logger.info(f"  - Mean stability index: {composite_df['fire_regime_stability_index'].mean():.3f}")
            logger.info(f"  - Fire regime types: {composite_df['fire_regime_type'].value_counts().to_dict()}")
        
        return composite_df
    
    def calculate_all_watershed_metrics(self, watershed_file: Union[str, Path],
                                      fire_events_file: Union[str, Path],
                                      buffer_km: float = 0.0) -> pd.DataFrame:
        """
        Calculate all fire metrics for watersheds (main workflow).
        
        Args:
            watershed_file: Path to watershed boundaries file
            fire_events_file: Path to fire events file
            buffer_km: Buffer around fire centroids in kilometers
            
        Returns:
            pd.DataFrame: Complete fire metrics for all watersheds
        """
        logger.info("Starting complete watershed fire metrics calculation")
        
        # Load data
        self.load_watershed_boundaries(watershed_file)
        self.load_fire_events(fire_events_file)
        
        # Intersect fires with watersheds
        fire_watershed_df = self.intersect_fires_with_watersheds(buffer_km)
        
        # Calculate all metric categories
        fri_df = self.calculate_fire_return_intervals(fire_watershed_df)
        burn_df = self.calculate_burn_fractions(fire_watershed_df)
        seasonality_df = self.calculate_fire_seasonality(fire_watershed_df)
        intensity_df = self.calculate_fire_intensity_metrics(fire_watershed_df)
        
        # Calculate composite indices
        composite_df = self.calculate_composite_indices(
            fire_watershed_df, fri_df, burn_df, seasonality_df, intensity_df
        )
        
        # Merge all metrics into final dataset
        final_metrics = fri_df.copy()
        
        for df in [burn_df, seasonality_df, intensity_df, composite_df]:
            final_metrics = final_metrics.merge(df, on='huc12', how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            final_metrics = final_metrics.loc[:, ~final_metrics.columns.str.endswith('_dup')]
        
        # Add watersheds without any fires
        all_huc12s = set(self.watersheds['huc12'])
        watersheds_with_fires = set(final_metrics['huc12'])
        watersheds_without_fires = all_huc12s - watersheds_with_fires
        
        if watersheds_without_fires:
            logger.info(f"Adding {len(watersheds_without_fires)} watersheds with no fires")
            
            no_fire_rows = []
            for huc12_id in watersheds_without_fires:
                watershed_area = self.watersheds[self.watersheds['huc12'] == huc12_id]['area_km2'].iloc[0]
                
                no_fire_row = {
                    'huc12': huc12_id,
                    'watershed_area_km2': watershed_area,
                    'n_fires': 0,
                    'fire_frequency_per_year': 0,
                    'burned_fraction_estimate': 0,
                    'fire_activity_index': 0,
                    'fire_regime_stability_index': np.nan,
                    'fire_intensity_index': np.nan,
                    'fire_regime_type': 'Fire_Suppressed',
                    'fire_risk_category': 'Minimal'
                }
                no_fire_rows.append(no_fire_row)
            
            no_fire_df = pd.DataFrame(no_fire_rows)
            final_metrics = pd.concat([final_metrics, no_fire_df], ignore_index=True, sort=False)
        
        # Cache the final result
        self.watershed_metrics = final_metrics.copy()
        
        logger.info(f"Final watershed metrics:")
        logger.info(f"  - Total watersheds: {len(final_metrics)}")
        logger.info(f"  - Watersheds with fires: {(final_metrics['n_fires'] > 0).sum()}")
        logger.info(f"  - Watersheds without fires: {(final_metrics['n_fires'] == 0).sum()}")
        logger.info(f"  - Total metrics per watershed: {len(final_metrics.columns)}")
        
        return final_metrics
    
    def export_metrics(self, metrics_df: pd.DataFrame,
                      output_path: Optional[Union[str, Path]] = None,
                      format: str = 'csv') -> Path:
        """
        Export watershed fire metrics to file.
        
        Args:
            metrics_df: Watershed fire metrics DataFrame
            output_path: Output file path (auto-generated if None)
            format: Export format ('csv', 'json', 'parquet')
            
        Returns:
            Path: Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"watershed_fire_metrics_{timestamp}.{format}"
            output_path = PROCESSED_DATA_DIR / filename
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(metrics_df)} watershed fire metrics to {output_path}")
        
        try:
            if format.lower() == 'csv':
                metrics_df.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                # Handle datetime and NaN values for JSON export
                metrics_df_json = metrics_df.copy()
                metrics_df_json = metrics_df_json.fillna('null')  # Replace NaN with string for JSON
                metrics_df_json.to_json(output_path, orient='records', indent=2)
            elif format.lower() == 'parquet':
                metrics_df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Export completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of calculated metrics.
        
        Returns:
            Dict: Summary statistics
        """
        if self.watershed_metrics is None:
            return {"error": "No metrics calculated yet"}
        
        df = self.watershed_metrics
        
        summary = {
            'total_watersheds': len(df),
            'watersheds_with_fires': int((df['n_fires'] > 0).sum()),
            'watersheds_without_fires': int((df['n_fires'] == 0).sum()),
            'study_period_years': self.study_period_years,
            'fire_frequency_stats': {
                'mean': float(df['fire_frequency_per_year'].mean()),
                'median': float(df['fire_frequency_per_year'].median()),
                'max': float(df['fire_frequency_per_year'].max()),
                'min': float(df['fire_frequency_per_year'].min())
            },
            'burned_fraction_stats': {
                'mean': float(df['burned_fraction_estimate'].mean()),
                'median': float(df['burned_fraction_estimate'].median()),
                'max': float(df['burned_fraction_estimate'].max()),
                'watersheds_over_10pct': int((df['burned_fraction_estimate'] > 0.1).sum())
            },
            'fire_regime_types': df['fire_regime_type'].value_counts().to_dict(),
            'fire_risk_categories': df['fire_risk_category'].value_counts().to_dict(),
            'metrics_available': list(df.columns)
        }
        
        return summary