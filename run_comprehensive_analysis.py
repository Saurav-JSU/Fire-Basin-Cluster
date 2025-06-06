#!/usr/bin/env python
"""
Comprehensive Wildfire Watershed Analysis - Complete Pipeline

This script runs the complete analysis workflow from Steps 1-3:
1. Load HUC12 watersheds and FIRMS data from Google Earth Engine
2. Process FIRMS data into fire events using spatial-temporal clustering
3. Calculate comprehensive fire metrics for each watershed

Uses REAL data from Google Earth Engine - no simulations.
Designed for the Western United States study area.
"""
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Optional, Tuple
import geopandas as gpd

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set up comprehensive logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveWildfireAnalysis:
    """
    Master class to run the complete wildfire watershed analysis pipeline.
    
    Orchestrates Steps 1-3:
    - Google Earth Engine data loading
    - FIRMS fire event preprocessing  
    - Watershed fire metrics calculation
    """
    
    def __init__(self, project_id: str = None, study_area: str = "western_us", 
                 start_date: str = None, end_date: str = None, max_watersheds: int = None,
                 batch_size: Optional[int] = None, chunk_size: str = 'monthly'):
        """
        Initialize the Comprehensive Wildfire Analysis.
        
        Args:
            project_id: Google Earth Engine project ID
            study_area: Study area identifier 
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            max_watersheds: Maximum number of watersheds to process
            batch_size: Number of watersheds to process in each batch
            chunk_size: Temporal chunk size for FIRMS data processing ('monthly' or 'yearly')
        """
        # Configuration
        self.project_id = project_id or PROJECT_CONFIG["project_id"]
        self.study_area = study_area
        self.start_date = start_date or ANALYSIS_CONFIG["start_date"]
        self.end_date = end_date or ANALYSIS_CONFIG["end_date"]
        self.max_watersheds = max_watersheds
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # Create output directories
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.data_dir / "results"
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # ✅ Initialize data storage attributes
        self.gee_loader = None
        self.firms_preprocessor = None
        
        # Data containers
        self.huc12_data = None
        self.firms_data = None          # ✅ Critical: Initialize this attribute
        self.fire_events = None
        self.watershed_metrics = None
        self.clustering_results = None
        
        # Analysis results
        self.dataset_info = None
        self.processing_summary = None
        
        logger.info("Comprehensive Analysis initialized:")
        logger.info(f"  - Project ID: {self.project_id}")
        logger.info(f"  - Study period: {self.start_date} to {self.end_date}")
        logger.info(f"  - Max watersheds: {self.max_watersheds or 'All'}")
    
    def step1_load_data(self) -> Tuple[bool, str]:
        """
        Step 1: Load data from Google Earth Engine.
        
        Returns:
            Tuple[bool, str]: (success, status_message)
        """
        logger.info("="*60)
        logger.info("STEP 1: Loading Data from Google Earth Engine")
        logger.info("="*60)
        
        try:
            # Initialize GEE Data Loader
            from src.data.gee_loader import GEEDataLoader
            
            # Initialize with only the project_id parameter
            self.gee_loader = GEEDataLoader(project_id=self.project_id)
            
            # Authenticate with Google Earth Engine (needed for FIRMS data)
            logger.info("Authenticating with Google Earth Engine...")
            if not self.gee_loader.authenticate():
                return False, "GEE authentication failed"
            logger.info("✅ GEE authentication successful")
            
            # Check for local HUC12 file first
            local_huc12_file = self.raw_dir / "huc12_western_us.geojson"
            if local_huc12_file.exists():
                logger.info(f"Found local HUC12 file: {local_huc12_file}")
                import geopandas as gpd
                self.huc12_data = gpd.read_file(local_huc12_file)
                logger.info(f"✅ Loaded {len(self.huc12_data)} HUC12 watersheds from local file")
            else:
                # Load HUC12 watershed boundaries from GEE
                logger.info("Loading HUC12 watershed boundaries for Western US...")
                self.huc12_data = self.gee_loader.load_huc12_watersheds()
                logger.info("✅ Loaded HUC12 watersheds")
            
            # Load FIRMS fire data
            logger.info(f"Loading FIRMS fire data ({self.start_date} to {self.end_date})...")
            firms_data = self.gee_loader.load_firms_data(
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # ✅ Store FIRMS data as instance attribute
            self.firms_data = firms_data
            
            logger.info("✅ Loaded FIRMS fire data")
            
            # Get dataset info for logging (if method exists)
            try:
                info = self.gee_loader.get_dataset_info()
                logger.info("📊 Dataset Summary:")
                logger.info(f"  - HUC12 watersheds: {info['datasets']['huc12']['count']}")
                logger.info(f"  - FIRMS images: {info['datasets']['firms']['count']}")
                self.dataset_info = info
            except AttributeError:
                # If get_dataset_info method doesn't exist, get sizes manually
                logger.info("📊 Dataset Summary:")
                
                huc12_count = "Unknown"
                if self.huc12_data is not None:
                    try:
                        huc12_count = len(self.huc12_data) if isinstance(self.huc12_data, gpd.GeoDataFrame) else self.huc12_data.size().getInfo()
                    except:
                        huc12_count = "Available"
                
                firms_count = "Unknown" 
                if firms_data is not None:
                    try:
                        firms_count = firms_data.size().getInfo()
                    except:
                        firms_count = "Available"
                        
                logger.info(f"  - HUC12 watersheds: {huc12_count}")
                logger.info(f"  - FIRMS images: {firms_count}")
                
                # Store basic info
                self.dataset_info = {
                    'datasets': {
                        'huc12': {'count': huc12_count},
                        'firms': {'count': firms_count}
                    }
                }
            
            logger.info("✅ Step 1 completed successfully")
            return True, "Step 1 completed successfully"
            
        except Exception as e:
            logger.error(f"❌ Step 1 failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False, f"Step 1 error: {e}"
    
    def step2_process_fire_events(self) -> Tuple[bool, str]:
        """
        Step 2: Process FIRMS data into fire events (respects scale parameter).
        
        Returns:
            Tuple[bool, str]: (success, status_message)
        """
        logger.info("="*60)
        logger.info("STEP 2: Processing FIRMS Data into Fire Events")
        logger.info("="*60)
        
        try:
            # Import and initialize FIRMS preprocessor
            from src.data.preprocessor import FIRMSPreprocessor
            
            self.firms_preprocessor = FIRMSPreprocessor(
                chunk_size=self.chunk_size
            )
            
            logger.info("✅ FIRMS preprocessor initialized")
            
            # Verify we have the required data
            if not hasattr(self, 'firms_data') or self.firms_data is None:
                logger.error("❌ No FIRMS data available")
                return False, "No FIRMS data available for processing"
            
            if not hasattr(self, 'huc12_data') or self.huc12_data is None:
                logger.error("❌ No HUC12 data available")
                return False, "No HUC12 data available for processing"
            
            # Get FIRMS data size for verification
            try:
                firms_size = self.firms_data.size().getInfo()
                logger.info(f"📊 FIRMS data available: {firms_size} images")
                
                if firms_size == 0:
                    logger.warning("⚠️ FIRMS collection is empty")
                    return True, "Step 2 completed - empty FIRMS collection"
                    
            except Exception as e:
                logger.error(f"❌ Error checking FIRMS data size: {e}")
                return False, f"Error accessing FIRMS data: {e}"
            
            # ✅ FIX: Determine number of watersheds based on scale parameter
            import geopandas as gpd
            import pandas as pd
            import ee
            from tqdm import tqdm
            
            # Load all HUC12 watersheds
            if isinstance(self.huc12_data, gpd.GeoDataFrame):
                # Already loaded as GeoDataFrame from cache
                all_watersheds = self.huc12_data
            else:
                # Convert from Earth Engine FeatureCollection to GeoDataFrame
                logger.info("Converting HUC12 data to GeoDataFrame...")
                import geemap
                all_watersheds = geemap.ee_to_gdf(self.huc12_data)
            
            total_available = len(all_watersheds)
            logger.info(f"📊 Total watersheds available: {total_available:,}")
            
            # ✅ Apply scale-based limits
            if self.max_watersheds is not None:
                if self.max_watersheds < total_available:
                    watersheds_to_process = all_watersheds.head(self.max_watersheds)
                    logger.info(f"🎯 Scale limit: Processing {self.max_watersheds:,} watersheds (out of {total_available:,})")
                else:
                    watersheds_to_process = all_watersheds
                    logger.info(f"🎯 Scale limit higher than available: Processing all {total_available:,} watersheds")
            else:
                watersheds_to_process = all_watersheds
                logger.info(f"🎯 No scale limit: Processing all {total_available:,} watersheds")
            
            num_to_process = len(watersheds_to_process)
            
            # ✅ Scale-aware batch size
            if num_to_process <= 10:
                # Small scale: process individually for better logging
                batch_size = 1
                logger.info(f"🔍 Small scale analysis: Processing {num_to_process} watersheds individually")
            elif num_to_process <= 100:
                # Medium scale: small batches
                batch_size = 10
                logger.info(f"🔍 Medium scale analysis: Processing {num_to_process} watersheds in batches of {batch_size}")
            else:
                # Large scale: larger batches
                batch_size = 100
                logger.info(f"🔍 Large scale analysis: Processing {num_to_process} watersheds in batches of {batch_size}")
            
            total_batches = (num_to_process + batch_size - 1) // batch_size
            logger.info(f"📦 Will process in {total_batches} batches")
            
            all_fire_detections = []
            successful_extractions = 0
            failed_extractions = 0
            
            # Process watersheds in batches
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, num_to_process)
                batch_watersheds = watersheds_to_process.iloc[start_idx:end_idx]
                
                logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} (watersheds {start_idx}-{end_idx-1})")
                
                batch_detections = []
                
                # Process each watershed in the batch
                progress_desc = f"Batch {batch_num + 1}/{total_batches}"
                for idx, watershed in tqdm(batch_watersheds.iterrows(), 
                                         total=len(batch_watersheds), 
                                         desc=progress_desc):
                    
                    huc12_id = watershed['huc12']
                    
                    # ✅ For small scale, log each watershed individually
                    if num_to_process <= 10:
                        logger.info(f"  🔍 Processing watershed: {huc12_id}")
                    
                    try:
                        # Get watershed geometry as ee.Geometry
                        if hasattr(watershed.geometry, 'exterior'):
                            # Convert Shapely polygon to Earth Engine geometry
                            watershed_geom = ee.Geometry.Polygon(
                                [[list(coord) for coord in watershed.geometry.exterior.coords]]
                            )
                        else:
                            logger.warning(f"⚠️ Invalid geometry for {huc12_id}, skipping")
                            failed_extractions += 1
                            continue
                        
                        # Extract fire points for this watershed
                        watershed_fires = self.firms_preprocessor.extract_firms_points_from_gee(
                            self.firms_data,
                            watershed_geom,
                            max_pixels=50000  # Reasonable limit per watershed
                        )
                        
                        if len(watershed_fires) > 0:
                            watershed_fires['source_huc12'] = huc12_id
                            batch_detections.append(watershed_fires)
                            successful_extractions += 1
                            
                            # Log for small scale
                            if num_to_process <= 10:
                                logger.info(f"    ✅ Found {len(watershed_fires)} fire detections")
                            # Log progress periodically for larger scales
                            elif successful_extractions % 50 == 0:
                                logger.info(f"  🔥 Progress: {successful_extractions} watersheds with fire detections found")
                        else:
                            if num_to_process <= 10:
                                logger.info(f"    ℹ️ No fire detections found")
                        
                    except Exception as e:
                        failed_extractions += 1
                        if num_to_process <= 10:
                            logger.warning(f"    ⚠️ Error processing {huc12_id}: {e}")
                        elif failed_extractions % 100 == 0:  # Log every 100 failures for large scale
                            logger.warning(f"  ⚠️ {failed_extractions} watersheds failed processing (latest: {huc12_id}: {e})")
                        continue
                
                # Add batch detections to total
                if batch_detections:
                    batch_combined = pd.concat(batch_detections, ignore_index=True)
                    all_fire_detections.append(batch_combined)
                    logger.info(f"  ✅ Batch {batch_num + 1} completed: {len(batch_detections)} watersheds with fires, {len(batch_combined):,} total detections")
                else:
                    logger.info(f"  ℹ️ Batch {batch_num + 1} completed: No fire detections found")
            
            # Final processing summary
            logger.info(f"📊 Fire Extraction Summary:")
            logger.info(f"  - Total watersheds processed: {num_to_process:,}")
            logger.info(f"  - Watersheds with fire detections: {successful_extractions:,}")
            logger.info(f"  - Watersheds with no fires: {num_to_process - successful_extractions - failed_extractions:,}")
            logger.info(f"  - Failed extractions: {failed_extractions:,}")
            
            if num_to_process > 0:
                logger.info(f"  - Success rate: {(successful_extractions / num_to_process * 100):.1f}%")
            
            if all_fire_detections:
                # Combine all detections
                firms_detections = pd.concat(all_fire_detections, ignore_index=True)
                
                logger.info(f"🔥 Total fire detections extracted: {len(firms_detections):,}")
                
                # Save raw detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                scale_suffix = f"_scale{num_to_process}" if num_to_process < total_available else "_full"
                detections_file = self.processed_dir / f"fire_detections{scale_suffix}_{timestamp}.csv"
                firms_detections.to_csv(detections_file, index=False)
                logger.info(f"💾 Raw fire detections saved to: {detections_file}")
                
                # Identify fire events using spatial-temporal clustering
                logger.info("🔄 Identifying fire events using spatial-temporal clustering...")
                
                self.fire_events = self.firms_preprocessor.identify_fire_events(firms_detections)
                
                logger.info(f"✅ Fire events identified: {len(self.fire_events):,} events")
                
                # Export fire events
                if len(self.fire_events) > 0:
                    events_file = self.processed_dir / f"fire_events{scale_suffix}_{timestamp}.csv"
                    self.fire_events.to_csv(events_file, index=False)
                    logger.info(f"💾 Fire events exported to: {events_file}")
                    
                    # Get processing summary
                    summary = self.firms_preprocessor.get_processing_summary()
                    if 'fire_events' in summary.get('data_summary', {}):
                        events_summary = summary['data_summary']['fire_events']
                        logger.info(f"📊 Event Summary:")
                        logger.info(f"  - Total events: {events_summary.get('total_events', 0):,}")
                        logger.info(f"  - Mean duration: {events_summary.get('duration_stats_days', {}).get('mean', 'N/A')} days")
                        logger.info(f"  - Single detection events: {events_summary.get('single_detection_events', 0):,}")
                
                return True, f"Step 2 completed successfully - {len(self.fire_events):,} events from {successful_extractions:,} watersheds"
                
            else:
                logger.warning("⚠️ No fire detections found in any watersheds")
                logger.info("Possible reasons:")
                logger.info("  - Time period with low fire activity")
                logger.info("  - Confidence threshold too high (current: 80%)") 
                logger.info("  - Study area with naturally low fire occurrence")
                logger.info("  - Technical issues with fire detection extraction")
                
                # Create empty fire events file for Step 3 compatibility
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                scale_suffix = f"_scale{num_to_process}" if num_to_process < total_available else "_full"
                events_file = self.processed_dir / f"fire_events_empty{scale_suffix}_{timestamp}.csv"
                
                empty_events = pd.DataFrame(columns=[
                    'event_id', 'start_date', 'end_date', 'centroid_lon', 'centroid_lat',
                    'duration_days', 'n_detections', 'spatial_extent_km'
                ])
                empty_events.to_csv(events_file, index=False)
                
                return True, f"Step 2 completed - no fire detections found in {num_to_process} watersheds"
                
        except Exception as e:
            logger.error(f"❌ Step 2 failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False, f"Step 2 error: {e}"
    
    def step3_calculate_fire_metrics(self) -> Tuple[bool, str]:
        """
        Step 3: Calculate comprehensive fire metrics for each watershed.
        
        Returns:
            Tuple[bool, str]: (success, status_message)
        """
        logger.info("="*60)
        logger.info("STEP 3: Calculating Watershed Fire Metrics")
        logger.info("="*60)
        
        try:
            # Import fire metrics calculator
            from src.features.fire_metrics import WatershedFireMetrics
            from src.features.temporal_analysis import TemporalFireAnalyzer
            
            # Calculate study period
            start_year = int(self.start_date[:4])
            end_year = int(self.end_date[:4])
            study_period_years = end_year - start_year + 1
            
            self.fire_metrics_calculator = WatershedFireMetrics(study_period_years=study_period_years)
            self.temporal_analyzer = TemporalFireAnalyzer()
            
            logger.info(f"✅ Fire metrics calculator initialized (study period: {study_period_years} years)")
            
            # Check for required data files
            watershed_files = (list(self.raw_dir.glob("*watershed*.geojson")) + 
                             list(self.raw_dir.glob("*huc12*.geojson")) +
                             list(self.raw_dir.glob("sample_watersheds*.geojson")))
            
            fire_event_files = (list(self.processed_dir.glob("*fire_event*.csv")) + 
                               list(self.processed_dir.glob("*events*.csv")))
            
            logger.info(f"🔍 Looking for data files:")
            logger.info(f"  - Watershed files found: {len(watershed_files)}")
            logger.info(f"  - Fire event files found: {len(fire_event_files)}")
            
            if watershed_files and fire_event_files:
                watershed_file = watershed_files[0]
                fire_events_file = fire_event_files[0]
                
                logger.info(f"📁 Using files:")
                logger.info(f"  - Watersheds: {watershed_file.name}")
                logger.info(f"  - Fire events: {fire_events_file.name}")
                
                # Run complete fire metrics calculation
                logger.info("🔥 Running complete watershed fire metrics calculation...")
                
                self.watershed_fire_metrics = self.fire_metrics_calculator.calculate_all_watershed_metrics(
                    watershed_file=watershed_file,
                    fire_events_file=fire_events_file,
                    buffer_km=1.0  # 1km buffer around fire centroids
                )
                
                logger.info(f"✅ Fire metrics calculation completed!")
                logger.info(f"  - Watersheds processed: {len(self.watershed_fire_metrics)}")
                logger.info(f"  - Metrics per watershed: {len(self.watershed_fire_metrics.columns)}")
                
                # Export results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = self.results_dir / f"watershed_fire_metrics_{timestamp}.csv"
                
                self.fire_metrics_calculator.export_metrics(
                    self.watershed_fire_metrics, 
                    output_path=results_file,
                    format='csv'
                )
                
                logger.info(f"💾 Results exported to: {results_file}")
                
                # Generate comprehensive summary
                summary = self.fire_metrics_calculator.get_metrics_summary()
                
                logger.info(f"📊 Final Results Summary:")
                logger.info(f"  - Total watersheds: {summary['total_watersheds']}")
                logger.info(f"  - Watersheds with fires: {summary['watersheds_with_fires']}")
                logger.info(f"  - Watersheds without fires: {summary['watersheds_without_fires']}")
                logger.info(f"  - Study period: {summary['study_period_years']} years")
                logger.info(f"  - Mean fire frequency: {summary['fire_frequency_stats']['mean']:.3f} fires/year")
                logger.info(f"  - Mean burned fraction: {summary['burned_fraction_stats']['mean']:.3f}")
                logger.info(f"  - Fire regime types: {summary['fire_regime_types']}")
                
                # Advanced temporal analysis
                logger.info("🧠 Running advanced temporal analysis...")
                
                # Get fire-watershed intersection data for temporal analysis
                fire_watershed_intersections = self.fire_metrics_calculator.intersect_fires_with_watersheds(buffer_km=1.0)
                
                if len(fire_watershed_intersections) > 0:
                    temporal_summary = self.temporal_analyzer.create_temporal_summary(fire_watershed_intersections)
                    
                    # Export temporal analysis results
                    temporal_file = self.results_dir / f"temporal_analysis_{timestamp}.csv"
                    temporal_summary.to_csv(temporal_file, index=False)
                    
                    logger.info(f"✅ Temporal analysis completed: {len(temporal_summary)} watersheds")
                    logger.info(f"💾 Temporal results exported to: {temporal_file}")
                    
                    # Temporal summary statistics
                    trend_counts = temporal_summary['trend_direction'].value_counts().to_dict()
                    logger.info(f"📊 Temporal Analysis Summary:")
                    logger.info(f"  - Trend analysis: {trend_counts}")
                    
                    if 'regime_change_detected' in temporal_summary.columns:
                        regime_changes = temporal_summary['regime_change_detected'].sum()
                        logger.info(f"  - Regime changes detected: {regime_changes}")
                else:
                    logger.warning("⚠️ No fire-watershed intersections for temporal analysis")
                
                return True, "Step 3 completed successfully"
                
            else:
                logger.error("❌ Required data files not found")
                logger.error("Expected files:")
                logger.error("  - Watershed boundaries: *watershed*.geojson or *huc12*.geojson")
                logger.error("  - Fire events: *fire_event*.csv or *events*.csv")
                logger.error("These should be outputs from Steps 1 & 2")
                return False, "Required data files not found"
                
        except Exception as e:
            logger.error(f"❌ Step 3 failed: {e}")
            return False, f"Step 3 error: {e}"
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete analysis pipeline (Steps 1-3).
        
        Returns:
            bool: True if all steps completed successfully
        """
        logger.info("🚀 Starting Comprehensive Wildfire Watershed Analysis")
        logger.info("🔥 Western United States - HUC12 Watersheds")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Step 1: Load GEE Data
        step1_success, step1_msg = self.step1_load_data()
        if not step1_success:
            logger.error(f"🛑 Analysis stopped: {step1_msg}")
            return False
        
        logger.info(f"✅ {step1_msg}")
        
        # Step 2: Process Fire Events
        step2_success, step2_msg = self.step2_process_fire_events()
        if not step2_success:
            logger.error(f"🛑 Analysis stopped: {step2_msg}")
            return False
        
        logger.info(f"✅ {step2_msg}")
        
        # Step 3: Calculate Fire Metrics
        step3_success, step3_msg = self.step3_calculate_fire_metrics()
        if not step3_success:
            logger.error(f"🛑 Analysis stopped: {step3_msg}")
            return False
        
        logger.info(f"✅ {step3_msg}")
        
        # Analysis completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("🎉 COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {duration}")
        logger.info(f"📊 Results available in: {self.results_dir}")
        
        # List output files
        result_files = list(self.results_dir.glob("*.csv"))
        if result_files:
            logger.info(f"📁 Output files created:")
            for file in sorted(result_files):
                logger.info(f"  - {file.name}")
        
        logger.info("🚀 Ready for Step 4: Clustering Algorithm Implementation!")
        
        return True
    
    def get_analysis_summary(self) -> dict:
        """Get summary of the completed analysis."""
        summary = {
            'analysis_completed': datetime.now().isoformat(),
            'study_period': {
                'start': self.start_date,
                'end': self.end_date,
                'years': int(self.end_date[:4]) - int(self.start_date[:4]) + 1
            },
            'data_processed': {
                'watersheds': len(self.watershed_fire_metrics) if self.watershed_fire_metrics is not None else 0,
                'fire_events': len(self.fire_events) if self.fire_events is not None else 0
            },
            'output_files': list(self.results_dir.glob("*.csv")),
            'next_step': "Step 4: Clustering Algorithm Implementation"
        }
        
        return summary

def main():
    """Main function to run comprehensive analysis with command line options."""
    parser = argparse.ArgumentParser(description="Comprehensive Wildfire Watershed Analysis")
    
    parser.add_argument(
        "--start-date", 
        default="2000-01-01", 
        help="Start date for analysis (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        default="2025-01-01", 
        help="End date for analysis (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--project-id", 
        default="ee-jsuhydrolabenb", 
        help="Google Earth Engine project ID"
    )
    
    # ✅ Scale configuration options
    parser.add_argument(
        "--scale", 
        choices=['test', 'sample', 'full'], 
        default='test',  # Default to test for safety
        help="Analysis scale: test (10 watersheds), sample (1000 watersheds), full (all ~35K watersheds)"
    )
    parser.add_argument(
        "--max-watersheds", 
        type=int, 
        help="Maximum number of watersheds to process (overrides --scale)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100,
        help="Number of watersheds to process per batch (default: 100)"
    )
    
    # ✅ Add chunk size option
    parser.add_argument(
        "--chunk-size",
        choices=['monthly', 'yearly'],
        default='monthly',
        help="Temporal chunk size for FIRMS data processing (default: monthly)"
    )
    
    parser.add_argument(
        "--step", 
        choices=['1', '2', '3', 'all'], 
        default='all',
        help="Run specific step only (default: all)"
    )
    
    args = parser.parse_args()
    
    # ✅ Determine max_watersheds based on scale
    if args.max_watersheds:
        max_watersheds = args.max_watersheds
        scale_name = f"custom ({max_watersheds:,})"
    elif args.scale == 'test':
        max_watersheds = 10  # ✅ Small test scale
        scale_name = "test (10 watersheds)"
    elif args.scale == 'sample': 
        max_watersheds = 1000  # ✅ Medium sample scale
        scale_name = "sample (1,000 watersheds)"
    elif args.scale == 'full':
        max_watersheds = None  # ✅ Process all watersheds
        scale_name = "full (all watersheds)"
    else:
        max_watersheds = 10  # Default fallback
        scale_name = "default test (10 watersheds)"
    
    print("🎯 Wildfire Watershed Clustering Analysis")
    print(f"📊 Analysis Scale: {scale_name}")
    print(f"📅 Date Range: {args.start_date} to {args.end_date}")
    print(f"🔬 Project ID: {args.project_id}")
    
    if args.scale == 'full':
        print("⚠️  Full scale processing will take several hours and consume significant GEE quota")
        print("💡 Consider starting with --scale test or --scale sample for initial testing")
        
        # Optional: Add confirmation for full scale
        response = input("Continue with full scale analysis? (y/N): ")
        if response.lower() != 'y':
            print("Analysis cancelled. Use --scale test or --scale sample for smaller runs.")
            sys.exit(0)
    elif args.scale == 'test':
        print("🧪 Test scale: Quick analysis with 10 watersheds for code validation")
    elif args.scale == 'sample':
        print("🔬 Sample scale: Medium analysis with 1,000 watersheds for method development")
    
    print("=" * 60)
    
    # Initialize analysis with scale parameters
    analysis = ComprehensiveWildfireAnalysis(
        project_id=args.project_id,
        start_date=args.start_date,
        end_date=args.end_date,
        max_watersheds=max_watersheds,  # ✅ Pass scale parameter
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    # Store additional parameters
    analysis.scale_name = scale_name
    
    success = False
    
    try:
        if args.step == 'all':
            # Run complete analysis
            success = analysis.run_complete_analysis()
        elif args.step == '1':
            success, msg = analysis.step1_load_data()
            logger.info(msg)
        elif args.step == '2':
            success, msg = analysis.step2_process_fire_events()
            logger.info(msg)
        elif args.step == '3':
            success, msg = analysis.step3_calculate_fire_metrics()
            logger.info(msg)
        
        if success:
            # Print final summary
            summary = analysis.get_analysis_summary()
            print("\n" + "="*60)
            print("📋 ANALYSIS SUMMARY")
            print("="*60)
            print(f"Scale: {scale_name}")
            print(f"Study Period: {summary['study_period']['start']} to {summary['study_period']['end']}")
            print(f"Duration: {summary['study_period']['years']} years")
            print(f"Watersheds Processed: {summary['data_processed']['watersheds']}")
            print(f"Fire Events Identified: {summary['data_processed']['fire_events']}")
            print(f"Output Files: {len(summary['output_files'])}")
            print(f"Next Step: {summary['next_step']}")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Analysis interrupted by user")
        success = False
    except Exception as e:
        logger.error(f"❌ Analysis failed with unexpected error: {e}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()