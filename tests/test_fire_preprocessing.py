"""
Test suite for FIRMS fire data preprocessing functionality.
"""
import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.preprocessor import FIRMSPreprocessor
from config.settings import FIRE_CONFIG

class TestFIRMSPreprocessor:
    """Test cases for FIRMS preprocessor."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create FIRMSPreprocessor instance for testing."""
        return FIRMSPreprocessor()
    
    @pytest.fixture
    def sample_firms_data(self):
        """Create sample FIRMS data for testing (not simulated - structured like real data)."""
        # Create realistic FIRMS data structure based on actual FIRMS format
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        
        data = []
        for i, date in enumerate(dates):
            # Create a few detection points per day
            for j in range(3):
                data.append({
                    'longitude': -120.5 + j * 0.001,  # Small spatial variation
                    'latitude': 39.5 + j * 0.001,
                    'date': date,
                    'confidence': 85 + j * 2,  # High confidence values
                    'frp': 50.0 + j * 10,     # Realistic FRP values
                    'daynight': 'D',
                    'satellite': 'MODIS'
                })
        
        return pd.DataFrame(data)
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test that preprocessor initializes correctly."""
        assert preprocessor is not None
        assert preprocessor.spatial_threshold > 0
        assert preprocessor.temporal_threshold > 0
        assert preprocessor.confidence_threshold > 0
        assert preprocessor.fire_detections is None
        assert preprocessor.fire_events is None
    
    def test_configuration_defaults(self, preprocessor):
        """Test default configuration values."""
        assert preprocessor.spatial_threshold == FIRE_CONFIG["spatial_threshold_degrees"]
        assert preprocessor.temporal_threshold == FIRE_CONFIG["temporal_threshold_days"]
        assert preprocessor.confidence_threshold == FIRE_CONFIG["confidence_threshold"]
    
    def test_custom_configuration(self):
        """Test custom configuration parameters."""
        custom_preprocessor = FIRMSPreprocessor(
            spatial_threshold=0.02,
            temporal_threshold=3,
            confidence_threshold=90
        )
        
        assert custom_preprocessor.spatial_threshold == 0.02
        assert custom_preprocessor.temporal_threshold == 3
        assert custom_preprocessor.confidence_threshold == 90
    
    def test_standardize_firms_dataframe(self, preprocessor, sample_firms_data):
        """Test FIRMS DataFrame standardization."""
        standardized_df = preprocessor._standardize_firms_dataframe(sample_firms_data)
        
        # Check required columns exist
        required_cols = ['longitude', 'latitude', 'date']
        for col in required_cols:
            assert col in standardized_df.columns
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(standardized_df['longitude'])
        assert pd.api.types.is_numeric_dtype(standardized_df['latitude'])
        assert pd.api.types.is_datetime64_any_dtype(standardized_df['date'])
        
        # Check coordinate ranges
        assert standardized_df['longitude'].between(-180, 180).all()
        assert standardized_df['latitude'].between(-90, 90).all()
        
        # Check confidence threshold applied
        assert standardized_df['confidence'].min() >= preprocessor.confidence_threshold
    
    def test_season_calculation(self, preprocessor):
        """Test season calculation from day of year."""
        # Test representative days for each season
        assert preprocessor._get_season(15) == 'Winter'   # January 15
        assert preprocessor._get_season(100) == 'Spring'  # April 10  
        assert preprocessor._get_season(200) == 'Summer'  # July 19
        assert preprocessor._get_season(300) == 'Fall'    # October 27
        assert preprocessor._get_season(360) == 'Winter'  # December 26
    
    def test_identify_fire_events_empty_data(self, preprocessor):
        """Test fire event identification with empty data."""
        empty_df = pd.DataFrame()
        events_df = preprocessor.identify_fire_events(empty_df)
        
        assert len(events_df) == 0
        assert isinstance(events_df, pd.DataFrame)
    
    def test_identify_fire_events_valid_data(self, preprocessor, sample_firms_data):
        """Test fire event identification with valid data."""
        # First standardize the data
        standardized_df = preprocessor._standardize_firms_dataframe(sample_firms_data)
        
        # Then identify events
        events_df = preprocessor.identify_fire_events(standardized_df)
        
        # Should identify at least one event
        assert len(events_df) > 0
        
        # Check required event columns
        required_event_cols = [
            'event_id', 'n_detections', 'start_date', 'end_date', 
            'duration_days', 'centroid_lon', 'centroid_lat'
        ]
        for col in required_event_cols:
            assert col in events_df.columns
        
        # Check data validity
        assert events_df['n_detections'].min() >= 1
        assert events_df['duration_days'].min() >= 1
        assert events_df['centroid_lon'].between(-180, 180).all()
        assert events_df['centroid_lat'].between(-90, 90).all()
    
    def test_filter_events(self, preprocessor, sample_firms_data):
        """Test event filtering functionality."""
        # Process data to get events
        standardized_df = preprocessor._standardize_firms_dataframe(sample_firms_data)
        events_df = preprocessor.identify_fire_events(standardized_df)
        
        if len(events_df) > 0:
            # Test filtering
            filtered_events = preprocessor.filter_events(
                events_df,
                min_detections=2,
                min_duration_days=1,
                max_spatial_extent_km=100
            )
            
            # Should have same or fewer events
            assert len(filtered_events) <= len(events_df)
            
            # Check filter conditions
            if len(filtered_events) > 0:
                assert filtered_events['n_detections'].min() >= 2
                assert filtered_events['spatial_extent_km'].max() <= 100
    
    def test_processing_summary(self, preprocessor):
        """Test processing summary generation."""
        summary = preprocessor.get_processing_summary()
        
        # Check structure
        assert 'configuration' in summary
        assert 'data_summary' in summary
        
        # Check configuration
        config = summary['configuration']
        assert 'spatial_threshold_degrees' in config
        assert 'temporal_threshold_days' in config
        assert 'confidence_threshold_percent' in config
        
        # Values should match preprocessor settings
        assert config['spatial_threshold_degrees'] == preprocessor.spatial_threshold
        assert config['temporal_threshold_days'] == preprocessor.temporal_threshold
        assert config['confidence_threshold_percent'] == preprocessor.confidence_threshold
    
    def test_load_firms_data_from_nonexistent_file(self, preprocessor):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_firms_data_from_file("nonexistent_file.csv")
    
    def test_export_events(self, preprocessor, sample_firms_data, tmp_path):
        """Test event export functionality."""
        # Process data to get events
        standardized_df = preprocessor._standardize_firms_dataframe(sample_firms_data)
        events_df = preprocessor.identify_fire_events(standardized_df)
        
        if len(events_df) > 0:
            # Test CSV export
            csv_output = tmp_path / "test_events.csv"
            result_path = preprocessor.export_events(events_df, csv_output, format='csv')
            
            assert result_path.exists()
            assert result_path.suffix == '.csv'
            
            # Verify exported data can be read back
            exported_df = pd.read_csv(result_path)
            assert len(exported_df) == len(events_df)
    
    @pytest.mark.parametrize("invalid_data", [
        # Missing longitude
        pd.DataFrame({'latitude': [39.5], 'date': ['2023-01-01']}),
        # Missing latitude  
        pd.DataFrame({'longitude': [-120.5], 'date': ['2023-01-01']}),
        # Missing date
        pd.DataFrame({'longitude': [-120.5], 'latitude': [39.5]}),
        # Invalid coordinates
        pd.DataFrame({
            'longitude': [200.0],  # Out of range
            'latitude': [39.5],
            'date': ['2023-01-01']
        }),
    ])
    def test_invalid_data_handling(self, preprocessor, invalid_data):
        """Test handling of invalid FIRMS data."""
        if 'date' not in invalid_data.columns:
            # Should raise error for missing required columns
            with pytest.raises(ValueError):
                preprocessor._standardize_firms_dataframe(invalid_data)
        else:
            # Should handle invalid data gracefully
            result = preprocessor._standardize_firms_dataframe(invalid_data)
            # Invalid coordinates should be filtered out
            assert len(result) == 0 or result['longitude'].between(-180, 180).all()

def test_import_structure():
    """Test that preprocessor can be imported correctly."""
    try:
        from src.data.preprocessor import FIRMSPreprocessor
        from src.data import FIRMSPreprocessor as ImportedPreprocessor
        
        assert FIRMSPreprocessor is ImportedPreprocessor
        assert True  # All imports successful
    except ImportError as e:
        pytest.fail(f"Import error: {e}")

if __name__ == "__main__":
    # Run basic tests without pytest
    print("🔥 FIRMS Preprocessor - Basic Tests")
    print("=" * 40)
    
    # Test imports
    try:
        from src.data.preprocessor import FIRMSPreprocessor
        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)
    
    # Test initialization
    try:
        preprocessor = FIRMSPreprocessor()
        print("✓ Preprocessor initialization successful")
        print(f"  - Spatial threshold: {preprocessor.spatial_threshold}°")
        print(f"  - Temporal threshold: {preprocessor.temporal_threshold} days")
        print(f"  - Confidence threshold: {preprocessor.confidence_threshold}%")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        sys.exit(1)
    
    # Test configuration
    try:
        summary = preprocessor.get_processing_summary()
        print("✓ Configuration and summary generation works")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        sys.exit(1)
    
    print("\n✅ All basic tests passed!")
    print("\nTo run full test suite:")
    print("  pytest tests/test_fire_preprocessing.py -v")