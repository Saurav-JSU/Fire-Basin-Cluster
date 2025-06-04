"""
Test suite for Google Earth Engine data loading functionality.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.gee_loader import GEEDataLoader
from config.settings import STUDY_AREA, FIRE_CONFIG, DATASETS

class TestGEEDataLoader:
    """Test cases for GEE data loader."""
    
    @pytest.fixture
    def loader(self):
        """Create GEEDataLoader instance for testing."""
        return GEEDataLoader(project_id='ee-jsuhydrolabenb')
    
    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly."""
        assert loader is not None
        assert loader.authenticated is False
        assert loader.study_area_bounds is not None
        assert loader.huc12_data is None
        assert loader.firms_data is None
    
    def test_study_area_bounds(self, loader):
        """Test study area configuration."""
        bounds = loader.study_area_bounds
        
        # Check all required keys present
        required_keys = ["west", "east", "south", "north"]
        assert all(key in bounds for key in required_keys)
        
        # Check reasonable values for Western US
        assert bounds["west"] < bounds["east"]
        assert bounds["south"] < bounds["north"]
        assert -130 <= bounds["west"] <= -100  # Reasonable longitude range
        assert -110 <= bounds["east"] <= -95   # Reasonable longitude range
        assert 25 <= bounds["south"] <= 45     # Reasonable latitude range
        assert 40 <= bounds["north"] <= 50     # Reasonable latitude range
    
    @pytest.mark.skip(reason="Requires GEE authentication")
    def test_authentication(self, loader):
        """Test GEE authentication (requires manual setup)."""
        # This test requires actual GEE credentials
        # Skip by default, run manually when testing with credentials
        success = loader.authenticate()
        assert success is True
        assert loader.authenticated is True
    
    def test_dataset_config(self):
        """Test dataset configuration values."""
        # Test HUC12 config
        huc12_config = DATASETS["huc12"]
        assert huc12_config["asset_id"] == "USGS/WBD/2017/HUC12"
        assert huc12_config["type"] == "FeatureCollection"
        
        # Test FIRMS config
        firms_config = DATASETS["firms"]
        assert firms_config["asset_id"] == "FIRMS"
        assert firms_config["type"] == "ImageCollection"
        assert "start_date" in firms_config
        assert "end_date" in firms_config
    
    def test_fire_config(self):
        """Test fire processing configuration."""
        assert FIRE_CONFIG["confidence_threshold"] == 80
        assert FIRE_CONFIG["spatial_threshold_degrees"] == 0.01
        assert FIRE_CONFIG["temporal_threshold_days"] == 5
        assert FIRE_CONFIG["fire_end_threshold_days"] == 16
    
    def test_get_dataset_info_without_auth(self, loader):
        """Test dataset info without authentication."""
        info = loader.get_dataset_info()
        
        assert "authenticated" in info
        assert info["authenticated"] is False
        assert "study_area" in info
        assert "datasets" in info
        assert "huc12" in info["datasets"]
        assert "firms" in info["datasets"]

def test_import_structure():
    """Test that all required modules can be imported."""
    try:
        from src.data.gee_loader import GEEDataLoader
        from config.settings import (
            GEE_CONFIG, DATASETS, STUDY_AREA, FIRE_CONFIG
        )
        assert True  # All imports successful
    except ImportError as e:
        pytest.fail(f"Import error: {e}")

if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running basic data loading tests...")
    
    # Test imports first
    try:
        from src.data.gee_loader import GEEDataLoader
        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)
    
    # Test loader initialization
    loader = GEEDataLoader(project_id='ee-jsuhydrolabenb')
    print(f"✓ Loader initialized: {loader is not None}")
    
    # Test configuration
    bounds = loader.study_area_bounds
    print(f"✓ Study area bounds: {bounds}")
    
    # Test dataset info (without auth)
    try:
        info = loader.get_dataset_info()
        print(f"✓ Dataset info retrieved: {info['authenticated']}")
    except Exception as e:
        print(f"✗ Dataset info error: {e}")
    
    print("\nBasic tests completed!")
    print("\nTo run full tests with GEE authentication:")
    print("1. Set up Google Earth Engine authentication")
    print("2. Run: pytest tests/test_data_loading.py -v")