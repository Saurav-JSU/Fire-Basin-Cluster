#!/usr/bin/env python
"""
Simple standalone test script that doesn't rely on complex imports.
"""
import sys
from pathlib import Path

def test_basic_setup():
    """Test basic project setup without complex imports."""
    print("🔥 Wildfire Watershed Clustering - Simple Test")
    print("=" * 50)
    
    # Test Python environment
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    
    # Test project structure
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    required_dirs = ["src", "config", "data", "tests"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\nMissing directories: {missing_dirs}")
        return False
    
    # Test key files
    key_files = [
        "config/settings.py",
        "src/data/gee_loader.py",
        "requirements.txt",
        "environment.yml"
    ]
    
    missing_files = []
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    # Test basic imports
    print("\nTesting basic imports...")
    
    # Add paths for imports
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    try:
        # Test geemap (GEE dependency)
        import geemap
        print("✓ geemap import successful")
    except ImportError as e:
        print(f"✗ geemap import failed: {e}")
        return False
    
    try:
        # Test earthengine
        import ee
        print("✓ earthengine-api import successful")
    except ImportError as e:
        print(f"✗ earthengine-api import failed: {e}")
        return False
    
    try:
        # Test geopandas
        import geopandas as gpd
        print("✓ geopandas import successful")
    except ImportError as e:
        print(f"✗ geopandas import failed: {e}")
        return False
    
    # Test our modules
    try:
        # Try importing config using direct file loading
        import importlib.util
        config_file = project_root / "config" / "settings.py"
        spec = importlib.util.spec_from_file_location("settings", config_file)
        settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings)
        
        # Check if required config variables exist
        required_configs = ["GEE_CONFIG", "DATASETS", "STUDY_AREA", "FIRE_CONFIG"]
        for config_name in required_configs:
            if hasattr(settings, config_name):
                print(f"✓ Config {config_name} found")
            else:
                print(f"✗ Config {config_name} missing")
                return False
        
        # Check project ID
        project_id = settings.GEE_CONFIG.get("project_id")
        if project_id:
            print(f"✓ GEE Project ID set: {project_id}")
        else:
            print("✗ GEE Project ID not configured")
            return False
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    try:
        # Test our GEE loader
        from src.data.gee_loader import GEEDataLoader
        print("✓ GEEDataLoader import successful")
        
        # Test initialization
        loader = GEEDataLoader(project_id='ee-jsuhydrolabenb')
        print(f"✓ GEEDataLoader initialized with project: {loader.project_id}")
        
    except Exception as e:
        print(f"✗ GEEDataLoader test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All basic tests passed!")
    print("\nNext steps:")
    print("1. Authenticate with Google Earth Engine:")
    print("   earthengine authenticate")
    print("2. Test authentication:")
    print("   python simple_test.py --auth")
    print("3. Run full test suite:")
    print("   python run_tests.py")
    
    return True

def test_authentication():
    """Test Google Earth Engine authentication."""
    print("🌍 Testing Google Earth Engine Authentication")
    print("=" * 50)
    
    # Add paths for imports
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    try:
        from src.data.gee_loader import GEEDataLoader
        
        loader = GEEDataLoader(project_id='ee-jsuhydrolabenb')
        print(f"Loader created with project: {loader.project_id}")
        
        if loader.authenticate():
            print("✅ Google Earth Engine authentication successful!")
            
            # Test basic data access
            print("Testing basic data access...")
            try:
                info = loader.get_dataset_info()
                print(f"✓ Dataset info: {info['authenticated']}")
                
                # Try to load a very small sample
                print("Testing HUC12 data loading...")
                huc12_sample = loader.load_huc12_watersheds()
                print("✅ HUC12 data accessible!")
                
                return True
                
            except Exception as e:
                print(f"⚠ Data access test failed: {e}")
                print("Authentication successful but data access has issues")
                return False
        else:
            print("❌ Google Earth Engine authentication failed!")
            print("\nTroubleshooting:")
            print("1. Run: earthengine authenticate")
            print("2. Follow the browser authentication flow")
            print("3. Make sure you have access to Google Earth Engine")
            return False
            
    except Exception as e:
        print(f"❌ Authentication test failed with error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple test for wildfire watershed clustering setup")
    parser.add_argument("--auth", action="store_true", help="Test GEE authentication")
    
    args = parser.parse_args()
    
    if args.auth:
        success = test_authentication()
    else:
        success = test_basic_setup()
    
    sys.exit(0 if success else 1)