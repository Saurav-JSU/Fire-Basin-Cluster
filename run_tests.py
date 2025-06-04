#!/usr/bin/env python
"""
Simple test runner script that handles Python path setup.
"""
import sys
from pathlib import Path

# Add project root and src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def run_basic_tests():
    """Run basic tests without pytest."""
    print("🔥 Wildfire Watershed Clustering - Basic Tests")
    print("=" * 50)
    
    # Test imports
    print("Testing imports...")
    try:
        # Test src imports
        from src.data.gee_loader import GEEDataLoader
        print("✓ GEEDataLoader import successful")
        
        # Test config imports - try different approaches
        try:
            from config.settings import GEE_CONFIG, DATASETS, STUDY_AREA, FIRE_CONFIG
            print("✓ Config imports successful")
        except ImportError:
            # Alternative import method
            import sys
            import os
            config_path = Path(__file__).parent / "config"
            sys.path.insert(0, str(config_path.parent))
            from config.settings import GEE_CONFIG, DATASETS, STUDY_AREA, FIRE_CONFIG
            print("✓ Config imports successful (alternative method)")
            
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nTrying alternative import approach...")
        
        # Try direct file import as fallback
        try:
            import importlib.util
            config_file = project_root / "config" / "settings.py"
            spec = importlib.util.spec_from_file_location("settings", config_file)
            settings = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(settings)
            
            GEE_CONFIG = settings.GEE_CONFIG
            DATASETS = settings.DATASETS
            STUDY_AREA = settings.STUDY_AREA
            FIRE_CONFIG = settings.FIRE_CONFIG
            
            print("✓ Config loaded using direct file import")
        except Exception as e2:
            print(f"✗ All import methods failed: {e2}")
            return False
    
    # Test loader initialization
    print("\nTesting loader initialization...")
    try:
        loader = GEEDataLoader(project_id='ee-jsuhydrolabenb')
        print(f"✓ Loader initialized successfully")
        print(f"  - Project ID: {loader.project_id}")
        print(f"  - Authenticated: {loader.authenticated}")
    except Exception as e:
        print(f"✗ Loader initialization failed: {e}")
        return False
    
    # Test configuration
    print("\nTesting configuration...")
    try:
        bounds = loader.study_area_bounds
        print(f"✓ Study area bounds loaded: {bounds}")
        
        # Validate bounds make sense for Western US
        if (bounds['west'] < bounds['east'] and 
            bounds['south'] < bounds['north'] and
            -130 <= bounds['west'] <= -100 and
            25 <= bounds['south'] <= 50):
            print("✓ Bounds validation passed")
        else:
            print("⚠ Bounds seem incorrect for Western US")
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    # Test dataset info (without authentication)
    print("\nTesting dataset info...")
    try:
        info = loader.get_dataset_info()
        print(f"✓ Dataset info retrieved")
        print(f"  - Authenticated: {info['authenticated']}")
        print(f"  - HUC12 loaded: {info['datasets']['huc12']['loaded']}")
        print(f"  - FIRMS loaded: {info['datasets']['firms']['loaded']}")
    except Exception as e:
        print(f"✗ Dataset info test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All basic tests passed!")
    print("\nNext steps:")
    print("1. Authenticate with Google Earth Engine:")
    print("   earthengine authenticate")
    print("2. Test with authentication:")
    print("   python -c \"from src.data.gee_loader import GEEDataLoader; loader = GEEDataLoader('ee-jsuhydrolabenb'); loader.authenticate(); print('✅ GEE authentication successful!')\"")
    print("3. Run full test suite:")
    print("   pytest tests/ -v")
    
    return True

def run_pytest():
    """Run pytest with proper path setup."""
    import pytest
    
    # Run pytest with current directory
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short"
    ])
    
    return exit_code == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for wildfire watershed clustering")
    parser.add_argument("--pytest", action="store_true", help="Run pytest instead of basic tests")
    parser.add_argument("--auth", action="store_true", help="Test GEE authentication")
    
    args = parser.parse_args()
    
    if args.auth:
        print("Testing Google Earth Engine authentication...")
        try:
            # Import with error handling
            try:
                from src.data.gee_loader import GEEDataLoader
            except ImportError as e:
                print(f"Import error: {e}")
                # Add current directory to path and try again
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from src.data.gee_loader import GEEDataLoader
            
            loader = GEEDataLoader('ee-jsuhydrolabenb')
            if loader.authenticate():
                print("✅ GEE authentication successful!")
                # Try loading a small sample
                print("Testing data loading...")
                huc12_data = loader.load_huc12_watersheds()
                print("✅ HUC12 data loading successful!")
            else:
                print("❌ GEE authentication failed!")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            sys.exit(1)
    elif args.pytest:
        success = run_pytest()
        sys.exit(0 if success else 1)
    else:
        success = run_basic_tests()
        sys.exit(0 if success else 1)