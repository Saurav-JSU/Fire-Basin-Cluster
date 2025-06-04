"""
Pytest configuration file to set up proper import paths.
"""
import sys
from pathlib import Path

# Add project root and src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"

# Add to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))