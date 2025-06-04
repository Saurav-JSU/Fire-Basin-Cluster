# Wildfire Watershed Clustering Analysis

## 🎯 Project Overview
This project clusters HUC12 watershed subbasins in the Western United States based on their wildfire history and fire activity characteristics using Google Earth Engine data. The clustering serves as a benchmark for post-wildfire hydrological analysis.

## 📊 Datasets
- **HUC12 Subwatersheds**: `USGS/WBD/2017/HUC12` (2017-04-22)
- **FIRMS Fire Data**: `FIRMS` (2000-11-01 to 2025-06-03)
- **Study Area**: Western United States
- **Fire Confidence Filter**: >80% (high-accuracy detections only)

## 🚀 Quick Start

### Prerequisites
1. **Google Earth Engine Account**: Sign up at [https://earthengine.google.com/](https://earthengine.google.com/)
2. **Python 3.8+**: Ensure you have Python 3.8 or higher installed
3. **Git**: For cloning the repository

### Installation

#### Quick Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd wildfire_watershed_clustering

# Run setup script
# On Linux/Mac:
chmod +x setup.sh
./setup.sh

# On Windows:
setup.bat
```

#### Manual Installation

##### Option 1: Conda (Recommended for geospatial packages)
```bash
# Clone the repository
git clone <repository-url>
cd wildfire_watershed_clustering

# Create conda environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate wildfire-watershed-clustering

# Authenticate with Google Earth Engine (first time only)
earthengine authenticate
```

##### Option 2: Pip + Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd wildfire_watershed_clustering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (first time only)
earthengine authenticate
```

> **Note**: Conda is recommended for this project because it handles geospatial dependencies (GDAL, GEOS, PROJ) more reliably than pip.

### Basic Usage
```python
from src.data.gee_loader import GEEDataLoader

# Initialize the data loader with your project ID
loader = GEEDataLoader(project_id='ee-jsuhydrolabenb')

# Authenticate with Google Earth Engine
loader.authenticate()

# Load HUC12 watersheds for Western US
huc12_data = loader.load_huc12_watersheds()

# Load FIRMS fire data (example: 2020 data)
firms_data = loader.load_firms_data(
    start_date="2020-01-01",
    end_date="2020-12-31"
)

# Get dataset information
info = loader.get_dataset_info()
print(info)
```

### Testing
```bash
# Option 1: Run simple test first (recommended for troubleshooting)
python simple_test.py

# Option 2: Run custom test runner  
python run_tests.py

# Option 3: Test with authentication
python simple_test.py --auth
# OR
python run_tests.py --auth

# Option 4: Run pytest directly  
python run_tests.py --pytest

# Option 5: Traditional pytest (may need path setup)
pytest tests/ -v
```

## 🔧 Troubleshooting

### Common Issues

**1. GDAL/GEOS/PROJ Installation Issues**
- **Solution**: Use conda instead of pip for geospatial packages:
  ```bash
  conda env create -f environment.yml
  ```

**2. Google Earth Engine Authentication**
- **Error**: `ee.EEException: Please authorize access to your Earth Engine account`
- **Solution**: Run the authentication command:
  ```bash
  earthengine authenticate
  ```

**3. Import/Module Errors**
- **Error**: `ModuleNotFoundError: No module named 'config.settings'` or `No module named 'src'`
- **Solution**: Try these approaches in order:
  1. Use the simple test: `python simple_test.py`
  2. Install in development mode: `pip install -e .`
  3. Use the custom test runner: `python run_tests.py`
  4. Check you're in the project root directory: `ls` should show `config/`, `src/`, etc.

**4. Config Import Issues**
- **Error**: Config-related import failures
- **Solution**: The `simple_test.py` script has robust fallback config loading and will help diagnose the issue

**4. Memory Issues with Large Datasets**
- **Solution**: Adjust chunk sizes in `config/settings.py`:
  ```python
  PROCESSING_CONFIG["chunk_size"] = 500  # Reduce from 1000
  PROCESSING_CONFIG["memory_limit"] = "32GB"  # Adjust based on your system
  ```

**5. Windows Path Issues**
- **Error**: Path-related errors on Windows
- **Solution**: Use the `setup.bat` script or ensure you're using forward slashes in paths

### Getting Help
- Check the [Issues](https://github.com/your-repo/issues) page for known problems
- Create a new issue with detailed error messages and system information
- Include your Python version, OS, and whether you're using conda or pip
### Primary Metrics
- Fire return intervals (mean, median, variability)
- Fire frequency (events per decade) 
- Burn area fraction per watershed
- Fire seasonality patterns
- Fire intensity proxies (FRP from FIRMS)
- Time since last fire

### Advanced Metrics
- Fire regime stability indices
- Spatial burn patterns within watersheds
- Multi-temporal fire clustering
- Fire duration characteristics

## 🏗️ Project Structure
```
wildfire_watershed_clustering/
├── README.md
├── requirements.txt
├── environment.yml           # Conda environment specification
├── setup.py                  # Package installation configuration
├── setup.sh                  # Setup script for Linux/Mac
├── setup.bat                 # Setup script for Windows
├── conftest.py              # Pytest configuration
├── run_tests.py             # Custom test runner
├── simple_test.py           # Simple standalone test (recommended first)
├── test_preprocessor.py     # FIRMS preprocessing test script
├── test_step3_fire_metrics.py # Step 3 fire metrics test script
├── .gitignore               # Git ignore patterns
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── gee_loader.py      # Google Earth Engine data loading
│   │   └── preprocessor.py    # FIRMS fire event preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── fire_metrics.py    # Watershed fire characteristics calculation
│   │   └── temporal_analysis.py  # Advanced temporal pattern analysis
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── algorithms.py      # Clustering implementations
│   │   └── validation.py      # Cluster validation methods
│   └── visualization/
│       ├── __init__.py
│       ├── maps.py           # Geospatial visualizations
│       └── plots.py          # Statistical plots
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering_analysis.ipynb
│   └── 04_results_visualization.ipynb
├── data/
│   ├── raw/                  # Downloaded GEE data
│   ├── processed/            # Processed datasets
│   └── results/              # Clustering results
└── tests/
    ├── __init__.py
    ├── test_data_loading.py
    ├── test_fire_preprocessing.py
    └── test_fire_metrics.py      # Unit tests for Step 3 components
```

## 🖥️ System Specifications
- **CPU**: AMD Ryzen Threadripper PRO 7995WX (96 cores, 192 threads)
- **GPU**: NVIDIA RTX 6000 Ada Generation (49.1 GB VRAM)
- **Parallel Processing**: Leveraged for large-scale data processing

## 📋 Development Progress

### ✅ Completed
- [x] Project structure setup
- [x] Initial README documentation
- [x] **Step 1**: Google Earth Engine data loading infrastructure
  - [x] Configuration management (`config/settings.py`)
  - [x] GEE authentication and data loading (`src/data/gee_loader.py`)
  - [x] HUC12 watershed boundary loading with Western US filtering
  - [x] FIRMS fire data loading with confidence filtering (>80%)
  - [x] Study area geometry creation and spatial filtering
  - [x] Basic test suite for data loading functionality
  - [x] Package structure and imports setup
  - [x] Conda environment configuration (`environment.yml`)
  - [x] Automated setup scripts for Linux/Mac/Windows
  - [x] Git repository structure with .gitignore
  - [x] Cross-platform development environment support
  - [x] **Fixed**: Python import path issues and module structure
  - [x] **Added**: Project ID configuration (ee-jsuhydrolabenb)
  - [x] **Added**: Custom test runner for easier development
  - [x] **Added**: Development package installation setup
  - [x] **Added**: Robust config import handling with fallbacks
  - [x] **Added**: Simple standalone test script (`simple_test.py`)
  - [x] **Enhanced**: Error handling and troubleshooting guides
- [x] **Step 2**: FIRMS data preprocessing and fire event identification ✅
  - [x] Real FIRMS data extraction from Google Earth Engine
  - [x] Spatial-temporal clustering using DBSCAN algorithm
  - [x] Fire event characterization and quality metrics
  - [x] Scientific methodology based on published research
  - [x] Comprehensive testing and validation framework
  - [x] Export capabilities for processed fire events
- [x] **Step 3**: Watershed fire metrics calculation ✅
  - [x] Fire return interval analysis (mean, median, variability, trends)
  - [x] Burn fraction and spatial coverage calculations  
  - [x] Fire seasonality and temporal distribution analysis
  - [x] Fire intensity metrics (FRP, duration, confidence aggregation)
  - [x] Composite fire regime indices and classifications
  - [x] Advanced temporal pattern analysis (trends, cycles, regime changes)
  - [x] Comprehensive fire characterization pipeline
  - [x] Scientific methodology based on fire ecology research
  - [x] Export capabilities and summary statistics

### 🚧 In Progress
- [x] **Step 1**: Google Earth Engine data loading infrastructure ✅ 
- [x] **Step 2**: FIRMS data preprocessing and fire event identification ✅
- [x] **Step 3**: Watershed fire metrics calculation ✅
  - [x] Fire return interval analysis (mean, median, variability)
  - [x] Burn fraction calculations per watershed
  - [x] Fire seasonality and temporal patterns
  - [x] Fire intensity aggregation metrics (FRP, duration, confidence)
  - [x] Composite fire regime indices and classifications
  - [x] Advanced temporal pattern analysis (trends, cycles, regime changes)
  - [x] Complete watershed fire characterization pipeline
- [ ] **Step 4**: Clustering algorithm implementation and validation
  - [ ] Feature selection and standardization
  - [ ] Multiple clustering algorithms (K-means, DBSCAN, GMM, Hierarchical)
  - [ ] Optimal cluster number determination
  - [ ] Cluster validation and interpretation

### 📅 Upcoming
- [ ] **Step 4**: Clustering algorithm implementation and validation
- [ ] **Step 5**: Feature selection and dimensionality reduction
- [ ] **Step 6**: Cluster interpretation and validation
- [ ] **Step 7**: Visualization and reporting tools
- [ ] **Step 8**: Hydrological relevance testing and case studies
- [ ] **Step 9**: Documentation and publication preparation

## 🔬 Methodology
Based on recent research in fire regime classification and watershed analysis:
- **Clustering Approach**: Multi-scale hierarchical clustering (DBSCAN, GMM, Hierarchical)
- **Fire Event Definition**: Spatial-temporal clustering with 5-day temporal and 0.01° spatial thresholds
- **Validation**: Geographic coherence, hydrological relevance, expert knowledge integration

## 📚 Key References
- Global pyromes and fire regime classification research
- FIRMS fire detection and characterization methods
- HUC12 watershed boundary applications
- Post-wildfire hydrological impact studies

---
*Last Updated: June 4, 2025*