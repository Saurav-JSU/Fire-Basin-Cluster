@echo off
REM Wildfire Watershed Clustering Project Setup Script for Windows
REM This script helps set up the development environment on Windows

echo.
echo 🔥 Wildfire Watershed Clustering Project Setup 🔥
echo ==================================================

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Conda detected
    set USE_CONDA=true
) else (
    echo ! Conda not found, will use pip
    set USE_CONDA=false
)

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

python --version
echo ✓ Python detected

REM Setup environment
if "%USE_CONDA%"=="true" (
    echo.
    echo 📦 Setting up Conda environment...
    
    REM Check if environment already exists
    conda env list | findstr "wildfire-watershed-clustering" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ! Environment 'wildfire-watershed-clustering' already exists
        set /p choice="Do you want to remove and recreate it? (y/n): "
        if /i "%choice%"=="y" (
            conda env remove -n wildfire-watershed-clustering
        ) else (
            echo Skipping environment creation
            pause
            exit /b 0
        )
    )
    
    REM Create conda environment
    conda env create -f environment.yml
    
    echo.
    echo ✅ Conda environment created successfully!
    echo.
    echo Installing package in development mode...
    call conda activate wildfire-watershed-clustering
    pip install -e .
    
    echo.
    echo To activate the environment, run:
    echo     conda activate wildfire-watershed-clustering
    
) else (
    echo.
    echo 📦 Setting up Python virtual environment...
    
    REM Create virtual environment
    python -m venv venv
    
    REM Activate virtual environment
    call venv\Scripts\activate.bat
    
    REM Install dependencies
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    REM Install package in development mode
    pip install -e .
    
    echo.
    echo ✅ Virtual environment created successfully!
    echo.
    echo To activate the environment, run:
    echo     venv\Scripts\activate
)

REM Create necessary directories
echo.
echo 📁 Creating project directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\results" mkdir data\results
if not exist "logs" mkdir logs

echo ✓ Directories created

REM Google Earth Engine setup reminder
echo.
echo 🌍 Google Earth Engine Setup
echo ==============================
echo After activating your environment, authenticate with Google Earth Engine:
echo.
if "%USE_CONDA%"=="true" (
    echo     conda activate wildfire-watershed-clustering
) else (
    echo     venv\Scripts\activate
)
echo     earthengine authenticate
echo.

REM Testing reminder
echo 🧪 Testing
echo ==========
echo To test the installation:
echo     python tests\test_data_loading.py
echo.

REM Final message
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Activate your environment (see commands above)
echo 2. Authenticate with Google Earth Engine
echo 3. Run the basic tests
echo 4. Check out the notebooks\ directory for examples
echo.
echo For detailed usage instructions, see README.md

pause