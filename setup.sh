#!/bin/bash

# Wildfire Watershed Clustering Project Setup Script
# This script helps set up the development environment

set -e  # Exit on any error

echo "🔥 Wildfire Watershed Clustering Project Setup 🔥"
echo "=================================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda detected"
    USE_CONDA=true
else
    echo "! Conda not found, will use pip"
    USE_CONDA=false
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python detected: $(python --version)"

# Setup environment
if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "📦 Setting up Conda environment..."
    
    # Check if environment already exists
    if conda env list | grep -q "wildfire-watershed-clustering"; then
        echo "! Environment 'wildfire-watershed-clustering' already exists"
        read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n wildfire-watershed-clustering
        else
            echo "Skipping environment creation"
            exit 0
        fi
    fi
    
    # Create conda environment
    conda env create -f environment.yml
    
    echo ""
    echo "✅ Conda environment created successfully!"
    echo ""
    echo "Installing package in development mode..."
    eval "$(conda shell.bash hook)"
    conda activate wildfire-watershed-clustering
    pip install -e .
    
    echo ""
    echo "To activate the environment, run:"
    echo "    conda activate wildfire-watershed-clustering"
    
else
    echo ""
    echo "📦 Setting up Python virtual environment..."
    
    # Create virtual environment
    python -m venv venv
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Install package in development mode
    pip install -e .
    
    echo ""
    echo "✅ Virtual environment created successfully!"
    echo ""
    echo "To activate the environment, run:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "    venv\\Scripts\\activate"
    else
        echo "    source venv/bin/activate"
    fi
fi

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/results logs

echo "✓ Directories created"

# Google Earth Engine setup reminder
echo ""
echo "🌍 Google Earth Engine Setup"
echo "=============================="
echo "After activating your environment, authenticate with Google Earth Engine:"
echo ""
if [ "$USE_CONDA" = true ]; then
    echo "    conda activate wildfire-watershed-clustering"
else
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "    venv\\Scripts\\activate"
    else
        echo "    source venv/bin/activate"
    fi
fi
echo "    earthengine authenticate"
echo ""

# Testing reminder
echo "🧪 Testing"
echo "=========="
echo "To test the installation:"
echo "    python tests/test_data_loading.py"
echo ""

# Final message
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate your environment (see commands above)"
echo "2. Authenticate with Google Earth Engine"
echo "3. Run the basic tests"
echo "4. Check out the notebooks/ directory for examples"
echo ""
echo "For detailed usage instructions, see README.md"