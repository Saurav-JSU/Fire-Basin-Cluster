"""
Setup script for wildfire watershed clustering project.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="wildfire-watershed-clustering",
    version="0.1.0",
    author="Wildfire Watershed Clustering Team",
    author_email="",
    description="Clustering HUC12 watersheds based on wildfire characteristics using Google Earth Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/wildfire-watershed-clustering",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black",
            "flake8",
            "isort",
        ],
        "gpu": [
            "cupy-cuda12x>=12.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wildfire-clustering=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
    },
    zip_safe=False,
)