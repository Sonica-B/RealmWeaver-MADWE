#!/bin/bash

# Setup script for MADWE project
set -e

echo "Setting up MADWE development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [[ ! "$python_version" == "$required_version"* ]]; then
    echo "Error: Python $required_version is required, but $python_version is installed."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/{raw,processed,models}/{textures,sprites,gameplay,train,val,test,checkpoints,final}
touch data/{raw,processed,models}/.gitkeep

# Download initial datasets (if needed)
echo "Ready to download datasets. Run: python scripts/download_data.py"

echo "Setup complete! Activate environment with: source venv/bin/activate"