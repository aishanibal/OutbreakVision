#!/bin/bash

echo "Setting up OutbreakVision environment..."

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
