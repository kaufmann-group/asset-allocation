#!/bin/bash
set -e

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

python3 -m venv ocean
source ocean/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate the environment with:"
echo "source ocean/bin/activate"
