#!/bin/bash
set -e

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

python3 -m venv "$REPO/ocean"
source "$REPO/ocean/bin/activate"

pip install --upgrade pip
pip install -r "$REPO/scripts/requirements.txt"

echo ""
echo "Setup complete."
echo "Activate the environment from the project root with:"
echo "source ocean/bin/activate"