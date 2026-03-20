#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --quiet torch numpy matplotlib

echo "Running benchmark..."
python benchmark.py
