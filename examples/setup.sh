#!/bin/bash
# Setup virtual environment for llm-abstention-research examples

echo "Creating Python virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run examples:"
echo "  python examples/01_confidence_based.py"
echo "  python examples/02_selective_prediction.py"
echo "  etc."
echo ""
echo "To deactivate environment when done:"
echo "  deactivate"
