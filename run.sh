
#chmod +x run.sh

#!/bin/bash
# Run script for ISAC-UAV project

echo "Setting up environment..."
pip install -e .

echo "Running ISAC-UAV pipeline..."
python src/main.py
