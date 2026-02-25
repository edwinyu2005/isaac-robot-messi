#!/bin/bash
# setup.sh: Prepare external dependencies inside Isaac Sim's embedded Python environment.

# PREREQUISITE: You MUST install Isaac Sim (4.5) first.
# Download here: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html

# Update the path to your extracted standalone folder
ISAAC_SIM_PATH=${1:-"$HOME/isaacsim"}

if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "[ERROR] Isaac Sim python wrapper not found at: $ISAAC_SIM_PATH"
    echo "========================================================================"
    echo "[ACTION REQUIRED] You must install the base engine first!"
    echo "1. Install 'Isaac Sim' (Version 4.5) standalone. Download and unzip it to your desired location."
    echo "2. Rerun this script pointing to your exact installation path:"
    echo "   Usage: ./setup.sh /path/to/your/isaac-sim-folder"
    echo "========================================================================"
    exit 1
fi

echo "[INFO] Using Isaac Sim environment at: $ISAAC_SIM_PATH"

# Install dependencies using Isaac Sim's built-in python.
echo "[INFO] Installing external dependencies into Isaac Sim..."
$ISAAC_SIM_PATH/python.sh -m pip install -r requirements.txt

echo "[SUCCESS] Setup complete!"
echo "--------------------------------------------------------"
echo "To run scripts, ALWAYS use the Isaac Sim python wrapper:"
echo "$ISAAC_SIM_PATH/python.sh scripts/check_scene.py"
echo "--------------------------------------------------------"
