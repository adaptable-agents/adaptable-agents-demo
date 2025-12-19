#!/bin/bash

# Script to run the side-by-side browser demo with venv python
# Usage:
#   ./demo_side_by_side.sh standard    # Run with standard agent
#   ./demo_side_by_side.sh adaptable   # Run with adaptable agent

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for venv in common locations
if [ -d "venv" ]; then
    VENV_PATH="venv"
elif [ -d ".venv" ]; then
    VENV_PATH=".venv"
elif [ -d "env" ]; then
    VENV_PATH="env"
else
    echo "Error: No virtual environment found (checked venv, .venv, env)"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate venv and run the script
source "$VENV_PATH/bin/activate"

# Get agent type from command line argument (default: standard)
AGENT_TYPE="${1:-standard}"

if [ "$AGENT_TYPE" != "standard" ] && [ "$AGENT_TYPE" != "adaptable" ]; then
    echo "Error: Invalid agent type: $AGENT_TYPE"
    echo "Usage: $0 [standard|adaptable]"
    exit 1
fi

echo "Running with $AGENT_TYPE agent..."
echo "=================================="
python demo_browser_side_by_side.py --agent-type "$AGENT_TYPE"
