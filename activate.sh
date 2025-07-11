#!/bin/bash
# Activation script for zcore virtual environment

# Check if we're already in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Already in virtual environment: $VIRTUAL_ENV"
else
    # Get the script directory
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    
    # Activate the virtual environment
    source "$DIR/venv/bin/activate"
    echo "Activated zcore virtual environment"
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
fi
