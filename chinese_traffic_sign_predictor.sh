#!/usr/bin/env bash
set -euo pipefail

# Bash wrapper for Chinese_Traffic_Sign_Predictor.py
# Predict Chinese traffic sign type, display type and confidence.
#
# Inputs:
#   Path to saved CNN keras model
#   Paths to all traffic sign images desired to be categorized
#
# Outputs:
#   matplotlib.pyplot pop-up with classification results for each Chinese
#   traffic sign image input.

if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 <model_path> <image_path_1> <image_path_2> ..."
    exit 1
fi

model_path="$1"

shift

python3 Chinese_Traffic_Sign_Predictor.py "$model_path" "$@"