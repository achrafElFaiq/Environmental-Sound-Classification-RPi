#!/bin/bash

# Directory containing .wav files
WAV_DIR="../urbansound/fold1/"

# Python script to run
SCRIPT="classify_ps2024.py"

# Output file
OUTPUT_FILE="test.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Loop through each .wav file in the directory
for wav_file in "$WAV_DIR"*.wav; do
    # Run the Python script, filter and get the predicted class line, and append it to the output file
    prediction=$(python3 "$SCRIPT" "$wav_file" | grep 'Predicted class' | tail -n 1)
    # Append the file name and the predicted class line to the output file
    echo "$wav_file: $prediction" >> "$OUTPUT_FILE"
done
