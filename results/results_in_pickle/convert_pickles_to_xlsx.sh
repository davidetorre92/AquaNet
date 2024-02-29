#!/bin/bash

# Directory containing the pickle files
PICKLE_DIR="./"

# Directory to save the xlsx files
XLSX_DIR="../"

# Make sure the output directory exists
mkdir -p "$XLSX_DIR"

# Iterate over all pickle files in the directory
for pickle_file in "$PICKLE_DIR"/*.pickle; do
    # Get the base name of the file without the extension
    base_name=$(basename "$pickle_file" .pickle)
    
    # Define the output xlsx file path
    xlsx_file="$XLSX_DIR/${base_name}.xlsx"
    
    # Use Python to convert the pickle file to an xlsx file
    python3 -c "import pandas as pd; df = pd.read_pickle('$pickle_file'); df.to_excel('$xlsx_file', index=False)"
    
    echo "Converted $pickle_file to $xlsx_file"
done

