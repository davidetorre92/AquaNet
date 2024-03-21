#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -t <table_path> or $0 --table <table_path>"
    exit 1
}

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    usage
fi

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--table) FILE_PATH="$2"; shift ;;  # Get the table path
        *) usage ;;
    esac
    shift
done

# Check if the file exists
if [ -f "$FILE_PATH" ]; then
    # File exists, run the python command
    python3 -c "import pandas as pd; df = pd.read_pickle('$FILE_PATH'); print(df)"
else
    # File does not exist, print an error message
    echo "Error: File $FILE_PATH does not exist."
    exit 1
fi

