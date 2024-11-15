#!/bin/bash
# This script uses regex to update the working directory in Python scripts.

# Check all Python scripts in the subdirectories, excluding the venv/ directory.
# Replace any existing sys.path.append statement with the current working directory.

WORKING_DIRECTORY=$(pwd)
VENV_DIR="./venv"
for file in $(find . -name "*.py" -not -path "$VENV_DIR/*"); do
    # Generate a hash of the file
    hash_before=$(md5sum "$file" | awk '{print $1}')
    # Update sys.path.append with the working directory
    sed -i "s|sys\.path\.append(.*)|sys.path.append('$WORKING_DIRECTORY')|" "$file"
    # Generate a hash of the file
    hash_after=$(md5sum "$file" | awk '{print $1}')
    # If the hash has changed, print a message
    if [ "$hash_before" != "$hash_after" ]; then
        echo "Updated $file"
    fi
done

# Change settings
if [ -f settings.py ]; then
    echo "Settings file settings.py found."
else
    echo "Settings file settings.py not found. Aborting"
    exit 1
fi
# Generate a hash of the file
hash_before=$(md5sum settings.py | awk '{print $1}')
# Update sys.path.append with the working directory
sed -i "s\aquanet_folder = .*\aquanet_folder = '$WORKING_DIRECTORY'\g" settings.py
# Generate a hash of the file
hash_after=$(md5sum settings.py | awk '{print $1}')
# If the hash has changed, print a message
if [ "$hash_before" != "$hash_after" ]; then
    echo "Updated settings."
fi

# Activate environment
source "$VENV_DIR"/bin/activate
# Check if environment is activated correctly
if [ $? -eq 0 ]; then
    echo "Environment activated successfully."
else
    echo "Failed to activate environment in $VENV_DIR."
fi