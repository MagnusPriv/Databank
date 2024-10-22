#!/bin/bash

# Directory path:
TARGET_DIR="Scripts/BuildDatabank/info_files"

GITHUB_USERNAME="MagnusSletten_Bot"  
GITHUB_EMAIL="magnus.elias.sletten@gmail.com"  

# Set git config for the bot account
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

# Path to AddData.py:
ADD_DATA_SCRIPT="Scripts/BuildDatabank/AddData.py"

# Store absolute paths for later use:
DATABANK_ABS_PATH=$(pwd)
cd "$TARGET_DIR"
TARGET_DIR_ABS=$(pwd)
cd "$DATABANK_ABS_PATH/Scripts/BuildDatabank/"
BUILDDATABANKPATH=$(pwd)

# Fetch the latest changes from the branches:
git fetch origin "$BRANCH_NAME" || { echo "git fetch failed"; exit 1; }
git pull origin "$BRANCH_NAME" || { echo "git pull failed"; exit 1; }

# Create a single work directory before processing all files:
WORK_DIR="/tmp/databank_workdir"
mkdir -p "$WORK_DIR" || { echo "Failed to create work directory"; exit 1; }

# Get sorted list of all .yaml files in BUILDDATABANKPATH
sorted_files=($(find "$BUILDDATABANKPATH/info_files" -name "*.yaml"))


# Calculate the subset of files based on $start_index and $end_index
subset_files=("${sorted_files[@]:$start_index:$(($end_index - $start_index + 1))}")

# Process each file in the subset
for file in "${subset_files[@]}"; do
  if [[ $file == *.yaml ]]; then
    folder=$(dirname "$file") 
    echo "Running AddData.py for $file in folder $folder"
    cd "$BUILDDATABANKPATH"
    python3 "AddData.py" -f "$file" -w "$WORK_DIR" || { echo "AddData.py failed"; exit 1; }
   
  fi
done

# If no new files were detected
if [[ ${#subset_files[@]} -eq 0 ]]; then
  echo "No new files detected in $BUILDDATABANKPATH for the specified range."
fi

cd "$BUILDDATABANKPATH"

# Push changes to the repository:
cd "$DATABANK_ABS_PATH"
git status
git pull
git add .
git commit -m "Files recomputed" || { echo "git commit failed"; exit 1; }
git push https://x-access-token:$GITHUB_TOKEN@github.com/MagnusSletten/Databank.git || { echo "git push failed"; exit 1; }
