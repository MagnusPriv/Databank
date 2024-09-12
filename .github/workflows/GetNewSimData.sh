#!/bin/bash

# Directory path:
TARGET_DIR="Scripts/BuildDatabank/info_files"

#The branch we are comparing to. For final build this will be the branch the user opens merge requests to:
TARGET_BRANCH="docker_test"


# Name of file we're going to store new filenames in: 
OUTPUT_FILE="new_files.txt"

# Path to the AddData:
ADD_DATA_SCRIPT="Scripts/BuildDatabank/AddData.py"

#Makes the output file:
> $OUTPUT_FILE


git fetch origin
git checkout $BRANCH_NAME

#Finding new added files in this branch relative to the other branch meantioned here:
NEW_FILES=$(git diff --name-status origin/$BRANCH_NAME origin/$TARGET_BRANCH | grep "$TARGET_DIR" | awk '{print $2}')

# If new files is not Null:
if [ -n "$NEW_FILES" ]; then
  echo "$NEW_FILES"
  echo "$NEW_FILES" > "$OUTPUT_FILE"  
  
  # Run AddData.py for each new file listed in the output file:
  while IFS= read -r file; do
    echo "Running AddData.py for $file"
    python3 "$ADD_DATA_SCRIPT" -f "$file"  
  done < "$OUTPUT_FILE"

else
  echo "No new files detected in $TARGET_DIR."
fi

pwd
git remote -v
git status 