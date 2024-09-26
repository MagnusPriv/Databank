#!/bin/bash

GITHUB_USERNAME="MagnusSletten_Bot"  
GITHUB_EMAIL="magnus.elias.sletten@gmail.com"  


git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"


# Name of file we're going to store new filenames in:
ORDERPARAMETER_FILE="orderparameters.txt"

# Initialize the output file:
> "$ORDERPARAMETER_FILE"

DATABANK_ABS_PATH=$(pwd)
ORDERPARAMETERS_DIR="Data/experiments/OrderParameters"
cd "$ORDERPARAMETERS_DIR" || exit

git fetch origin $BRANCH_NAME
git pull origin $BRANCH_NAME
cd $DATABANK_ABS_PATH

# Find new added files in this branch relative to the other branch mentioned here:
NEW_ORDERPARAMETER_FILES=$(git diff --name-only origin/$BRANCH_NAME origin/$TARGET_BRANCH -- Data/experiments/)

if [-n $NEW_ORDERPARAMETER_FILES ]; then
  echo "$NEW_ORDERPARAMETER_FILES" > "$ORDERPARAMETER_FILE"
  while IFS= read -r file; do
    # Check if the file is a .dat file
    if [[ $file == *.dat ]]; then
      echo "Running data_to_json.py for $file"
      python3 "$DATABANK_ABS_PATH/data_to_json.py" -f "$DATABANK_ABS_PATH/$file"
      break   # Temporary for testing purposes.
    else
  echo "No new files detected in $TARGET_DIR."
fi

rm "$ORDERPARAMETER_FILE"

cd "$DATABANK_ABS_PATH" 
git status 
git add .
git commit -m "Automated push by NREC"
git push