#!/bin/bash

#This script creates branch in repository identical to one coming from a pull request from a fork. Then it creates a pull request to variable BRANCH_NAME

# Update the repository to ensure it’s up to date
git pull  || { echo "Failed to pull latest changes"; exit 1; }

# Fetch the PR branch from the fork and create a new branch based on it
git fetch origin pull/${PR_NUMBER}/head:${BRANCH_NAME} &&
git checkout ${BRANCH_NAME} || { echo "Failed to fetch or checkout the branch"; exit 1; }

# Set Git username and email
./Scripts/WorkflowScripts/RunnerGitConfig.sh || { echo "Failed to configure Git user settings"; exit 1; }

# Push the new branch to the main repository
git push https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY} ${BRANCH_NAME}:${BRANCH_NAME} || { echo "Failed to push the branch"; exit 1; }

