# Create the pull request using GitHub CLI
gh pr create --head ${BRANCH_NAME} --base ${TARGET_BRANCH_NAME} \
  --title "${GITHUB_ACTOR}'s New Data" --body "This PR brings in new data from ${GITHUB_ACTOR}" || { echo "Failed to create pull request"; exit 1; }
