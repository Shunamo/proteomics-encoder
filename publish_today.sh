#!/bin/bash
# Publish TODAY.md to central repository
# Usage: ./publish_today.sh

set -e

# Configuration
CENTRAL_REPO="allison-eunse/gene-brain"
REPO_OWNER="Shunamo"
REPO_NAME="proteomics-encoder"
TODAY_FILE="TODAY.md"

# Check if TODAY.md exists
if [ ! -f "$TODAY_FILE" ]; then
    echo "Error: $TODAY_FILE not found in current directory"
    echo "Please create $TODAY_FILE first"
    exit 1
fi

# Get today's date in YYYY-MM-DD format
TODAY=$(date +%Y-%m-%d)
TARGET_DIR="team-tracking/${REPO_OWNER}/${REPO_NAME}/daily"
TARGET_FILE="${TARGET_DIR}/${TODAY}.md"

# Check for token
if [ -z "$BIG_REPO_TOKEN" ]; then
    echo "Error: BIG_REPO_TOKEN environment variable is not set"
    echo "Please set it with: export BIG_REPO_TOKEN=your_token_here"
    exit 1
fi

echo "Publishing $TODAY_FILE to central repository..."
echo "Target: $TARGET_FILE"

# Clone central repo
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone --depth 1 "https://${BIG_REPO_TOKEN}@github.com/${CENTRAL_REPO}.git" central-repo
cd central-repo

# Configure git
git config user.name "Shunamo"
git config user.email "shunamo@users.noreply.github.com"

# Create target directory
mkdir -p "$TARGET_DIR"

# Copy TODAY.md to target location
cp "$OLDPWD/$TODAY_FILE" "$TARGET_FILE"

# Commit and push
git add "$TARGET_FILE"
git commit -m "docs: add daily log for ${REPO_OWNER}/${REPO_NAME} - ${TODAY}" || {
    echo "No changes to commit (file might be identical)"
    exit 0
}
git push

# Cleanup
cd "$OLDPWD"
rm -rf "$TEMP_DIR"

echo "Successfully published daily log to central repository!"
echo "View at: https://github.com/${CENTRAL_REPO}/blob/main/${TARGET_FILE}"
