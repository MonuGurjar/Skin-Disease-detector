#!/bin/bash

# ==========================================
# GitHub Auto-Push Script
# ==========================================

# 🔴 PLACE YOUR GITHUB PERSONAL ACCESS TOKEN HERE
GITHUB_TOKEN="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"

GITHUB_USERNAME="MonuGurjar"
REPO_NAME="Skin-Disease-detector"

# Build the authenticated URL
REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

if [ "$GITHUB_TOKEN" == "YOUR_GITHUB_TOKEN_HERE" ]; then
    echo "❌ Error: Please edit this script and replace YOUR_GITHUB_TOKEN_HERE with your actual GitHub token."
    echo "You can generate one at: https://github.com/settings/tokens"
    exit 1
fi

# Ensure git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing new Git repository..."
    git init
    git branch -M main
fi

# Setup Git LFS for large files
echo "Configuring Git LFS for large files..."
git lfs install
git lfs track "*.pth"
git lfs track "*.csv"
git lfs track "*.pb"
git lfs track "*.mp4"

# Add all files (including the newly generated .gitattributes)
echo "Staging files..."
git add .gitattributes
git add .

# Ask for a commit message
read -p "Enter commit message (or press enter for 'Update project'): " COMMIT_MSG
git commit -m "${COMMIT_MSG:-"Update project"}"

# Set the remote and push
git remote remove origin 2>/dev/null
git remote add origin "$REPO_URL"

echo "Pushing code to https://github.com/${GITHUB_USERNAME}/${REPO_NAME} ..."
git push origin main --force

echo "✅ Successfully pushed to GitHub!"
