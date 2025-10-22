#!/bin/bash

# GitHub Repository Setup Script
# Run this after creating your GitHub repository

echo "🐙 Setting up GitHub repository for Food Recognition System"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    echo "Run 'git init' first"
    exit 1
fi

# Get repository URL from user
echo "📝 Please provide your GitHub repository URL:"
echo "Example: https://github.com/yourusername/food-recognition-system.git"
read -p "Repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ Error: Repository URL is required"
    exit 1
fi

echo ""
echo "🔗 Adding remote repository..."
git remote add origin $REPO_URL

echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "✅ Repository successfully pushed to GitHub!"
echo "🌐 Visit your repository at: $REPO_URL"
echo ""
echo "📋 Next steps:"
echo "1. Add repository description"
echo "2. Add topics/tags"
echo "3. Set up GitHub Pages (optional)"
echo "4. Create releases (optional)"
echo ""
echo "🎉 Your Food Recognition System is now on GitHub!"
