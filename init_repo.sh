#!/bin/bash
# Initialize a new Git repository for supervised_radar-sounder

# Exit immediately if a command fails
set -e

# Create README file
echo "# supervised_radar-sounder" >> README.md

# Initialize Git and make first commit
git init
git add README.md
git commit -m "first commit"

# Set branch name
git branch -M main

# Add remote origin (replace with your actual GitHub repo if different)
git remote add origin https://github.com/milkisa/supervised_lorasam.git

# Create .gitignore and ignore results folder
echo "results/" >> .gitignore

# (Optional) Ignore common files for PyTorch projects
cat <<EOL >> .gitignore
__pycache__/
*.pyc
*.pbs
*.pbs.o
*.pt
*.pth
*.log
.DS_Store
runs/
checkpoints/
result/
datasets/
EOL

# Commit .gitignore
git add .gitignore
git commit -m "add .gitignore to ignore results and temporary files"

# Push to GitHub
git push -u origin main

echo "✅ Repository initialized and pushed successfully!"
