#!/bin/bash

echo "🧹 Cleaning Git Repository of Unnecessary Files..."
echo ""

# Step 1: Verify .gitignore is complete
echo "Step 1: Verifying .gitignore..."
IGNORE_FILE=".gitignore"
REQUIRED_PATTERNS=(".venv/" "__pycache__/" "*.db" ".env" ".pytest_cache/")

for pattern in "${REQUIRED_PATTERNS[@]}"; do
    if grep -q "$(echo $pattern | sed 's/*/\\*/g')" "$IGNORE_FILE"; then
        echo "  ✅ $pattern found"
    else
        echo "  ⚠️  $pattern missing - adding..."
        echo "$pattern" >> "$IGNORE_FILE"
    fi
done

# Step 2: Remove cached files
echo ""
echo "Step 2: Removing cached files from git tracking..."
git rm -r --cached .venv/ 2>/dev/null && echo "  ✅ Removed .venv/" || true
git rm -r --cached __pycache__/ 2>/dev/null && echo "  ✅ Removed __pycache__/" || true
git rm -r --cached .pytest_cache/ 2>/dev/null && echo "  ✅ Removed .pytest_cache/" || true
git rm -r --cached alembic/__pycache__/ 2>/dev/null && echo "  ✅ Removed alembic/__pycache__/" || true
git rm -r --cached src/__pycache__/ 2>/dev/null && echo "  ✅ Removed src/__pycache__/" || true
git rm --cached citationai.db 2>/dev/null && echo "  ✅ Removed citationai.db" || true
git rm --cached citationai.db-shm 2>/dev/null && echo "  ✅ Removed citationai.db-shm" || true
git rm --cached citationai.db-wal 2>/dev/null && echo "  ✅ Removed citationai.db-wal" || true
git rm --cached .env 2>/dev/null && echo "  ✅ Removed .env" || true
git rm --cached .DS_Store 2>/dev/null && echo "  ✅ Removed .DS_Store" || true

# Step 3: Rebuild index
echo ""
echo "Step 3: Rebuilding git index..."
git add .gitignore
echo "  ✅ Updated .gitignore"

# Step 4: Status check
echo ""
echo "Step 4: Current git status:"
echo "========================================"
git status --short
echo "========================================"

# Step 5: Ask for commit
echo ""
read -p "Proceed with commit? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Step 5: Creating commit..."
    git commit -m "Remove unnecessary files from tracking

- Remove .venv/ and all virtual environment dependencies
- Remove __pycache__/ and compiled Python files  
- Remove .pytest_cache/ and test cache
- Remove database files (citationai.db, citationai.db-*)
- Remove .env file with sensitive credentials
- Remove OS-specific files (.DS_Store)

These directories and files are now properly gitignored and 
won't be tracked in future commits.

Setup instructions:
- Run: pip install -r requirements.txt
- Create .env from .env.example
- Run: alembic upgrade head"
    
    echo ""
    echo "✅ Cleanup successful!"
    echo ""
    echo "Next step: Push to GitHub"
    echo "  git push origin main"
else
    echo "❌ Aborted - no changes committed"
    git reset HEAD .
    exit 1
fi
