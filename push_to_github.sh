#!/usr/bin/env bash
# =============================================================================
# push_to_github.sh
# =============================================================================
# One-click script to:
#   1. Create the GitHub repository via API
#   2. Initialise a local Git repo
#   3. Add all files and push to GitHub
#
# USAGE:
#   1. Set your GitHub username and Personal Access Token below
#   2. chmod +x push_to_github.sh
#   3. ./push_to_github.sh
#
# How to create a GitHub Personal Access Token (PAT):
#   → GitHub.com → Settings → Developer settings
#     → Personal access tokens → Tokens (classic)
#     → Generate new token → select scopes: [repo] → Generate token
#   Copy the token and paste it below.
# =============================================================================

# ── FILL THESE IN ─────────────────────────────────────────────────────────────
GITHUB_USERNAME="lokendra712"
GITHUB_TOKEN=""  # Leave empty to use git credential helper or SSH
REPO_NAME="maize-yield-nepal-ml"
REPO_DESCRIPTION="ML & Remote Sensing assessment of climate variability impacts on maize yield stability in Nepal (1990-2022). Random Forest, SHAP, NDVI, Eberhart-Russell stability. Q1 journal paper code."
# ─────────────────────────────────────────────────────────────────────────────

set -e   # Exit immediately on any error

# Colour codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}  Nepal Maize ML — GitHub Repository Setup ${NC}"
echo -e "${YELLOW}============================================${NC}"

# Validate inputs
if [ "$GITHUB_USERNAME" = "YOUR_GITHUB_USERNAME" ] || \
   [ "$GITHUB_TOKEN" = "YOUR_PERSONAL_ACCESS_TOKEN" ]; then
    echo -e "${RED}ERROR: Please set GITHUB_USERNAME and GITHUB_TOKEN in this script.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Step 1: Creating GitHub repository '${REPO_NAME}'...${NC}"
HTTP_CODE=$(curl -s -o /tmp/gh_create_response.json -w "%{http_code}" \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user/repos \
  -d "{
    \"name\": \"${REPO_NAME}\",
    \"description\": \"${REPO_DESCRIPTION}\",
    \"public\": true,
    \"has_issues\": true,
    \"has_wiki\": false,
    \"auto_init\": false
  }")

if [ "$HTTP_CODE" = "201" ]; then
    REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    CLONE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    echo -e "  ✅ Repository created: ${REPO_URL}"
elif [ "$HTTP_CODE" = "422" ]; then
    echo -e "  ${YELLOW}⚠  Repository already exists — pushing to existing repo.${NC}"
    CLONE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
else
    echo -e "${RED}ERROR: GitHub API returned HTTP ${HTTP_CODE}${NC}"
    cat /tmp/gh_create_response.json
    exit 1
fi

echo -e "\n${GREEN}Step 2: Initialising local Git repository...${NC}"
git init
git checkout -b main 2>/dev/null || git checkout main

echo -e "\n${GREEN}Step 3: Staging all files...${NC}"
git add .
git status --short | head -30
echo "  (showing first 30 files)"

echo -e "\n${GREEN}Step 4: Creating initial commit...${NC}"
git config user.email "lokendrapaudel@example.com"
git config user.name  "${GITHUB_USERNAME}"
git commit -m "🌽 Initial commit: Nepal maize-climate ML paper code & figures

- Full analysis pipeline: data prep → EDA → ML training → SHAP → stability
- 5 models: Random Forest, Gradient Boosting, SVR, Lasso, OLS
- Best RF: R²=0.887, RMSE=0.248 t ha⁻¹, NSE=0.883 (test set)
- SHAP beeswarm + dependence plots
- Eberhart-Russell stability analysis (R/metan)
- All 6 publication figures (300 DPI)
- Synthetic demo dataset for reproducibility
- Target journal: Computers and Electronics in Agriculture (Q1)"

echo -e "\n${GREEN}Step 5: Pushing to GitHub...${NC}"
if [ -n "$GITHUB_TOKEN" ]; then
    PUSH_URL="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@$(echo $CLONE_URL | sed 's|https://||')"
    git remote add origin "${PUSH_URL}" 2>/dev/null || \
      git remote set-url origin "${PUSH_URL}"
else
    # Use standard URL; git will use credential manager or SSH
    git remote add origin "${CLONE_URL}" 2>/dev/null || \
      git remote set-url origin "${CLONE_URL}"
fi
git push -u origin main

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}  ✅ SUCCESS!                               ${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Repository URL : ${REPO_URL}"
echo -e "  Clone with     : git clone https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
echo ""
echo -e "  ${YELLOW}Next steps:${NC}"
echo -e "  1. Add a repository description/topics on GitHub"
echo -e "  2. Upload raw data to GitHub Releases or Zenodo"
echo -e "  3. Add the repository URL to your manuscript"
echo -e "     (replace placeholder in manuscript references)"
echo ""
