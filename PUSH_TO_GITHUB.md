# Pushing HALO_lib_header to GitHub

This guide shows how to push the repository to a remote GitHub repository.

## Prerequisites

1. **GitHub Account**: Create one at https://github.com if you don't have one
2. **Git Configured**: Set your name and email
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
3. **SSH Key** (recommended) or **GitHub Personal Access Token**

## Step-by-Step Instructions

### Option 1: Create a NEW Repository (Recommended)

#### 1. Create Repository on GitHub
1. Go to https://github.com/new
2. Enter repository name: `HALO_lib_header`
3. Enter description:
   ```
   Header-only library for halo-pattern GPU data exchange using MPI and NCCL
   ```
4. Choose **Public** or **Private**
5. Do NOT initialize with README, .gitignore, or license
6. Click "Create repository"

#### 2. Add Remote and Push
```bash
cd /home/ceoas/chenchon/FEZ/HALO_lib_header

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/HALO_lib_header.git

# Verify remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option 2: Use SSH (If SSH Key Configured)

```bash
cd /home/ceoas/chenchon/FEZ/HALO_lib_header

# Add remote with SSH
git remote add origin git@github.com:YOUR_USERNAME/HALO_lib_header.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option 3: Use GitHub CLI

If you have `gh` installed:

```bash
cd /home/ceoas/chenchon/FEZ/HALO_lib_header

# Create and push in one command
gh repo create HALO_lib_header --public --source=. --remote=origin --push
```

## Handling Authentication

### HTTPS with Personal Access Token

1. **Generate Token**: https://github.com/settings/tokens/new
   - Click "Generate new token"
   - Select scopes: `repo` (full control of private repositories)
   - Copy the token

2. **Use Token**:
   ```bash
   git push -u origin main
   # When prompted for password, paste your token
   ```

3. **Store Credentials** (optional):
   ```bash
   git config --global credential.helper store
   git push -u origin main
   # Credentials will be saved for future use
   ```

### SSH Key Setup

1. **Generate SSH Key** (if not already done):
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```

2. **Add to SSH Agent**:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. **Add to GitHub**: https://github.com/settings/ssh/new
   - Copy content of `~/.ssh/id_ed25519.pub`
   - Paste into GitHub SSH keys

4. **Test Connection**:
   ```bash
   ssh -T git@github.com
   ```

## Verify Successful Push

After pushing, verify everything is on GitHub:

```bash
# Check remote status
git remote -v

# Check current branch
git branch

# View log
git log --oneline -5
```

Visit `https://github.com/YOUR_USERNAME/HALO_lib_header` to see your repository.

## Repository Contents After Push

Your GitHub repository should contain:
- `.gitignore` - Excludes build files
- `BUG_FIX.md` - Bug fixes documentation
- `CLAUDE.md` - Project overview and architecture
- `CMakeLists.txt` - Build configuration
- `CONTRIBUTING.md` - Contribution guidelines
- `README.md` - Quick start and features
- `halo_lib.hpp` - Header-only library
- `main.cu` - Test application
- `PUSH_TO_GITHUB.md` - This file

## Repository Structure Visualization

```
HALO_lib_header/
├── .git/                 (auto-generated)
├── .gitignore
├── BUG_FIX.md
├── CLAUDE.md
├── CMakeLists.txt
├── CONTRIBUTING.md
├── README.md
├── halo_lib.hpp          (main library, ~600 lines)
├── main.cu               (test app, ~430 lines)
└── PUSH_TO_GITHUB.md     (this file)
```

## After Pushing - Recommended Next Steps

### 1. Add License
```bash
# Add MIT License (optional)
curl https://opensource.org/licenses/MIT > LICENSE
git add LICENSE
git commit -m "Add MIT License"
git push
```

### 2. Create Issues for Future Work
- GitHub Issues → New Issue
- Add labels: `enhancement`, `bug`, `documentation`

### 3. Set Up GitHub Pages (Optional)
- Settings → Pages
- Choose "docs" folder or main branch
- Auto-generate site from README.md

### 4. Add Badges to README
Example badges:
```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-7.0%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI%2FMPICH-blue)](https://www.mpi-forum.org/)
```

## Troubleshooting

### Remote Already Exists
```bash
# Remove existing remote
git remote remove origin

# Add correct one
git remote add origin https://github.com/YOUR_USERNAME/HALO_lib_header.git
```

### Push Rejected
```bash
# Check branch name
git branch

# If not 'main', rename it
git branch -M main

# Try push again
git push -u origin main
```

### Authentication Failed
1. **For HTTPS**: Use personal access token instead of password
2. **For SSH**: Verify SSH key is added to GitHub
3. **Check**: `git config --list` to verify settings

### Wrong Username/Email
```bash
# View current settings
git config --global user.name
git config --global user.email

# Update if needed
git config --global user.name "New Name"
git config --global user.email "new.email@example.com"
```

## Making First Contribution After Push

Once pushed, contribute like this:

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes
# ... edit files ...

# Commit changes
git add .
git commit -m "Add your feature description"

# Push to GitHub
git push -u origin feature/your-feature

# Create Pull Request on GitHub UI
```

## Keeping Repository Updated

```bash
# Fetch updates
git fetch origin

# Pull latest
git pull origin main

# Check for changes
git log --oneline -5
```

## Useful Git Commands

```bash
# View remotes
git remote -v

# Change remote URL
git remote set-url origin NEW_URL

# Remove remote
git remote remove origin

# Show current branch
git branch

# List all branches (including remote)
git branch -a

# View unpushed commits
git log --branches --not --remotes

# Sync fork with upstream (if you fork a repo)
git fetch upstream
git rebase upstream/main
```

## Common Workflow

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/HALO_lib_header.git
cd HALO_lib_header

# 2. Create feature branch
git checkout -b feature/awesome-feature

# 3. Make changes
# ... edit files ...

# 4. Stage and commit
git add .
git commit -m "Add awesome feature"

# 5. Push to GitHub
git push -u origin feature/awesome-feature

# 6. Create Pull Request via GitHub UI
```

## For Collaborative Development

If working with team members:

1. **Invite Collaborators**: Settings → Collaborators
2. **Use Branches**: Everyone creates separate branches
3. **Use Pull Requests**: Review before merging to main
4. **Set Branch Protection**: Settings → Branches
   - Require PR reviews
   - Require status checks
   - Dismiss stale PR approvals

---

## Summary

**Quick Start** (5 minutes):
1. Create GitHub repo at https://github.com/new
2. Run:
   ```bash
   cd /home/ceoas/chenchon/FEZ/HALO_lib_header
   git remote add origin https://github.com/YOUR_USERNAME/HALO_lib_header.git
   git branch -M main
   git push -u origin main
   ```
3. Done! Repository is on GitHub

For questions: See [CONTRIBUTING.md](CONTRIBUTING.md) or open a GitHub Issue.
