# Git Large File Storage (Git LFS) Guide

All multi-gigabyte handwriting scans, stylus renders, kinematics logs, and student data are versioned and stored on GitHub using **Git LFS**.

## Architecture & How It Works

```
                      +------------------------------------------+
                      |         GitHub Repository (Git)          |
                      |           Branch: baseline-v1            |
                      |                                          |
                      |  - Python code, configs, feature models  |
                      |  - .gitattributes (tracks data/**)       |
                      |  - Lightweight LFS pointers (3 lines)    |
                      +--------------------+---------------------+
                                           |
                              git push / git lfs pull
                                           |
                                           v
                      +------------------------------------------+
                      |          GitHub LFS Cloud Store          |
                      |  github.com/Sparrow375/Dysgraphia-...    |
                      |                                          |
                      |  - 647 files (1.34 GB) uploaded          |
                      |  - Full School A dataset                 |
                      +------------------------------------------+
```

## Daily Workflow for Developers & Collaborators

### 1. First-time Setup
Ensure Git LFS is installed and registered on your machine:
```bash
git lfs install
```

### 2. Cloning the Repo with Dataset
```bash
# Clone the baseline branch
git clone -b baseline-v1 https://github.com/Sparrow375/Dysgraphia-Detection.git
cd Dysgraphia-Detection

# Pull the heavy dataset files from GitHub LFS
git lfs pull
```

### 3. Adding or Updating Data
Whenever new handwriting scans, labels, or cohorts are added to `data/`:
```bash
# Git LFS automatically intercepts files inside data/ based on .gitattributes
git add data/
git commit -m "Update dataset: added new cohort data"
git push origin baseline-v1
```
Git will push the code commit to GitHub, and Git LFS will automatically upload the large data objects to GitHub's LFS cloud servers.

### 4. Useful Git LFS Commands
```bash
# Check which files are tracked by LFS
git lfs ls-files

# Check upload/download status
git lfs status

# Only fetch pointers without downloading multi-GB files (useful on slow internet)
git config lfs.fetchexclude "data/**"
```
