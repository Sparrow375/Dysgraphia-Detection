# Data Version Control (DVC) & Cloud Sync Guide

This project decouples source code (tracked via Git in branch `baseline-v1`) from large multimodal datasets (multi-gigabyte handwriting scans, stylus renders, and trajectory kinematics).

## Architecture Overview

```
                      +---------------------------------------+
                      |       GitHub Repository (Git)         |
                      |     Branch: baseline-v1               |
                      |                                       |
                      |  - Code & ML pipelines                |
                      |  - Configuration & manifests          |
                      |  - data.dvc (tiny pointer, ~6 lines)  |
                      |  - .gitignore (excludes data/)        |
                      +-------------------+-------------------+
                                          |
                                dvc pull / dvc push
                                          |
               +--------------------------+--------------------------+
               |                                                     |
               v                                                     v
+-------------------------------+             +----------------------------------+
|    Local DVC Storage Remote   |             |   Google Cloud Storage (GCS)     |
| (F:\...\VV data\dvc_storage)  |             |   gs://<bucket-name>/dataset     |
|                               |             |                                  |
| Instant local dev sync on PC  |             | Team & Cloud collaboration sync  |
+-------------------------------+             +----------------------------------+
```

## How It Works

1. **Lightweight Git Repository**:
   - Git tracks all code, configs, model definitions, and the `.dvc` tracking files.
   - The entire `data/` folder is gitignored. Large image and CSV files are never committed to Git.
2. **Deterministic Data Pointers**:
   - `data.dvc` contains a cryptographic hash (MD5) representing the exact state of all files in `data/`.
   - When switching Git branches (e.g. from `baseline-v1` to another experiment), running `dvc checkout` immediately synchronizes `data/` to the exact dataset version paired with that branch commit.
3. **Dual-Remote Storage**:
   - **Local Remote (`local-storage`)**: Points to `F:\Avaneesh\download\VV data-20261006T155534Z-1-001\VV data\dvc_storage`. Fast, zero internet needed, no cloud egress fees during local iteration.
   - **Google Cloud Storage Remote (`gcs-storage`)**: For cloud backups, team sync, Colab, or cloud training.

---

## Daily Workflow Cheatsheet

### 1. Activating Environment
In the root of `Dysgraphia-Detection`:
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Checking Dataset Status
```bash
dvc status
```

### 3. Pulling Dataset (Sync from Remote)
To pull dataset files into `data/`:
```bash
# Pull from default remote (local storage)
dvc pull

# Or pull directly from Google Cloud Storage
dvc pull -r gcs-storage
```

### 4. Making Changes to the Dataset
If you add new handwriting scans, preprocess new folds, or update manifests:
```bash
# 1. Update DVC tracking
dvc add data

# 2. Push heavy files to storage remote(s)
dvc push                # pushes to local-storage
dvc push -r gcs-storage # pushes to Google Cloud Storage (when configured)

# 3. Commit pointer file to Git
git add data.dvc
git commit -m "Update dataset: added new student splits"
git push origin baseline-v1
```

### 5. Switching Git Branches
When you switch Git branches, your dataset seamlessly follows:
```bash
git checkout <another-branch>
dvc checkout
```

---

## Configuring Google Cloud Storage (GCS) Remote

To connect your Google Cloud Storage bucket:

### Step 1: Create a GCS Bucket
In your Google Cloud Console (or using `gcloud` / Cloud Storage UI):
Create a bucket, e.g., `gs://dysgraphia-dataset-storage/data`.

### Step 2: Add GCS Remote to DVC
```powershell
dvc remote add gcs-storage gs://dysgraphia-dataset-storage/data
```

### Step 3: Authenticate with Google Cloud
DVC uses standard Google Cloud Application Default Credentials (ADC) or a service account key:
```bash
# Option A: User login (via gcloud CLI)
gcloud auth application-default login

# Option B: Service Account JSON key
dvc remote modify --local gcs-storage credentialpath "C:\path\to\service-account-key.json"
```

### Step 4: Push / Pull to Google Cloud
```bash
# Push your local dataset to GCS
dvc push -r gcs-storage

# Or make GCS the default remote if working across multiple machines:
dvc remote default gcs-storage
dvc push
```
