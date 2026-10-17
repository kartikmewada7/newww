# Face Recognition Learning Lab

A standalone Windows / Python 3.11 computer vision and machine learning prototype for exploring face detection, 512-D ArcFace embeddings, teacher-confirmed feedback, versioned SVC classification, and held-out evaluation using the Kaggle Face Recognition Dataset.

> [!NOTE]
> This is an independent educational learning lab. It is completely isolated and does not connect to AttendAI, Supabase, or external services.

---

## 1. Environment & Architecture Overview

- **Operating System:** Windows 10/11 (AMD64)
- **Python:** 3.11 in virtual environment (`.venv`)
- **Face Analysis Engine:** InsightFace `buffalo_l`
  - Detection: RetinaFace / SCRFD (`det_10g.onnx`, input size 640×640)
  - Recognition: ArcFace ResNet50 (`w600k_r50.onnx`, output 512-D normalized embedding)
- **Runtime:** ONNX Runtime GPU (`CUDAExecutionProvider`) with automatic fallback to CPU (`CPUExecutionProvider`)
- **Classifier:** Versioned Scikit-Learn Support Vector Classifier (`sklearn.svm.SVC`, RBF kernel, balanced weights)
- **UI:** Streamlit (`app.py`)

### Biometric Decision Architecture

Identity prediction uses a **dual-gate consensus mechanism**:

```
Input Image 
  │
  ▼
[ Face Detection (RetinaFace 640x640) ]
  │ (Rejects if face < 40px or no face)
  ▼
[ Embedding Extraction (ArcFace 512-D L2-Normalized) ]
  │
  ├───► [ Cosine Similarity Gate ] ───► If max_sim < 0.45 ───► REJECT: "Unknown"
  │                                ───► If (best - 2nd) < 0.05 ──► FLAG: "Review"
  │
  ├───► [ Versioned SVC Classifier ] ──► Predicts class label
  │
  ▼
[ Decision Consensus ] ──► If Classifier agrees with Nearest Memory ──► ACCEPT: "Matched"
                        ──► If Classifier disagrees with Memory ──────► FLAG: "Review"
```

> [!IMPORTANT]
> **Pretrained Weights vs Classifier:** The pretrained `buffalo_l` neural network weights are **frozen**; they are not fine-tuned. The learning pipeline stores verified embeddings in memory and trains/versions the scikit-learn classifier. This provides fast, incremental retraining without catastrophic forgetting.

---

## 2. Setup & Virtual Environment

Open PowerShell in `D:\recognize`:

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 3. GPU and CPU Verification

On Windows Python 3.8+, dependent DLLs inside `site-packages/nvidia` (such as `cublasLt64_13.dll` and `cudnn64_9.dll`) must be added to the process DLL directory table via `os.add_dll_directory()`. This is automatically configured by `face_engine.configure_nvidia_dll_paths()`.

To verify your execution provider and session status:

```powershell
python gpu_check.py
```

Expected output:
- **Available Providers:** `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- **Active Session Providers:** `['CUDAExecutionProvider', 'CPUExecutionProvider']`
- **GPU session acceleration active:** `True`
- **Device:** `GPU (CUDAExecutionProvider)`

*If no NVIDIA GPU or compatible driver is found, the system gracefully falls back to `CPUExecutionProvider`.*

---

## 4. Kaggle Dataset Import Pipeline

### Dataset Structure

The dataset is located at:
`C:\Users\KARTIK MEWADA\.cache\kagglehub\datasets\vasukipatel\face-recognition-dataset\versions\1`

- **31 celebrity identities** (e.g., Akshay Kumar, Alia Bhatt, Brad Pitt, Robert Downey Jr, Tom Cruise, etc.)
- **2,562 total source images** (30 to 120 images per identity)
- **Original Images:** `Original Images\Original Images\<Identity>\<Identity>_X.jpg` (Full-resolution images where faces are reliably detected by `det_size=(640, 640)`)
- **Faces:** `Faces\Faces\` (Cropped 160×160 thumbnails—these are skipped for detection because detector requires context)

### Importer Commands

#### A. Dry-Run (Verification Only)
Inspects the CSV and directory structure without modifying any files:
```powershell
python import_kaggle_dataset.py --dry-run
```

#### B. Quick Validation Sample Run
Processes 2 images per identity to verify extraction, splitting, training, and evaluation in seconds:
```powershell
python import_kaggle_dataset.py --sample-limit 2 --reset-dataset-only
```

#### C. Full Dataset Import & Training
Processes all 2,562 original images, performs a 70% Train / 15% Validation / 15% Test stratified split, retrains the SVC classifier, and outputs benchmark metrics:
```powershell
python import_kaggle_dataset.py --reset-dataset-only
```

#### Command Options:
- `--dataset-dir <path>`: Custom path to dataset root.
- `--reset-dataset-only`: Safely removes previous Kaggle benchmark records while strictly preserving manual student registrations.
- `--seed 42`: Fixed seed for reproducible stratified train/validation/test splits.
- `--train-ratio 0.70`: Training split fraction.
- `--val-ratio 0.15`: Validation split fraction.
- `--test-ratio 0.15`: Held-out test split fraction.
- `--sample-limit <N>`: Process at most N images per identity.
- `--num-unknown-identities <K>`: Reserve K identities as unknown distractors for open-set FAR/FRR testing.

---

## 5. Evaluation & Threshold Calibration

A dedicated evaluation module is available to benchmark the classifier on validation or held-out test data:

```powershell
# Evaluate on held-out test split
python evaluation.py --split test

# Evaluate on validation split
python evaluation.py --split val

# Sweep similarity thresholds & margins on validation data
python evaluation.py --sweep
```

### Benchmark Results on Kaggle Dataset (31 Identities):

- **Discovered Source Images:** 2,562
- **Successfully Processed Faces:** 2,519 (98.3%)
- **Skipped Images:**
  - 6 images (no face detected)
  - 37 images (multiple faces detected in frame)
  - 0 processing errors
- **Split Breakdown:**
  - Training Set: 1,757 samples (31 identities) + 28 manual student records = 1,785 samples (32 classes)
  - Validation Set: 381 samples (held-out)
  - Held-out Test Set: 381 samples (held-out)
- **Top-1 Identification Accuracy:** 100.00%
- **Macro Precision / Recall / F1:** 1.0000 / 1.0000 / 1.0000
- **Threshold Calibration (Validation Set):**
  - Threshold 0.45: 100% matched, 0% false rejection
  - Threshold 0.55: 100% matched, 0% false rejection
  - Threshold 0.60: 99.2% matched, 0.8% rejected as unknown
  - Threshold 0.65: 96.3% matched, 3.7% rejected as unknown
- **Intra-identity similarity:** Mean 0.75 (range 0.64 – 0.99)
- **Inter-identity similarity:** Mean -0.04 (max 0.05)

---

## 6. Running Automated Tests

Run the complete test suite:

```powershell
python test_suite.py
```

Tests verify:
1. DLL auto-discovery and ONNX Runtime session providers
2. 512-D cosine similarity calculations and dimension guards
3. Face detection and normalized embedding extraction on sample images
4. Decision gating (matched, unknown rejection, ambiguous review)
5. Dataset-safe reset preserving manual registrations
6. Evaluation metrics computation (Accuracy, F1, Confusion Matrix, FAR/FRR)

---

## 7. Launching the Streamlit Application

Start the web UI:

```powershell
streamlit run app.py
```

The application provides:
- **Sidebar:** Live ONNX Runtime provider status (CUDA vs CPU fallback), classifier version, memory sample breakdown.
- **Tab 1 (Register / Add Memory):** Register student face captures into verified training memory.
- **Tab 2 (Group Photo + Feedback):** Upload group photos, preview detections, submit teacher feedback to retrain incrementally.
- **Tab 3 (Held-out Evaluation):** Run evaluation across the held-out test or validation splits, view per-identity accuracy and error filters.
- **Tab 4 (Model Versions & Dataset):** Inspect model metadata, view Kaggle benchmark status, and perform safe dataset resets without losing manual student records.

---

## 8. Managing and Resetting Dataset Artifacts

To safely remove Kaggle benchmark embeddings without deleting your manual registrations:
1. In the Streamlit app (Tab 4): Check the confirmation box and click **Reset Kaggle Dataset Records**.
2. Or via Python:
   ```python
   from storage import reset_dataset_records
   reset_dataset_records(preserve_user_data=True)
   ```
This removes only records with `source="kaggle_dataset"` and automatically retrains the classifier on the remaining user memories.

---

## 9. LFW Dataset Experiment (Isolated Benchmark)

The project includes an isolated pipeline for benchmarking on the **Labeled Faces in the Wild (LFW)** dataset (`jessicali9530/lfw-dataset`).

### Strict Data & Model Isolation

All LFW artifacts are stored in separate files and **never modify** production memory or current models:
- Training Memory: `data/lfw_memory.json`
- Validation Set: `data/lfw_validation.json`
- Held-Out Test Set: `data/lfw_evaluation.json`
- Full Manifest: `data/lfw_manifest.json`
- Evaluation Metrics: `data/lfw_evaluation_metrics.json`
- Classifier Artifact: `models/lfw_classifier.joblib`
- Classifier Metadata: `models/lfw_model.json`

### Closed-Set Classification vs. Verification Protocol

LFW contains 13,233 images across 5,749 identities. However, over 4,000 identities have only a single photo.
- **Verification Protocol (LFW Standard):** Classifies whether a pair of images belongs to the same person.
- **Identification Protocol (Our Closed-Set Classifier):** Trains an SVC classifier to identify individuals among known classes. For this protocol, identities must have multiple samples to support train/val/test splits.
- The pipeline provides `--min-images <N>` (default: 10):
  - With `--min-images 10`: 158 identities, 4,324 images.
  - With `--min-images 20`: 62 identities, 3,023 images.
  - Ineligible identities are explicitly recorded in `lfw_manifest.json` as `skipped_insufficient_samples` and are not silently discarded.

### Commands for LFW Pipeline

```powershell
# 1. Dry run: scan dataset and print identity distribution
python import_lfw_dataset.py --dry-run

# 2. Small sample validation run (2 images per identity)
python import_lfw_dataset.py --sample-limit 2

# 3. Full LFW import and training (identities with >= 10 images)
python import_lfw_dataset.py --min-images 10

# 4. Evaluate LFW held-out test set
python evaluation.py --dataset lfw --split test

# 5. Sweep thresholds on LFW validation set
python evaluation.py --dataset lfw --sweep

# 6. Run LFW automated unit tests
python test_lfw.py
```

### Streamlit Integration (Tab 5)

Visit Tab 5 in `streamlit run app.py` to:
- Inspect LFW dataset and classifier status
- View sample counts across train, validation, and test splits
- Run interactive evaluation on the held-out test or validation splits
- Filter predictions by errors, ambiguous reviews, or matches
- Inspect per-identity classification accuracy and manifest statistics

