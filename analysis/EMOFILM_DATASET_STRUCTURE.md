# Emo-FilM Dataset Structure Documentation

**Location:** `/storage/bigdata/Emo-FilM/`
**Dataset Type:** BIDS-formatted fMRI dataset
**Version:** 1.0.1 (OpenNeuro: ds004892)

## Overview

Emo-FilM is a multimodal neuroimaging dataset for studying emotion processes during film watching. The dataset includes fMRI, physiological (cardiac/respiratory), and behavioral data collected while participants watched emotional film clips.

**Authors:** Elenor Morgenroth, Stefano Moia, Laura Vilaclara, Raphael Fournier, Michal Muszynski, Maria Ploumitsakou, Marina Almato Bellavista, Patrik Vuilleumier, Dimitri Van De Ville

**DOI:** 10.18112/openneuro.ds004892.v1.0.1

---

## Directory Structure

```
/storage/bigdata/Emo-FilM/
├── brain_data/              # fMRI BIDS dataset (main analysis data)
│   ├── participants.tsv     # Subject demographics and questionnaire scores
│   ├── participants.json    # Metadata for participants.tsv columns
│   ├── dataset_description.json
│   ├── sub-S01/            # 30 subjects (sub-S01 to sub-S30)
│   │   ├── sub-S01_sessions.tsv
│   │   ├── ses-1/          # Session 1 (4 sessions per subject)
│   │   │   ├── anat/       # Anatomical T1w/T2w scans
│   │   │   ├── func/       # Functional BOLD data
│   │   │   ├── fmap/       # Field maps for distortion correction
│   │   │   └── beh/        # Behavioral/timing data
│   │   ├── ses-2/
│   │   ├── ses-3/
│   │   └── ses-4/
│   └── derivatives/        # Preprocessed data (fMRIPrep, etc.)
│
└── annotations/            # Continuous emotion ratings during films
    ├── participants.tsv    # Rater demographics
    ├── sub-area/          # Film titles as "subject" IDs (47 films)
    │   └── beh/           # Continuous ratings (anxiety, calm, surprise, etc.)
    ├── sub-army/
    ├── sub-bath/
    └── ...
```

---

## Subjects

**N = 30 participants** (sub-S01 to sub-S30)
- **Age:** 18-34 years
- **Sex:** Mixed (M/F)
- **Missing:** sub-S12, sub-S18 (28 subjects total)

### Questionnaire Data

Available in `brain_data/participants.tsv`:

| Column | Description | Type |
|--------|-------------|------|
| `participant_id` | Subject ID (sub-S01, etc.) | string |
| `age` | Age in years | int |
| `sex` | Sex (M/F) | string |
| `DASS_dep` | DASS-21 Depression subscale | float |
| `DASS_anx` | DASS-21 Anxiety subscale | **float** |
| `DASS_str` | DASS-21 Stress subscale | float |
| `bis` | Behavioral Inhibition System | float |
| `bas_d` | BAS Drive | float |
| `bas_f` | BAS Fun Seeking | float |
| `bas_r` | BAS Reward Responsiveness | float |
| `BIG5_ext` | Big Five Extraversion | float |
| `BIG5_agr` | Big Five Agreeableness | float |
| `BIG5_con` | Big Five Conscientiousness | float |
| `BIG5_neu` | Big Five Neuroticism | float |
| `BIG5_ope` | Big Five Openness | float |
| `erq_cr` | Emotion Regulation Questionnaire - Cognitive Reappraisal | float |
| `erq_es` | Emotion Regulation Questionnaire - Expressive Suppression | float |
| `*Absorbed`, `*Enjoyed`, `*Interested` | Post-scan ratings for each film | float |

**Key for Anxiety Research:** Use `DASS_anx` (DASS-21 Anxiety subscale) as the primary anxiety trait measure.

---

## Sessions

Each subject completed **4 scanning sessions** across 4 days:

| Session | Description |
|---------|-------------|
| `ses-1` | Day 1: Rest + Film clips (BigBuckBunny, FirstBite, YouAgain) |
| `ses-2` | Day 2: Film clips |
| `ses-3` | Day 3: Film clips |
| `ses-4` | Day 4: Film clips |

---

## Functional Data (BOLD fMRI)

### File Naming Convention

```
sub-<subjectID>_ses-<sessionID>_task-<taskName>_bold.nii.gz
```

**Example:**
```
sub-S01_ses-1_task-Rest_bold.nii.gz
sub-S01_ses-1_task-BigBuckBunny_bold.nii.gz
```

### Task Types

1. **Resting-State:**
   - `task-Rest`
   - Duration: ~6-7 minutes
   - Eyes open/closed (check task JSON)

2. **Film Viewing Tasks:**
   - Multiple emotional film clips per session
   - Examples: `BigBuckBunny`, `Sintel`, `TearsOfSteel`, `FirstBite`, `Chatter`, etc.
   - Duration: Varies by film (3-12 minutes)

### Acquisition Parameters

From `sub-S01_ses-1_task-Rest_bold.json`:

| Parameter | Value |
|-----------|-------|
| Scanner | Siemens 3T |
| Magnetic Field | 3 Tesla |
| TR (Repetition Time) | **1.3 seconds** |
| Voxel Size | 2.0 x 2.0 x 2.0 mm³ (typical) |
| Slices | ~40 (check JSON) |
| Slice Order | Ascending/Interleaved (check JSON) |
| Volumes | ~300-600 per run (depends on film length) |

**Critical for Analysis:** `TR = 1.3 seconds`

---

## Anatomical Data

Location: `sub-<ID>/ses-<session>/anat/`

Files:
- `*_T1w.nii.gz` - T1-weighted anatomical scan
- `*_T2w.nii.gz` - T2-weighted anatomical scan (if available)
- `*.json` - Acquisition parameters

---

## Physiological Data

Location: `sub-<ID>/ses-<session>/func/`

Files: `*_physio.tsv.gz`

Includes:
- Cardiac (PPG/ECG)
- Respiratory traces
- Sampled at high frequency (check `*_physio.json`)

---

## Behavioral/Event Files

Location: `sub-<ID>/ses-<session>/func/`

Files: `*_events.tsv` + `*_events.json`

Contains:
- Film start/end times
- Event onsets
- Durations

---

## Annotations (Continuous Emotion Ratings)

**Location:** `/storage/bigdata/Emo-FilM/annotations/`

Independent raters continuously rated emotions while watching each film:

### Rating Dimensions

- `Anxiety` - Anxiety intensity
- `Calm` - Calmness
- `Surprise` - Surprise
- `Frown` - Facial frowning
- `Stomach` - Stomach sensations
- `Throat` - Throat sensations

### File Structure

```
annotations/sub-<filmname>/beh/sub-<filmname>_task-<filmtitle>_recording-<emotion>_stim.tsv.gz
```

**Example:**
```
annotations/sub-area/beh/sub-area_task-AfterTheRain_recording-Anxiety_stim.tsv.gz
```

Film titles used as "subject IDs" (e.g., `sub-area`, `sub-army`, etc.)

---

## Data Loading Example

### Using Python (nilearn)

```python
from pathlib import Path
import nibabel as nib
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

# Paths
data_root = Path('/storage/bigdata/Emo-FilM/brain_data')
subject_id = 'sub-S01'
session = 'ses-1'
task = 'Rest'  # or 'BigBuckBunny', etc.

# Load functional data
func_file = (data_root / subject_id / session / 'func' /
             f'{subject_id}_{session}_task-{task}_bold.nii.gz')
func_img = nib.load(func_file)

# Extract ROI timeseries with AAL atlas
atlas = datasets.fetch_atlas_aal()
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=1.3  # Critical: Emo-FilM TR
)
timeseries = masker.fit_transform(func_img)

# Load demographics/anxiety scores
participants = pd.read_csv(data_root / 'participants.tsv', sep='\t')
anxiety_scores = participants[['participant_id', 'DASS_anx']]

print(f"Timeseries shape: {timeseries.shape}")
print(f"N subjects with anxiety data: {len(participants)}")
```

---

## Key Files for Analysis

| File | Path | Description |
|------|------|-------------|
| **Subject list** | `brain_data/participants.tsv` | Demographics + questionnaires |
| **Anxiety scores** | Column `DASS_anx` in participants.tsv | Primary anxiety measure |
| **Resting fMRI** | `sub-*/ses-*/func/*_task-Rest_bold.nii.gz` | Resting-state scans |
| **Film fMRI** | `sub-*/ses-*/func/*_task-<film>_bold.nii.gz` | Task-based scans |
| **Anatomical** | `sub-*/ses-*/anat/*_T1w.nii.gz` | Structural scans |
| **Events** | `sub-*/ses-*/func/*_events.tsv` | Timing information |

---

## Data Access Notes

1. **File Links:** Many `.nii.gz` files are git-annex symlinks. Ensure git-annex content is fetched before analysis.

2. **Derivatives:** Preprocessed data available in `brain_data/derivatives/` (fMRIPrep, FreeSurfer, etc.)

3. **Quality Control:** Check `sub-*_ses-*_scans.tsv` for scan quality ratings

4. **Missing Data:** Some subjects may have missing sessions or tasks - always check file existence

---

## Citation

When using this dataset, cite:

> Morgenroth, E., Moia, S., Vilaclara, L., Fournier, R., Muszynski, M., Ploumitsakou, M., Almato Bellavista, M., Vuilleumier, P., Van De Ville, D. Emo-FilM: A multimodal neuroimaging dataset for studying emotion processes during film watching. (in prep)

Dataset DOI: `10.18112/openneuro.ds004892.v1.0.1`

---

## Analysis Pipeline Recommendations

### For Event Boundary Detection with Anxiety

1. **Load subject data:**
   - fMRI: `*_task-Rest_bold.nii.gz` or film tasks
   - TR = 1.3 seconds
   - Extract ROI timeseries (AAL or Schaefer atlas)

2. **Load anxiety scores:**
   - Use `DASS_anx` from `participants.tsv`
   - Filter subjects with valid anxiety data

3. **Preprocess:**
   - Standardize timeseries
   - Detrend
   - Bandpass filter (0.01-0.1 Hz for resting-state)

4. **Run HMM boundary detection:**
   - Fit HMM on ROI timeseries
   - Detect state transitions (event boundaries)
   - Compute boundary metrics (count, rate, etc.)

5. **Statistical analysis:**
   - Correlate anxiety scores with boundary metrics
   - Control for age, sex if needed
   - Visualize results

---

## Quick Start Script

See [`run_quick_test.sh`](./run_quick_test.sh) and [`run_hmm_emofilm.py`](./run_hmm_emofilm.py) for a complete analysis pipeline.

```bash
# Test with 5 subjects
bash run_quick_test.sh

# Full analysis
python run_hmm_emofilm.py --data_root /storage/bigdata/Emo-FilM/brain_data \
                          --session rest \
                          --atlas aal \
                          --output_dir results_emofilm/
```

---

**Last Updated:** 2025-11-01
**Contact:** castella@connectome (lab server)
