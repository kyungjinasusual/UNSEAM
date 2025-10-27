# Traditional Methodology Proposals for Anxiety-Event Boundary Research

**Author**: Replication Engineer Agent + Supervisor Coordination
**Date**: 2025-10-27
**Purpose**: Propose conventional analysis approaches for investigating anxiety effects on event boundary detection

---

## Executive Summary

This document proposes five traditional methodological approaches for examining how anxiety modulates event boundary detection in fMRI data. Each method is evaluated for:

1. **Feasibility** - Data requirements and computational demands
2. **Theoretical Justification** - Alignment with research hypotheses
3. **Expected Outcomes** - Predicted results and interpretability
4. **Advantages** - Methodological strengths
5. **Disadvantages** - Limitations and potential issues
6. **Implementation Details** - Practical analysis pipeline

**Recommendation**: Multi-method convergent approach combining HMM (primary), GSBS (validation), and seed-based connectivity (mechanistic interpretation).

---

## Method 1: Hidden Markov Model (HMM) for State Detection

### Theoretical Rationale

Hidden Markov Models provide a probabilistic framework for detecting latent brain states and their transitions. Event boundaries correspond to state transitions where neural activity patterns shift discontinuously.

**Hypothesis Application**:
- **H1**: High anxiety → more state transitions (more boundaries)
- **H3**: Anxiety modulates state transition probabilities
- **H4**: Different states show distinct connectivity patterns

### Method Description

**Step 1: Data Preparation**

Input: Preprocessed resting-state fMRI (time × voxels)

Region extraction:
```python
# Extract time series from ROIs (e.g., AAL atlas)
from nilearn.input_data import NiftiLabelsMasker

masker = NiftiLabelsMasker(
    labels_img='AAL116.nii',
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

roi_time_series = masker.fit_transform(fmri_img)
# Shape: (n_timepoints, n_regions=116)
```

**Step 2: HMM Model Specification**

Model components:
- **K hidden states**: S₁, S₂, ..., Sₖ (k = 3-10, cross-validated)
- **Transition matrix A**: P(Sₜ₊₁ | Sₜ)
- **Emission**: Multivariate Gaussian for each state
- **Initial state probabilities**: π

```python
from hmmlearn import hmm
import numpy as np

# Determine optimal number of states via cross-validation
def select_n_states(X, state_range=range(3, 11)):
    """
    Cross-validation for HMM state number selection.
    """
    from sklearn.model_selection import KFold

    scores = []
    for n_states in state_range:
        kf = KFold(n_splits=5, shuffle=False)
        fold_scores = []

        for train_idx, test_idx in kf.split(X):
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=1000,
                random_state=42
            )
            model.fit(X[train_idx])
            fold_scores.append(model.score(X[test_idx]))

        scores.append(np.mean(fold_scores))

    optimal_n = state_range[np.argmax(scores)]
    return optimal_n, scores

# Fit HMM
optimal_k, cv_scores = select_n_states(roi_time_series)

model = hmm.GaussianHMM(
    n_components=optimal_k,
    covariance_type='full',
    n_iter=1000,
    random_state=42
)
model.fit(roi_time_series)

# Decode state sequence
state_sequence = model.predict(roi_time_series)
```

**Step 3: Boundary Detection**

Event boundaries = timepoints where state changes:

```python
def detect_boundaries(state_sequence):
    """
    Detect boundaries from HMM state sequence.
    """
    boundaries = []
    for t in range(1, len(state_sequence)):
        if state_sequence[t] != state_sequence[t-1]:
            boundaries.append(t)
    return np.array(boundaries)

boundaries = detect_boundaries(state_sequence)

# Compute boundary metrics
n_boundaries = len(boundaries)
mean_state_duration = np.mean(np.diff(np.concatenate([[0], boundaries, [len(state_sequence)]])))
boundary_rate = n_boundaries / (len(state_sequence) * TR)  # boundaries per second
```

**Step 4: Group Comparison**

```python
import pandas as pd
from scipy import stats

# Collect metrics for all subjects
results = pd.DataFrame({
    'subject': subject_ids,
    'anxiety_score': stai_t_scores,
    'n_boundaries': boundary_counts,
    'boundary_rate': boundary_rates,
    'mean_state_duration': state_durations
})

# Correlation analysis
r, p = stats.pearsonr(results['anxiety_score'], results['n_boundaries'])
print(f"Correlation STAI-T × Boundary Count: r={r:.3f}, p={p:.4f}")

# Group comparison (median split)
high_anxiety = results[results['anxiety_score'] >= results['anxiety_score'].median()]
low_anxiety = results[results['anxiety_score'] < results['anxiety_score'].median()]

t_stat, p_val = stats.ttest_ind(
    high_anxiety['n_boundaries'],
    low_anxiety['n_boundaries']
)
print(f"High vs Low Anxiety: t={t_stat:.2f}, p={p_val:.4f}")

# Multiple regression controlling for confounds
from statsmodels.formula.api import ols

model = ols(
    'n_boundaries ~ anxiety_score + age + C(sex) + mean_fd + depression_score',
    data=results
).fit()
print(model.summary())
```

**Step 5: State Characterization**

Analyze neural patterns of each state:

```python
# Extract state-specific connectivity
def compute_state_connectivity(roi_timeseries, state_sequence, state_id):
    """
    Compute functional connectivity for a specific state.
    """
    state_timepoints = (state_sequence == state_id)
    state_data = roi_timeseries[state_timepoints]
    conn_matrix = np.corrcoef(state_data.T)
    return conn_matrix

# Compare state connectivity between anxiety groups
for state_id in range(optimal_k):
    high_anx_conn = np.mean([
        compute_state_connectivity(ts, seq, state_id)
        for ts, seq, anx in zip(all_timeseries, all_sequences, anxiety_scores)
        if anx >= median_anxiety
    ], axis=0)

    low_anx_conn = np.mean([
        compute_state_connectivity(ts, seq, state_id)
        for ts, seq, anx in zip(all_timeseries, all_sequences, anxiety_scores)
        if anx < median_anxiety
    ], axis=0)

    # Statistical comparison (NBS for network-based statistics)
    diff_matrix = high_anx_conn - low_anx_conn
```

### Expected Outcomes

**Predicted Results**:

| Metric | High Anxiety | Low Anxiety | Effect Size | p-value |
|--------|--------------|-------------|-------------|---------|
| Boundary Count | 28.5 ± 6.3 | 21.2 ± 4.7 | d = 1.29 | < 0.01 |
| Boundary Rate (/min) | 2.85 ± 0.63 | 2.12 ± 0.47 | d = 1.29 | < 0.01 |
| Mean State Duration (s) | 22.4 ± 5.8 | 30.2 ± 7.1 | d = -1.19 | < 0.01 |
| Optimal K (states) | 7.2 ± 1.8 | 5.8 ± 1.4 | - | 0.08 |

**Correlation**:
- STAI-T × Boundary Count: r = 0.42, p < 0.001
- STAI-S × Boundary Rate: r = 0.36, p < 0.01

**State Connectivity**:
- State 1 (DMN-dominant): High anxiety shows reduced within-DMN connectivity
- State 2 (Salience-dominant): High anxiety shows elevated activation
- State 3 (Transitional): High anxiety more frequently occupies transitional states

### Advantages

1. **Data-Driven**: No need for predefined boundaries
2. **Probabilistic Framework**: Uncertainty quantification built-in
3. **Established Method**: Validated in multiple neuroscience studies (Baldassano et al., 2017)
4. **Interpretable States**: Can characterize neural patterns of each state
5. **Works with Resting-State**: No task required
6. **Individual Differences**: Can examine person-specific dynamics

### Disadvantages

1. **Model Selection**: Choosing K (number of states) is non-trivial
   - Solution: Cross-validation, BIC/AIC criteria

2. **Markov Assumption**: Assumes future states depend only on current state
   - May miss long-range dependencies
   - Solution: Higher-order HMM or hierarchical HMM

3. **Computational Cost**: Fitting can be slow for large K or many voxels
   - Solution: Use ROIs instead of whole-brain voxels

4. **Local Optima**: EM algorithm can converge to local maxima
   - Solution: Multiple random initializations, select best

5. **State Interpretation**: States may not have clear neurobiological meaning
   - Solution: Post-hoc characterization via connectivity, spatial patterns

### Implementation Timeline

- **Week 1-2**: Data preprocessing, ROI extraction
- **Week 3-4**: HMM parameter optimization, model fitting
- **Week 5-6**: Boundary detection, group comparisons
- **Week 7-8**: State characterization, connectivity analysis
- **Week 9-10**: Statistical analysis, visualization

**Total**: 10 weeks for complete analysis

### Software Requirements

```python
# Required packages
hmmlearn==0.3.0
scikit-learn>=1.0
nilearn>=0.9
statsmodels>=0.13
scipy>=1.7
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
```

---

## Method 2: Greedy State Boundary Search (GSBS)

### Theoretical Rationale

GSBS (Geerligs et al., 2015, 2021) is a data-driven algorithm that identifies temporal boundaries by maximizing correlation differences within vs across boundaries. It directly operationalizes the concept that event boundaries are moments of maximal neural pattern dissimilarity.

**Hypothesis Application**:
- **H1**: High anxiety → more boundaries detected
- **H2**: Boundary strength differs by anxiety
- **H4**: Connectivity patterns at boundaries differ by anxiety

### Method Description

**Step 1: Compute Time-Resolved Connectivity**

```python
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def compute_dynamic_connectivity(roi_timeseries, window_size=30):
    """
    Compute sliding window connectivity matrices.

    Parameters:
    - roi_timeseries: (n_timepoints, n_rois)
    - window_size: window length in TRs

    Returns:
    - conn_matrices: (n_windows, n_rois, n_rois)
    """
    n_timepoints, n_rois = roi_timeseries.shape
    n_windows = n_timepoints - window_size + 1

    conn_matrices = np.zeros((n_windows, n_rois, n_rois))

    for i in range(n_windows):
        window_data = roi_timeseries[i:i+window_size, :]
        conn_matrices[i] = np.corrcoef(window_data.T)

    return conn_matrices

# Compute connectivity
dyn_conn = compute_dynamic_connectivity(roi_time_series, window_size=30)
```

**Step 2: GSBS Algorithm**

```python
def gsbs_boundary_detection(conn_matrices, n_boundaries=None, max_iter=100):
    """
    Greedy State Boundary Search algorithm.

    Based on Geerligs et al. (2015).
    """
    n_windows = conn_matrices.shape[0]

    # Vectorize connectivity matrices (upper triangle)
    def vectorize_conn(conn):
        triu_idx = np.triu_indices(conn.shape[0], k=1)
        return conn[triu_idx]

    conn_vectors = np.array([vectorize_conn(c) for c in conn_matrices])

    if n_boundaries is None:
        # Auto-determine number of boundaries via cross-validation
        n_boundaries = optimize_n_boundaries(conn_vectors)

    # Initialize: evenly spaced boundaries
    boundaries = np.linspace(0, n_windows, n_boundaries+2, dtype=int)[1:-1]

    # Objective function: maximize between-state similarity / across-boundary dissimilarity
    def compute_objective(boundaries, conn_vectors):
        """
        Compute GSBS objective function.
        """
        # Add start and end points
        bounds_full = np.concatenate([[0], boundaries, [n_windows]])

        # Within-state similarity
        within_sim = 0
        for i in range(len(bounds_full)-1):
            segment = conn_vectors[bounds_full[i]:bounds_full[i+1]]
            if len(segment) > 1:
                # Mean pairwise correlation within segment
                within_sim += np.mean(pdist(segment, metric='correlation'))

        # Across-boundary dissimilarity
        across_dissim = 0
        for b in boundaries:
            if b > 0 and b < n_windows:
                before = conn_vectors[max(0, b-5):b]
                after = conn_vectors[b:min(n_windows, b+5)]
                if len(before) > 0 and len(after) > 0:
                    across_dissim += 1 - pearsonr(
                        np.mean(before, axis=0),
                        np.mean(after, axis=0)
                    )[0]

        # Objective: maximize ratio
        return across_dissim / (within_sim + 1e-10)

    # Greedy optimization
    best_objective = compute_objective(boundaries, conn_vectors)

    for iteration in range(max_iter):
        improved = False

        for i, boundary in enumerate(boundaries):
            # Try moving boundary
            for delta in [-2, -1, 1, 2]:
                new_boundary = boundary + delta

                # Check validity
                if new_boundary <= 0 or new_boundary >= n_windows:
                    continue
                if i > 0 and new_boundary <= boundaries[i-1]:
                    continue
                if i < len(boundaries)-1 and new_boundary >= boundaries[i+1]:
                    continue

                # Test new configuration
                new_boundaries = boundaries.copy()
                new_boundaries[i] = new_boundary
                new_objective = compute_objective(new_boundaries, conn_vectors)

                if new_objective > best_objective:
                    boundaries = new_boundaries
                    best_objective = new_objective
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break  # Converged

    return boundaries, best_objective

# Run GSBS
boundaries, objective_score = gsbs_boundary_detection(dyn_conn, n_boundaries=25)

print(f"Detected {len(boundaries)} boundaries")
print(f"Objective score: {objective_score:.4f}")
```

**Step 3: Optimize Number of Boundaries**

```python
def optimize_n_boundaries(conn_vectors, n_range=range(10, 50, 5)):
    """
    Cross-validation to select optimal number of boundaries.
    """
    from sklearn.model_selection import KFold

    scores = []

    for n_bound in n_range:
        kf = KFold(n_splits=5, shuffle=False)
        fold_scores = []

        for train_idx, test_idx in kf.split(conn_vectors):
            train_conn = conn_vectors[train_idx]
            test_conn = conn_vectors[test_idx]

            # Fit on train
            bounds_train, _ = gsbs_boundary_detection(
                train_conn.reshape(-1, conn_vectors.shape[1], conn_vectors.shape[2]),
                n_boundaries=n_bound,
                max_iter=50
            )

            # Evaluate on test
            # (Use training boundaries to compute test objective)
            # ... implementation details

        scores.append(np.mean(fold_scores))

    optimal_n = n_range[np.argmax(scores)]
    return optimal_n
```

**Step 4: Boundary Strength Quantification**

```python
def compute_boundary_strength(conn_vectors, boundaries, window=5):
    """
    Quantify strength of each boundary.

    Strength = dissimilarity before vs after boundary
    """
    strengths = []

    for b in boundaries:
        before = conn_vectors[max(0, b-window):b]
        after = conn_vectors[b:min(len(conn_vectors), b+window)]

        if len(before) > 0 and len(after) > 0:
            before_mean = np.mean(before, axis=0)
            after_mean = np.mean(after, axis=0)

            # Dissimilarity (1 - correlation)
            strength = 1 - pearsonr(before_mean, after_mean)[0]
            strengths.append(strength)
        else:
            strengths.append(np.nan)

    return np.array(strengths)

boundary_strengths = compute_boundary_strength(conn_vectors, boundaries)
```

**Step 5: Statistical Analysis**

```python
# Per-subject metrics
subject_metrics = []

for subject_id, anxiety_score in zip(subjects, anxiety_scores):
    # Load subject data
    subj_conn = load_subject_connectivity(subject_id)

    # Detect boundaries
    subj_boundaries, _ = gsbs_boundary_detection(subj_conn)
    subj_strengths = compute_boundary_strength(conn_vectors_subj, subj_boundaries)

    subject_metrics.append({
        'subject': subject_id,
        'anxiety': anxiety_score,
        'n_boundaries': len(subj_boundaries),
        'mean_strength': np.nanmean(subj_strengths),
        'max_strength': np.nanmax(subj_strengths)
    })

df = pd.DataFrame(subject_metrics)

# Correlation analysis
print("Anxiety × Boundary Count:")
r, p = stats.pearsonr(df['anxiety'], df['n_boundaries'])
print(f"r = {r:.3f}, p = {p:.4f}")

print("\nAnxiety × Mean Boundary Strength:")
r, p = stats.pearsonr(df['anxiety'], df['mean_strength'])
print(f"r = {r:.3f}, p = {p:.4f}")
```

### Expected Outcomes

**Boundary Detection**:
- High anxiety: 32.4 ± 7.2 boundaries per 10-minute scan
- Low anxiety: 24.8 ± 5.6 boundaries
- t(78) = 4.87, p < 0.001, d = 1.15

**Boundary Strength**:
- High anxiety: Mean strength = 0.42 ± 0.08
- Low anxiety: Mean strength = 0.38 ± 0.06
- t(78) = 2.41, p = 0.018, d = 0.57

**Correlation**:
- STAI-T × Boundary Count: r = 0.45, p < 0.001
- STAI-T × Mean Strength: r = 0.28, p = 0.012

### Advantages

1. **Purely Data-Driven**: No assumptions about state distributions
2. **Direct Boundary Detection**: Explicitly optimizes for boundaries
3. **Validated Method**: Used in multiple published studies
4. **Boundary Strength**: Quantifies "how much" of a boundary
5. **Flexible**: Can optimize number of boundaries
6. **Robust**: Less sensitive to outliers than some methods

### Disadvantages

1. **Greedy Algorithm**: May not find global optimum
   - Solution: Multiple random initializations

2. **Window Size Parameter**: Arbitrary choice for connectivity computation
   - Solution: Sensitivity analysis across window sizes

3. **Computational Cost**: Intensive for large datasets
   - Solution: Parallel processing, efficient implementations

4. **Boundary Placement**: Discrete timepoints may miss precise boundaries
   - Temporal resolution limited by TR

5. **No State Characterization**: Identifies boundaries but not states themselves
   - Solution: Combine with HMM for state interpretation

### Comparison with HMM

| Aspect | GSBS | HMM |
|--------|------|-----|
| **Approach** | Direct boundary optimization | Latent state modeling |
| **Boundaries** | Primary output | Derived from states |
| **States** | Not modeled | Explicitly modeled |
| **Strength** | Quantified directly | Inferred from transitions |
| **Assumptions** | Minimal | Markov property, Gaussian emissions |
| **Interpretability** | Boundaries only | States + transitions |

**Recommendation**: Use both for convergent validation.

---

## Method 3: Seed-Based Functional Connectivity Analysis

### Theoretical Rationale

Seed-based connectivity examines functional relationships between a seed region and all other brain regions. In anxiety research, amygdala connectivity is particularly relevant given its role in threat processing and emotion regulation.

**Hypothesis Application**:
- **H3**: Anxiety modulates amygdala-prefrontal connectivity
- **H4**: Connectivity at event boundaries differs by anxiety
- Mechanistic interpretation of boundary effects

### Method Description

**Step 1: Seed Region Selection**

Theory-driven ROIs:

```python
# Define seed regions (MNI coordinates)
seeds = {
    'left_amygdala': (-22, -4, -18),
    'right_amygdala': (24, -4, -18),
    'vmPFC': (0, 50, -10),
    'dmPFC': (0, 52, 28),
    'PCC': (0, -52, 26),
    'left_hippocampus': (-28, -20, -14),
    'right_hippocampus': (28, -20, -14)
}

# Create seed masks
from nilearn import datasets
from nilearn.image import resample_to_img

def create_seed_mask(coordinate, radius=6, ref_img=None):
    """
    Create spherical seed mask around coordinate.
    """
    from nilearn.image import new_img_like
    from nilearn.masking import compute_epi_mask

    # Create sphere
    sphere = datasets.fetch_coords_power_2011()
    # ... implementation
    return seed_mask
```

**Step 2: Extract Seed Time Series**

```python
from nilearn.input_data import NiftiSpheresMasker

# Extract time series from seeds
seed_masker = NiftiSpheresMasker(
    seeds=list(seeds.values()),
    radius=6.0,  # 6mm sphere
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

seed_timeseries = seed_masker.fit_transform(fmri_img)
# Shape: (n_timepoints, n_seeds)
```

**Step 3: Whole-Brain Connectivity Map**

```python
from nilearn.connectivity import ConnectivityMeasure

# Compute seed-to-voxel connectivity
def compute_seed_connectivity(fmri_img, seed_ts):
    """
    Compute whole-brain connectivity for seed.
    """
    # Load whole-brain data
    from nilearn.masking import apply_mask, unmask

    brain_mask = compute_epi_mask(fmri_img)
    brain_data = apply_mask(fmri_img, brain_mask)

    # Correlate seed with every voxel
    n_voxels = brain_data.shape[1]
    conn_map = np.zeros(n_voxels)

    for v in range(n_voxels):
        conn_map[v] = pearsonr(seed_ts, brain_data[:, v])[0]

    # Convert to Z-scores (Fisher transformation)
    conn_map_z = np.arctanh(conn_map)

    # Unmask to brain image
    conn_img = unmask(conn_map_z, brain_mask)

    return conn_img

# For each subject
amygdala_connectivity_maps = []
for subject_id in subjects:
    subj_img = load_subject_fmri(subject_id)
    subj_seed_ts = seed_masker.fit_transform(subj_img)

    conn_map = compute_seed_connectivity(subj_img, subj_seed_ts[:, 0])  # Left amygdala
    amygdala_connectivity_maps.append(conn_map)
```

**Step 4: Group Comparison**

```python
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# Second-level GLM: High vs Low Anxiety
design_matrix = pd.DataFrame({
    'high_anxiety': (anxiety_scores >= np.median(anxiety_scores)).astype(int),
    'age': ages,
    'sex': sex_codes,
    'mean_fd': mean_fds
})

second_level_model = SecondLevelModel(smoothing_fwhm=6.0)
second_level_model.fit(
    amygdala_connectivity_maps,
    design_matrix=design_matrix
)

# Contrast: High > Low anxiety
z_map = second_level_model.compute_contrast(
    'high_anxiety',
    output_type='z_score'
)

# Threshold and correct for multiple comparisons
thresholded_map, threshold = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control='fdr',  # FDR correction
    cluster_threshold=10
)

# Visualize
from nilearn import plotting
plotting.plot_stat_map(
    thresholded_map,
    threshold=threshold,
    title='High > Low Anxiety: Amygdala Connectivity',
    cut_coords=(-22, -4, -18)
)
```

**Step 5: ROI-to-ROI Connectivity**

```python
# Extract connectivity between predefined ROIs
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')

# For each subject
roi_connectivity_matrices = []
for subject_id in subjects:
    subj_roi_ts = load_subject_roi_timeseries(subject_id)
    conn_matrix = correlation_measure.fit_transform([subj_roi_ts])[0]
    roi_connectivity_matrices.append(conn_matrix)

roi_conn_array = np.array(roi_connectivity_matrices)

# Compare specific connections
amyg_vmPFC_conn = roi_conn_array[:, amyg_idx, vmPFC_idx]

# Correlation with anxiety
r, p = stats.pearsonr(anxiety_scores, amyg_vmPFC_conn)
print(f"Anxiety × Amygdala-vmPFC connectivity: r={r:.3f}, p={p:.4f}")
```

**Step 6: Connectivity at Event Boundaries**

Combine with HMM/GSBS boundaries:

```python
def connectivity_at_boundaries(roi_timeseries, boundaries, window=10):
    """
    Compute connectivity specifically at event boundaries.
    """
    boundary_connectivity = []

    for b in boundaries:
        # Extract time window around boundary
        start = max(0, b - window)
        end = min(len(roi_timeseries), b + window)

        boundary_data = roi_timeseries[start:end, :]
        conn_matrix = np.corrcoef(boundary_data.T)

        boundary_connectivity.append(conn_matrix)

    # Average across boundaries
    mean_boundary_conn = np.mean(boundary_connectivity, axis=0)

    return mean_boundary_conn

# Compare boundary vs non-boundary connectivity
def connectivity_non_boundaries(roi_timeseries, boundaries, window=10):
    """
    Compute connectivity away from boundaries.
    """
    # Identify non-boundary periods
    all_timepoints = set(range(len(roi_timeseries)))
    boundary_timepoints = set()
    for b in boundaries:
        boundary_timepoints.update(range(max(0, b-window), min(len(roi_timeseries), b+window)))

    non_boundary_timepoints = list(all_timepoints - boundary_timepoints)

    non_boundary_data = roi_timeseries[non_boundary_timepoints, :]
    conn_matrix = np.corrcoef(non_boundary_data.T)

    return conn_matrix

# Statistical comparison
for subject_id, anxiety_score in zip(subjects, anxiety_scores):
    subj_roi_ts = load_subject_roi_timeseries(subject_id)
    subj_boundaries = load_subject_boundaries(subject_id)  # From HMM/GSBS

    conn_boundary = connectivity_at_boundaries(subj_roi_ts, subj_boundaries)
    conn_non_boundary = connectivity_non_boundaries(subj_roi_ts, subj_boundaries)

    # Example: Hippocampus-PCC connectivity
    hipp_pcc_boundary = conn_boundary[hipp_idx, pcc_idx]
    hipp_pcc_nonboundary = conn_non_boundary[hipp_idx, pcc_idx]

    # Store for group analysis
    # ...

# Test H4: High anxiety shows reduced hippocampal-PMN connectivity at boundaries
```

### Expected Outcomes

**Seed-Based Connectivity (Amygdala)**:

High vs Low Anxiety Contrast:
- **Increased connectivity**: Amygdala → anterior insula, dorsal ACC (Z > 3.1, p < 0.05 FWE-corrected)
- **Decreased connectivity**: Amygdala → vmPFC, dmPFC (Z > 3.1, p < 0.05 FWE-corrected)

**ROI-to-ROI Connectivity**:

| Connection | High Anxiety | Low Anxiety | Cohen's d | p-value |
|------------|--------------|-------------|-----------|---------|
| Amyg-vmPFC | r = 0.12 ± 0.18 | r = 0.34 ± 0.16 | -1.30 | < 0.001 |
| Amyg-Insula | r = 0.41 ± 0.15 | r = 0.26 ± 0.14 | 1.04 | < 0.01 |
| PCC-Hippocampus | r = 0.52 ± 0.12 | r = 0.61 ± 0.11 | -0.78 | 0.02 |

**Connectivity at Boundaries**:
- High anxiety: Reduced hippocampus-PCC connectivity at boundaries (r = 0.31 vs 0.48, p = 0.008)
- Supports H4: Impaired memory consolidation

### Advantages

1. **Neurobiologically Interpretable**: Clear anatomical regions
2. **Theory-Driven**: Tests specific hypotheses about circuits
3. **Established in Anxiety Research**: Amygdala-PFC connectivity well-studied
4. **Mechanistic Insight**: Explains *how* anxiety affects boundaries
5. **Complements Data-Driven Methods**: Validates HMM/GSBS findings
6. **Clinical Relevance**: Connectivity biomarkers for treatment targets

### Disadvantages

1. **ROI Selection Bias**: Pre-selecting regions may miss effects
   - Solution: Whole-brain seed-based + ROI analysis

2. **Spatial Resolution**: Seeds may include heterogeneous subregions
   - Amygdala has functionally distinct subnuclei
   - Solution: Probabilistic anatomical ROIs

3. **Static Connectivity**: Standard approach doesn't capture dynamics
   - Solution: Combine with dynamic FC at boundaries

4. **Circular Analysis Risk**: If using boundaries to define connectivity windows
   - Solution: Independent boundary detection (separate sessions/data)

5. **Multiple Comparisons**: Many possible connections
   - Solution: Hypothesis-driven (amygdala-PFC) + correction (FDR/FWE)

### Integration with Boundary Detection

**Pipeline**:
1. Detect boundaries via HMM or GSBS (data-driven)
2. Compute connectivity at boundaries vs non-boundaries
3. Compare connectivity patterns between anxiety groups
4. Interpret boundary detection mechanisms

**Hypothesis**:
- Boundaries reflect DMN-SN switching
- High anxiety → excessive switching → more boundaries
- Reduced connectivity at boundaries → poor memory encoding

---

## Method 4: Sliding Window Dynamic Functional Connectivity (dFC)

### Theoretical Rationale

Dynamic functional connectivity captures time-varying network interactions. Event boundaries may correspond to transitions between distinct connectivity states. Anxiety could modulate the frequency and nature of these state transitions.

**Hypothesis Application**:
- **H1**: High anxiety → more frequent state transitions
- **H2**: State transition "volatility" correlates with state anxiety
- **H3**: Anxiety alters dFC state repertoire

### Method Description

**Step 1: Compute Sliding Window Connectivity**

```python
def compute_sliding_window_connectivity(roi_timeseries, window_size=30, step=1):
    """
    Compute dynamic functional connectivity using sliding window.

    Parameters:
    - roi_timeseries: (n_timepoints, n_rois)
    - window_size: window length in TRs (e.g., 30 TRs = 60 seconds at TR=2s)
    - step: step size in TRs

    Returns:
    - dfc_matrices: (n_windows, n_rois, n_rois)
    - window_centers: (n_windows,) timepoints of window centers
    """
    n_timepoints, n_rois = roi_timeseries.shape
    n_windows = (n_timepoints - window_size) // step + 1

    dfc_matrices = np.zeros((n_windows, n_rois, n_rois))
    window_centers = np.zeros(n_windows)

    for w in range(n_windows):
        start = w * step
        end = start + window_size

        if end <= n_timepoints:
            window_data = roi_timeseries[start:end, :]
            dfc_matrices[w] = np.corrcoef(window_data.T)
            window_centers[w] = (start + end) / 2

    return dfc_matrices, window_centers

# Compute dFC
dfc_matrices, time_centers = compute_sliding_window_connectivity(
    roi_time_series,
    window_size=30,  # 60 seconds at TR=2s
    step=1
)
```

**Step 2: Identify dFC States via Clustering**

```python
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

def cluster_dfc_states(dfc_matrices, n_states=5, method='kmeans'):
    """
    Cluster dynamic connectivity matrices into discrete states.
    """
    # Vectorize connectivity matrices (upper triangle)
    n_windows = dfc_matrices.shape[0]
    n_rois = dfc_matrices.shape[1]
    n_connections = n_rois * (n_rois - 1) // 2

    dfc_vectors = np.zeros((n_windows, n_connections))

    for w in range(n_windows):
        triu_idx = np.triu_indices(n_rois, k=1)
        dfc_vectors[w] = dfc_matrices[w][triu_idx]

    if method == 'kmeans':
        # K-means clustering
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=100)
        state_labels = kmeans.fit_predict(dfc_vectors)
        cluster_centers = kmeans.cluster_centers_

    elif method == 'hierarchical':
        # Hierarchical clustering
        linkage_matrix = linkage(dfc_vectors, method='ward')
        state_labels = fcluster(linkage_matrix, n_states, criterion='maxclust') - 1

        # Compute cluster centers
        cluster_centers = np.array([
            np.mean(dfc_vectors[state_labels == k], axis=0)
            for k in range(n_states)
        ])

    return state_labels, cluster_centers

# Determine optimal number of states
def elbow_method(dfc_matrices, k_range=range(2, 11)):
    """
    Elbow method for optimal K selection.
    """
    inertias = []

    for k in k_range:
        state_labels, _ = cluster_dfc_states(dfc_matrices, n_states=k)

        # Compute inertia (within-cluster sum of squares)
        dfc_vectors = vectorize_dfc(dfc_matrices)
        inertia = 0
        for state_id in range(k):
            state_samples = dfc_vectors[state_labels == state_id]
            if len(state_samples) > 0:
                center = np.mean(state_samples, axis=0)
                inertia += np.sum((state_samples - center) ** 2)

        inertias.append(inertia)

    # Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of States (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    return inertias

# Select optimal K
optimal_k = 5  # Based on elbow plot
state_labels, state_centers = cluster_dfc_states(dfc_matrices, n_states=optimal_k)
```

**Step 3: Detect State Transitions (Boundaries)**

```python
def detect_state_transitions(state_labels):
    """
    Detect timepoints where dFC state changes.
    """
    transitions = []

    for t in range(1, len(state_labels)):
        if state_labels[t] != state_labels[t-1]:
            transitions.append(t)

    return np.array(transitions)

# Detect transitions
transitions = detect_state_transitions(state_labels)

# Compute transition metrics
n_transitions = len(transitions)
transition_rate = n_transitions / (len(state_labels) * TR / 60)  # transitions per minute
mean_dwell_time = np.mean(np.diff(np.concatenate([[0], transitions, [len(state_labels)]]))) * TR
```

**Step 4: State Occupancy and Repertoire Analysis**

```python
def compute_state_occupancy(state_labels, n_states):
    """
    Compute fraction of time spent in each state.
    """
    occupancy = np.zeros(n_states)

    for state_id in range(n_states):
        occupancy[state_id] = np.sum(state_labels == state_id) / len(state_labels)

    return occupancy

# Compute for each subject
all_occupancies = []
all_transition_rates = []

for subject_id, anxiety_score in zip(subjects, anxiety_scores):
    subj_dfc = load_subject_dfc(subject_id)
    subj_states, _ = cluster_dfc_states(subj_dfc, n_states=optimal_k)

    occupancy = compute_state_occupancy(subj_states, optimal_k)
    transitions = detect_state_transitions(subj_states)
    trans_rate = len(transitions) / (len(subj_states) * TR / 60)

    all_occupancies.append(occupancy)
    all_transition_rates.append(trans_rate)

occupancy_array = np.array(all_occupancies)  # (n_subjects, n_states)

# Statistical analysis
# Correlation: Anxiety × Transition Rate
r, p = stats.pearsonr(anxiety_scores, all_transition_rates)
print(f"Anxiety × Transition Rate: r={r:.3f}, p={p:.4f}")

# State occupancy differences
for state_id in range(optimal_k):
    high_occ = occupancy_array[anxiety_high_idx, state_id]
    low_occ = occupancy_array[anxiety_low_idx, state_id]

    t, p = stats.ttest_ind(high_occ, low_occ)
    print(f"State {state_id}: High={np.mean(high_occ):.3f}, Low={np.mean(low_occ):.3f}, p={p:.4f}")
```

**Step 5: Characterize dFC States**

```python
def characterize_dfc_state(state_center, roi_names, networks):
    """
    Interpret a dFC state by its connectivity pattern.
    """
    # Reconstruct connectivity matrix from vector
    n_rois = len(roi_names)
    conn_matrix = np.zeros((n_rois, n_rois))

    triu_idx = np.triu_indices(n_rois, k=1)
    conn_matrix[triu_idx] = state_center
    conn_matrix = conn_matrix + conn_matrix.T  # Symmetrize

    # Analyze network structure
    network_connectivity = {}

    for net_name, net_rois in networks.items():
        net_idx = [roi_names.index(r) for r in net_rois]
        within_net_conn = conn_matrix[np.ix_(net_idx, net_idx)]
        network_connectivity[net_name] = {
            'within_network': np.mean(within_net_conn[np.triu_indices(len(net_idx), k=1)]),
            'to_other_networks': np.mean([
                np.mean(conn_matrix[np.ix_(net_idx, other_idx)])
                for other_net, other_rois in networks.items()
                if other_net != net_name
                for other_idx in [[roi_names.index(r) for r in other_rois]]
            ])
        }

    return network_connectivity

# Define networks
networks = {
    'DMN': ['PCC', 'mPFC', 'AG_L', 'AG_R'],
    'Salience': ['Insula_L', 'Insula_R', 'ACC'],
    'Executive': ['dlPFC_L', 'dlPFC_R', 'Parietal_L', 'Parietal_R']
}

# Characterize each state
for state_id in range(optimal_k):
    print(f"\n=== State {state_id} ===")
    net_conn = characterize_dfc_state(state_centers[state_id], roi_names, networks)
    for net, conn_vals in net_conn.items():
        print(f"{net}: Within={conn_vals['within_network']:.3f}, Between={conn_vals['to_other_networks']:.3f}")
```

### Expected Outcomes

**Transition Rate**:
- High anxiety: 3.8 ± 1.2 transitions/min
- Low anxiety: 2.4 ± 0.9 transitions/min
- t(78) = 5.42, p < 0.001, d = 1.28

**State Occupancy**:

| State | Description | High Anxiety | Low Anxiety | p-value |
|-------|-------------|--------------|-------------|---------|
| State 1 | DMN-dominant | 0.18 ± 0.06 | 0.28 ± 0.07 | < 0.001 |
| State 2 | Salience-dominant | 0.31 ± 0.08 | 0.21 ± 0.06 | < 0.001 |
| State 3 | Executive-dominant | 0.19 ± 0.05 | 0.22 ± 0.06 | 0.08 |
| State 4 | Transitional | 0.22 ± 0.07 | 0.18 ± 0.05 | 0.02 |
| State 5 | Segregated | 0.10 ± 0.04 | 0.11 ± 0.04 | 0.45 |

**Correlation**:
- STAI-T × Transition Rate: r = 0.48, p < 0.001
- STAI-S × State 2 Occupancy: r = 0.39, p < 0.001

### Advantages

1. **Captures Dynamics**: Time-varying connectivity, not static averages
2. **Established Method**: Widely used in resting-state fMRI research
3. **State Characterization**: Can interpret states via network structure
4. **Multiple Metrics**: Transition rate, occupancy, dwell time
5. **Directly Tests Hypotheses**: State switching relates to boundaries

### Disadvantages

1. **Window Size Arbitrary**: No consensus on optimal window length
   - 30-60 seconds common, but choice affects results
   - Solution: Test multiple window sizes, sensitivity analysis

2. **Spurious Fluctuations**: Sliding introduces autocorrelation
   - Solution: Statistical tests accounting for autocorrelation

3. **State Number Selection**: K-means requires pre-specifying K
   - Solution: Elbow method, silhouette analysis

4. **Computational Cost**: Large matrices for many ROIs
   - Solution: Focus on network ROIs, not whole brain

5. **Interpretation Challenges**: States may not have clear biological meaning
   - Solution: Post-hoc network analysis, validate with task data

---

## Method 5: Multi-Voxel Pattern Analysis (MVPA) with Temporal Clustering

### Theoretical Rationale

MVPA examines distributed patterns of neural activity. Event boundaries correspond to moments when multi-voxel patterns change substantially. Pattern similarity analysis can identify these transitions in a data-driven manner.

**Hypothesis Application**:
- **H1**: High anxiety → more pattern transitions
- **H2**: Pattern transition magnitude correlates with boundary strength
- Sensitive to subtle pattern changes that GLM might miss

### Method Description

**Step 1: Extract Multi-Voxel Patterns**

```python
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import StandardScaler

def extract_mvpa_patterns(fmri_img, mask='DMN', atlas='yeo'):
    """
    Extract multi-voxel patterns from network of interest.

    Parameters:
    - fmri_img: 4D fMRI image
    - mask: 'DMN', 'Salience', 'Whole_brain', or custom mask
    - atlas: 'yeo', 'schaefer', 'power', etc.

    Returns:
    - patterns: (n_timepoints, n_voxels)
    """
    if mask == 'DMN':
        # Use Yeo 7-network atlas, DMN = network 7
        from nilearn import datasets
        yeo_atlas = datasets.fetch_atlas_yeo_2011()
        dmn_mask = (yeo_atlas.thick_7 == 7)

    elif mask == 'Whole_brain':
        from nilearn.masking import compute_epi_mask
        dmn_mask = compute_epi_mask(fmri_img)

    # Create masker
    masker = NiftiMasker(
        mask_img=dmn_mask,
        standardize=True,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0
    )

    # Extract patterns
    patterns = masker.fit_transform(fmri_img)

    return patterns

patterns = extract_mvpa_patterns(fmri_img, mask='DMN')
# Shape: (n_timepoints, n_dmn_voxels)
```

**Step 2: Compute Pattern Similarity**

```python
from scipy.spatial.distance import pdist, squareform

def compute_pattern_similarity_matrix(patterns, metric='correlation'):
    """
    Compute pairwise pattern similarity across time.

    Returns:
    - similarity_matrix: (n_timepoints, n_timepoints)
    """
    if metric == 'correlation':
        # Pearson correlation between patterns
        similarity = np.corrcoef(patterns)

    elif metric == 'cosine':
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(patterns)

    elif metric == 'euclidean':
        # Euclidean distance (inverted)
        distances = squareform(pdist(patterns, metric='euclidean'))
        similarity = 1 / (1 + distances)  # Convert to similarity

    return similarity

similarity_matrix = compute_pattern_similarity_matrix(patterns, metric='correlation')
```

**Step 3: Detect Boundaries via Pattern Dissimilarity**

```python
def detect_pattern_boundaries(similarity_matrix, threshold_percentile=90, min_distance=10):
    """
    Detect event boundaries as moments of low pattern similarity.

    Boundaries occur when:
    - Similarity(t-k, t+k) is low (dissimilar before/after)
    - Local minimum in temporal similarity
    """
    n_timepoints = similarity_matrix.shape[0]
    boundary_scores = np.zeros(n_timepoints)

    # Compute boundary score for each timepoint
    for t in range(n_timepoints):
        # Define windows before and after
        window_size = 5

        if t >= window_size and t < n_timepoints - window_size:
            before_idx = range(t - window_size, t)
            after_idx = range(t + 1, t + window_size + 1)

            # Average similarity before vs after
            cross_similarity = similarity_matrix[np.ix_(before_idx, after_idx)]
            boundary_scores[t] = 1 - np.mean(cross_similarity)  # Dissimilarity

    # Threshold to identify boundaries
    threshold = np.percentile(boundary_scores, threshold_percentile)

    # Find peaks in boundary scores
    from scipy.signal import find_peaks

    boundaries, _ = find_peaks(
        boundary_scores,
        height=threshold,
        distance=min_distance
    )

    return boundaries, boundary_scores

boundaries, boundary_scores = detect_pattern_boundaries(
    similarity_matrix,
    threshold_percentile=90,
    min_distance=10
)

print(f"Detected {len(boundaries)} pattern-based boundaries")
```

**Step 4: Quantify Boundary Strength**

```python
def compute_pattern_boundary_strength(patterns, boundaries, window=5):
    """
    Quantify strength of pattern change at boundaries.
    """
    strengths = []

    for b in boundaries:
        if b >= window and b < len(patterns) - window:
            # Patterns before and after boundary
            before_patterns = patterns[b-window:b, :]
            after_patterns = patterns[b+1:b+window+1, :]

            # Average patterns
            before_mean = np.mean(before_patterns, axis=0)
            after_mean = np.mean(after_patterns, axis=0)

            # Dissimilarity (1 - correlation)
            strength = 1 - pearsonr(before_mean, after_mean)[0]
            strengths.append(strength)
        else:
            strengths.append(np.nan)

    return np.array(strengths)

pattern_strengths = compute_pattern_boundary_strength(patterns, boundaries)
```

**Step 5: Representational Similarity Analysis (RSA)**

```python
def temporal_rsa(similarity_matrix, window_size=20):
    """
    Analyze temporal structure via representational similarity.

    Identifies nested event structure by examining similarity at different timescales.
    """
    n_timepoints = similarity_matrix.shape[0]

    # Extract diagonal bands (different temporal lags)
    lags = range(1, window_size + 1)
    temporal_similarity = []

    for lag in lags:
        # Extract diagonal at lag
        diag_values = np.diag(similarity_matrix, k=lag)
        temporal_similarity.append(np.mean(diag_values))

    # Plot temporal similarity profile
    plt.figure(figsize=(10, 5))
    plt.plot(lags, temporal_similarity, 'o-')
    plt.xlabel('Temporal Lag (TRs)')
    plt.ylabel('Average Pattern Similarity')
    plt.title('Temporal Similarity Profile')
    plt.show()

    return temporal_similarity

# Analyze
temp_sim = temporal_rsa(similarity_matrix, window_size=30)
```

**Step 6: Statistical Analysis**

```python
# Per-subject pattern boundary analysis
subject_pattern_metrics = []

for subject_id, anxiety_score in zip(subjects, anxiety_scores):
    subj_img = load_subject_fmri(subject_id)
    subj_patterns = extract_mvpa_patterns(subj_img, mask='DMN')
    subj_sim_matrix = compute_pattern_similarity_matrix(subj_patterns)
    subj_boundaries, subj_scores = detect_pattern_boundaries(subj_sim_matrix)
    subj_strengths = compute_pattern_boundary_strength(subj_patterns, subj_boundaries)

    subject_pattern_metrics.append({
        'subject': subject_id,
        'anxiety': anxiety_score,
        'n_pattern_boundaries': len(subj_boundaries),
        'mean_boundary_strength': np.nanmean(subj_strengths),
        'mean_temporal_similarity': np.mean(subj_sim_matrix[np.triu_indices(len(subj_sim_matrix), k=1)])
    })

df_pattern = pd.DataFrame(subject_pattern_metrics)

# Statistical tests
print("Anxiety × Pattern Boundary Count:")
r, p = stats.pearsonr(df_pattern['anxiety'], df_pattern['n_pattern_boundaries'])
print(f"r = {r:.3f}, p = {p:.4f}")

print("\nAnxiety × Pattern Boundary Strength:")
r, p = stats.pearsonr(df_pattern['anxiety'], df_pattern['mean_boundary_strength'])
print(f"r = {r:.3f}, p = {p:.4f}")

# Compare with HMM boundaries
# Convergent validation: Do MVPA and HMM boundaries align?
from sklearn.metrics import adjusted_rand_score

hmm_boundaries = load_hmm_boundaries(subject_id)
mvpa_boundaries = df_pattern.loc[df_pattern['subject'] == subject_id, 'boundaries']

# Temporal alignment score
def boundary_alignment(boundaries1, boundaries2, tolerance=3):
    """
    Compute alignment between two boundary sets.

    Tolerance: TRs within which boundaries are considered aligned.
    """
    aligned = 0
    for b1 in boundaries1:
        if any(abs(b1 - b2) <= tolerance for b2 in boundaries2):
            aligned += 1

    precision = aligned / len(boundaries1) if len(boundaries1) > 0 else 0
    recall = aligned / len(boundaries2) if len(boundaries2) > 0 else 0

    return precision, recall

prec, rec = boundary_alignment(hmm_boundaries, mvpa_boundaries, tolerance=3)
print(f"HMM-MVPA Alignment: Precision={prec:.3f}, Recall={rec:.3f}")
```

### Expected Outcomes

**Pattern Boundaries**:
- High anxiety: 26.3 ± 5.8 pattern boundaries
- Low anxiety: 19.7 ± 4.2 pattern boundaries
- t(78) = 5.18, p < 0.001, d = 1.23

**Pattern Strength**:
- High anxiety: Mean dissimilarity = 0.54 ± 0.09
- Low anxiety: Mean dissimilarity = 0.47 ± 0.07
- t(78) = 3.64, p < 0.001, d = 0.86

**Convergent Validation**:
- MVPA-HMM boundary alignment: Precision = 0.68, Recall = 0.71
- MVPA-GSBS boundary alignment: Precision = 0.72, Recall = 0.69

**Correlation**:
- STAI-T × Pattern Boundary Count: r = 0.41, p < 0.001
- STAI-T × Pattern Boundary Strength: r = 0.33, p = 0.003

### Advantages

1. **Distributed Representations**: Captures patterns across multiple voxels
2. **Sensitive to Subtle Changes**: Detects gradual pattern shifts
3. **Flexible ROI**: Can apply to whole brain or specific networks
4. **Convergent Validation**: Compare with HMM/GSBS for robustness
5. **Hierarchical Structure**: RSA reveals nested event organization

### Disadvantages

1. **Computationally Intensive**: Large similarity matrices for whole brain
   - Solution: Focus on network ROIs (DMN, Salience)

2. **ROI Selection**: Requires choosing which brain regions to analyze
   - Solution: Theory-driven (DMN) + exploratory (whole-brain)

3. **Threshold Selection**: Boundary detection threshold is arbitrary
   - Solution: Percentile-based, cross-validation

4. **Interpretability**: Patterns are abstract, not easily visualized
   - Solution: Post-hoc decoding, network characterization

5. **Edge Effects**: Cannot detect boundaries at start/end of scan
   - Solution: Exclude first/last window_size TRs from analysis

---

## Summary Comparison of Methods

| Method | Boundary Detection | State Modeling | Connectivity | Computation | Validation | Best For |
|--------|-------------------|----------------|--------------|-------------|------------|----------|
| **HMM** | Derived from states | ✓✓✓ Explicit | Post-hoc | Moderate | Established | State transitions |
| **GSBS** | ✓✓✓ Direct optimization | States not modeled | Via similarity | High | Published | Pure boundaries |
| **Seed-Based FC** | Requires external method | No | ✓✓✓ Direct | Low | Gold standard | Mechanistic insight |
| **Sliding Window dFC** | Via state transitions | ✓✓ Clustering | ✓✓ Dynamic | Moderate | Widely used | Network dynamics |
| **MVPA Pattern** | ✓✓ Dissimilarity peaks | Post-hoc | Indirect | High | Emerging | Distributed patterns |

**✓✓✓** = Primary strength, **✓✓** = Secondary strength

---

## Recommended Multi-Method Approach

### Phase 1: Primary Analyses (Weeks 1-4)

**Method 1: HMM** (Primary hypothesis testing)
- Detect state transitions as event boundaries
- Test H1: Anxiety × boundary count
- Characterize states via connectivity

**Method 2: GSBS** (Convergent validation)
- Independent boundary detection
- Validate HMM findings
- Test boundary strength hypothesis (H2)

**Expected Convergence**: r(HMM, GSBS) > 0.6 for boundary alignment

### Phase 2: Mechanistic Interpretation (Weeks 5-6)

**Method 3: Seed-Based Connectivity**
- Amygdala-PFC connectivity at boundaries vs non-boundaries
- Test H3: Attention patterns (via connectivity proxies)
- Test H4: Hippocampal-PMN connectivity

**Integration**: Use HMM/GSBS boundaries to define analysis windows

### Phase 3: Exploratory Validation (Weeks 7-8)

**Method 4: Sliding Window dFC** (Optional)
- Alternative state detection approach
- Validate state count from HMM
- Examine network switching

**Method 5: MVPA** (Optional)
- Pattern-based validation
- Sensitivity to distributed effects
- DMN-specific analysis

### Phase 4: Synthesis (Weeks 9-10)

**Multi-Method Convergence Analysis**:
```python
# Compute inter-method reliability
methods = ['HMM', 'GSBS', 'dFC', 'MVPA']
boundary_counts = np.array([
    [hmm_counts[s], gsbs_counts[s], dfc_counts[s], mvpa_counts[s]]
    for s in subjects
])

# Inter-method correlation
from scipy.stats import spearmanr
corr_matrix = np.corrcoef(boundary_counts.T)

print("Inter-Method Correlation:")
print(pd.DataFrame(corr_matrix, index=methods, columns=methods))

# Composite boundary score (average across methods)
composite_score = np.mean(boundary_counts, axis=1)

# Ultimate test: Composite score × Anxiety
r, p = stats.pearsonr(anxiety_scores, composite_score)
print(f"\nComposite Boundary Score × Anxiety: r={r:.3f}, p={p:.4f}")
```

**Expected Outcome**: If hypothesis is robust, all methods should show positive correlation with anxiety.

---

## Implementation Recommendations

### Minimum Viable Analysis
For resource-constrained scenarios:
1. **HMM** (Primary method, well-validated)
2. **Seed-based connectivity** (Mechanistic interpretation)

**Timeline**: 6 weeks
**Software**: hmmlearn, nilearn, statsmodels

### Comprehensive Analysis
For thorough investigation:
1. **HMM** + **GSBS** (Convergent boundary detection)
2. **Seed-based connectivity** (Circuit mechanisms)
3. **Sliding window dFC** (Validation + exploratory)

**Timeline**: 10 weeks
**Software**: Add scikit-learn clustering

### Cutting-Edge Analysis
For methodological contribution:
1. All 5 methods
2. Multi-method synthesis
3. Hierarchical event structure (nested boundaries)

**Timeline**: 14 weeks
**Software**: All packages + custom implementations

---

## Quality Control and Validation

### Data Quality Checks
- **Head motion**: Exclude subjects with mean FD > 0.5mm
- **Signal quality**: DVARS outlier detection
- **Coverage**: Ensure full brain coverage in all subjects
- **Preprocessing**: Visual QC of registration, normalization

### Method-Specific Validation
- **HMM**: Test multiple K values (3-10), report cross-validation scores
- **GSBS**: Sensitivity analysis across window sizes (20-60 TRs)
- **Connectivity**: Ensure seed placement consistency across subjects
- **dFC**: Test window sizes (30, 45, 60 seconds)
- **MVPA**: Validate pattern stability via split-half reliability

### Statistical Robustness
- **Multiple comparisons**: FDR/FWE correction for neuroimaging
- **Confound control**: Regression with age, sex, motion, depression
- **Non-parametric tests**: If distributions non-normal
- **Effect sizes**: Report Cohen's d for all comparisons
- **Replication**: Split-sample validation if N permits

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
**Next Steps**: Dataset evaluation, initial pilot analysis
