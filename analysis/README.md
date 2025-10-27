# Analysis Scripts

This directory contains analysis code for the Anxiety-Event Boundary fMRI research project.

## Current Scripts

### `test_hmm_boundary_detection.py`

**Purpose**: Test implementation of HMM-based event boundary detection

**Status**: Development/Testing

**Description**:
This script implements the Hidden Markov Model approach proposed in the traditional methodology document. It includes:

- `EventBoundaryHMM` class for boundary detection
- Optimal state number selection via cross-validation
- Simulated data generation with anxiety effects
- Group-level statistical analysis
- Visualization generation

**Usage**:
```bash
# Install dependencies
pip install numpy pandas scipy scikit-learn hmmlearn matplotlib seaborn statsmodels

# Run test analysis
python test_hmm_boundary_detection.py
```

**Output**:
- `results_test/subject_metrics.csv`: Per-subject boundary metrics
- `results_test/fig1_anxiety_boundary_correlation.png`: Scatterplot
- `results_test/fig2_group_comparison.png`: Group boxplot

**Expected Results** (with simulated data):
- Correlation: r ≈ 0.35, p < 0.001
- Group difference: d ≈ 1.2, p < 0.001

## Next Steps

1. **Real Data Analysis**:
   - Download Spacetop dataset (primary)
   - Download ds002748 from OpenNeuro (backup)
   - Apply `test_hmm_boundary_detection.py` to real resting-state fMRI

2. **Additional Methods**:
   - Implement GSBS boundary detection
   - Implement seed-based connectivity analysis
   - Implement dynamic functional connectivity

3. **Integration**:
   - Combine multiple methods for convergent validation
   - Create unified analysis pipeline

## Directory Structure

```
analysis/
├── README.md                           # This file
├── test_hmm_boundary_detection.py      # HMM test implementation
├── hmm_real_data.py                    # [To be created] Real data analysis
├── gsbs_boundary_detection.py          # [To be created] GSBS method
├── seed_connectivity.py                # [To be created] Connectivity analysis
└── utils/                              # [To be created] Helper functions
    ├── preprocessing.py
    ├── visualization.py
    └── statistics.py
```

## Dependencies

```
numpy>=1.21
pandas>=1.3
scipy>=1.7
scikit-learn>=1.0
hmmlearn>=0.3.0
nilearn>=0.9
nibabel>=3.2
matplotlib>=3.4
seaborn>=0.11
statsmodels>=0.13
```

## Notes

- All analysis code follows the methodology outlined in `docs/traditional_methods_proposals.md`
- Simulated data is used for testing; real data analysis pending dataset access
- Results from real data will be saved to `results/` (not `results_test/`)
