# Research Planning Summary: Anxiety × Event Boundary Detection in fMRI

**Date**: 2025-10-27
**Status**: Planning Phase Complete
**Next Phase**: Dataset Access and Pilot Analysis

---

## Project Overview

**Research Question**: How does anxiety modulate event boundary detection in the brain during resting-state fMRI?

**Primary Hypothesis (H1)**: Trait anxiety correlates positively with event boundary frequency
- Mechanism: Anxiety → intolerance of uncertainty → lower prediction error threshold → more boundaries
- Expected: r(STAI-T, boundary count) > 0.3, p < 0.05

**Theoretical Framework**:
- Event Segmentation Theory (Zacks et al., 2007)
- Predictive processing models of anxiety (Clark, 2013; Grupe & Nitschke, 2013)
- DMN-Salience Network dynamics

---

## Completed Deliverables

### 1. Literature Review ✓

**File**: `/docs/literature_review_event_boundaries_anxiety.md`

**Sections**:
1. Event Boundary Detection in Cognitive Neuroscience
   - Event Segmentation Theory foundations
   - Neural substrates (DMN, MTL, PFC)
   - Temporal dynamics and connectivity patterns

2. Anxiety and Predictive Processing
   - Computational models of anxiety
   - Neural circuits (amygdala-PFC, DMN alterations)
   - Intolerance of uncertainty

3. Anxiety Effects on Perception and Event Segmentation
   - Behavioral evidence (hypervigilance, attentional bias)
   - Neural evidence (limited)
   - Resting-state event structure

4. Traditional Analytical Methods
   - Behavioral (human segmentation)
   - fMRI techniques (GLM, HMM, GSBS, dFC, MVPA)
   - Statistical approaches

5. Research Gaps and Novel Contributions
   - No prior work on anxiety × event boundaries
   - Resting-state event structure understudied
   - Transformer applications to neuroscience

**Key Finding**: **Research gap identified** - No studies have examined anxiety effects on neural event boundary detection in resting-state fMRI.

**Word Count**: ~18,000 words
**References**: 18 key papers cited

---

### 2. Traditional Methodology Proposals ✓

**File**: `/docs/traditional_methods_proposals.md`

**Five Methods Proposed**:

#### Method 1: Hidden Markov Model (HMM) - PRIMARY
- Detects latent brain states and transitions
- Data-driven boundary detection
- Advantages: Established, probabilistic framework
- Expected: High anxiety → 28.5 boundaries, Low anxiety → 21.2 boundaries

#### Method 2: Greedy State Boundary Search (GSBS) - VALIDATION
- Direct boundary optimization algorithm
- Maximizes correlation differences across boundaries
- Convergent validation with HMM
- Boundary strength quantification

#### Method 3: Seed-Based Functional Connectivity - MECHANISTIC
- Amygdala-PFC connectivity analysis
- Connectivity at boundaries vs non-boundaries
- Tests neural mechanism hypotheses
- Clinical relevance (biomarkers)

#### Method 4: Sliding Window Dynamic FC - EXPLORATORY
- Time-varying connectivity states
- Network switching patterns
- State occupancy analysis
- Complements static connectivity

#### Method 5: MVPA Pattern Analysis - VALIDATION
- Distributed pattern similarity
- Sensitive to subtle changes
- DMN-specific analysis
- Convergent evidence

**Recommended Approach**: Multi-method convergent validation
- Primary: HMM + GSBS
- Mechanistic: Seed-based connectivity
- Optional: dFC + MVPA for robustness

**Timeline**: 10-14 weeks for comprehensive analysis

**Word Count**: ~15,000 words
**Code Examples**: Python implementations for all methods

---

### 3. Dataset Evaluation ✓

**File**: `/docs/dataset_evaluation_emofilm_spacetop.md`

#### Dataset 1: Emo-Film Dataset
**Feasibility Score**: 4/10

**Status**:
- No single accessible "emo-film" dataset identified
- Multiple emotional film studies exist but lack anxiety measures
- Typical N = 15-50 (underpowered)
- Task-based (not ideal for spontaneous boundaries)

**Recommendation**: Secondary validation dataset only (if accessible)

#### Dataset 2: Spacetop Dataset ✓✓✓
**Feasibility Score**: 9/10

**Details**:
- **PIs**: Luke Chang, Marianne Reddan, Tor Wager (Dartmouth)
- **N**: 100-200 participants (excellent power)
- **Data**:
  - Resting-state fMRI (multiple runs)
  - Naturalistic movie viewing (validation)
  - Comprehensive behavioral assessments
- **Anxiety Measures**: STAI (expected), PANAS, IUS, Neuroticism
- **Quality**: High (multiband, modern acquisition)
- **Format**: BIDS-compliant
- **Access**: Public or restricted-open (pending verification)

**Recommendation**: **PRIMARY DATASET**

**Power Analysis**:
- N = 150, effect size r = 0.35 → Power = 0.98 ✓
- Even N = 100 → Power = 0.90 ✓

#### Backup Datasets
1. **OpenNeuro ds002748** (Social Anxiety)
   - N = 70, immediate access
   - STAI included, resting-state available
   - Use for pilot or replication

2. **HCP** (Human Connectome Project)
   - N = 1200, excellent resting-state
   - Proxy anxiety (Neuroticism, DSM scales)
   - Large-scale validation

**Word Count**: ~8,000 words
**Action Items**: Contact Spacetop authors, download ds002748 backup

---

### 4. Test Implementation ✓

**File**: `/analysis/test_hmm_boundary_detection.py`

**Features**:
- Complete HMM-based boundary detection pipeline
- `EventBoundaryHMM` class with methods:
  - Optimal state selection (cross-validation)
  - Model fitting and boundary detection
  - Metrics computation
- Simulated fMRI data generator (anxiety effect built-in)
- Group-level statistical analysis
- Publication-quality visualizations

**Simulation Parameters**:
- N = 80 subjects
- 300 TRs (10 minutes at TR=2s)
- 116 ROIs (AAL atlas)
- Anxiety effect size = 0.35
- Expected results match theoretical predictions

**Output**:
- Subject-level metrics (CSV)
- Correlation scatterplot
- Group comparison boxplot

**Status**: Tested successfully with simulated data
**Ready**: For real data from Spacetop/ds002748

**Lines of Code**: ~620 lines with documentation

---

## Summary Statistics

**Total Documentation**: ~41,000 words across 3 documents
**Code Implementation**: 1 complete test script (620 lines)
**Methods Proposed**: 5 traditional approaches with detailed protocols
**Datasets Evaluated**: 2 primary + 2 backup options
**Expected Timeline**: 14 weeks from data access to results

---

## Next Steps and Action Items

### Immediate (Week 1)

**Dataset Access**:
- [ ] Contact Luke Chang for Spacetop dataset access
  - Email: luke.j.chang@dartmouth.edu
  - Subject: "Data Access Request for Spacetop Dataset"
- [ ] Download OpenNeuro ds002748 as backup/pilot
- [ ] Check OpenNeuro for "spacetop" keyword

**Pilot Analysis Setup**:
- [ ] Install required Python packages
- [ ] Test `test_hmm_boundary_detection.py` on local machine
- [ ] Prepare preprocessing pipeline (fMRIPrep)

### Short-term (Weeks 2-4)

**If Spacetop Access Granted**:
- [ ] Download dataset (~500 GB)
- [ ] Quality control (motion, SNR, coverage)
- [ ] Extract ROI time series
- [ ] Verify anxiety measures (STAI-T distribution)

**If Spacetop Pending**:
- [ ] Run pilot analysis on ds002748 (N=70)
- [ ] Validate HMM pipeline with real data
- [ ] Implement GSBS method
- [ ] Test seed-based connectivity

### Medium-term (Weeks 5-10)

**Primary Analysis** (with Spacetop or ds002748):
1. HMM state detection and boundary identification
2. GSBS convergent validation
3. Statistical analysis (correlation, regression)
4. Seed-based connectivity at boundaries
5. Results visualization

**Expected Milestones**:
- Week 6: Primary hypothesis tested (H1)
- Week 8: Mechanistic analysis complete (H3, H4)
- Week 10: Multi-method synthesis

### Long-term (Weeks 11-14)

**Validation and Extension**:
- [ ] Dynamic FC analysis (optional)
- [ ] MVPA pattern analysis (optional)
- [ ] Multi-method convergence assessment
- [ ] Prepare manuscript figures and tables

**Manuscript Preparation**:
- [ ] Introduction (use literature review)
- [ ] Methods (use methodology proposals)
- [ ] Results (from analyses)
- [ ] Discussion (interpret findings)

---

## Research Contributions

### Theoretical
- First empirical test of anxiety effects on event boundary detection
- Extension of Event Segmentation Theory to clinical neuroscience
- Validation of predictive coding models of anxiety

### Methodological
- Novel application of HMM to resting-state event segmentation
- Multi-method convergent validation framework
- Hybrid approach: data-driven detection + theory-driven validation

### Clinical
- Event boundary metrics as potential anxiety biomarkers
- Mechanistic insights (DMN-SN switching, memory encoding)
- Treatment targets identification

---

## Risk Mitigation

### Risk 1: Spacetop Dataset Not Accessible
**Mitigation**:
- Use OpenNeuro ds002748 (N=70, immediate access)
- Supplement with HCP data (proxy anxiety, N=1200)
- Still publishable with smaller N

### Risk 2: Anxiety Measures Insufficient
**Mitigation**:
- Create composite anxiety score (Neuroticism + Negative Affect)
- Validate proxy against STAI in subset
- Dimensional approach robust to measurement

### Risk 3: No Anxiety Effect Found
**Mitigation**:
- Methodological contribution remains (event detection in rest)
- Explore non-linear relationships (inverted-U)
- Test alternative hypotheses (state anxiety, subgroups)

### Risk 4: Methods Don't Converge
**Mitigation**:
- Each method addresses different aspect (states vs boundaries)
- Partial convergence still informative
- Method comparison as secondary aim

---

## Resource Requirements

### Computational
- **Storage**: ~500 GB for Spacetop, ~50 GB for ds002748
- **Processing**: fMRIPrep (~8 hours per subject on GPU)
- **Analysis**: HMM/GSBS (~1 hour per subject on CPU)
- **Total**: ~2 TB storage, GPU access helpful

### Software
```python
# Core packages
numpy, pandas, scipy, scikit-learn
hmmlearn, nilearn, nibabel
matplotlib, seaborn, statsmodels

# Preprocessing
fmriprep (Docker/Singularity)
FreeSurfer (optional, for surfaces)

# Optional
pytorch (for future transformer work)
networkx (for graph analysis)
```

### Timeline
- **Data access**: 2-8 weeks (depending on dataset)
- **Quality control**: 1 week
- **Preprocessing**: 2-3 weeks (parallel processing)
- **Analysis**: 6-8 weeks
- **Manuscript**: 4-6 weeks
- **Total**: 4-6 months from data access to submission

---

## Commit Summary

**Commit Hash**: adaa860db1001b3fd5d9d6be3b4662ceee83cd18
**Commit Date**: 2025-10-27
**Files Added**: 5
- `docs/literature_review_event_boundaries_anxiety.md` (963 lines)
- `docs/traditional_methods_proposals.md` (1630 lines)
- `docs/dataset_evaluation_emofilm_spacetop.md` (867 lines)
- `analysis/test_hmm_boundary_detection.py` (622 lines)
- `analysis/README.md` (90 lines)

**Total Lines**: 4,172 lines of research planning and code

**Purpose**: Test commit tracking system and demonstrate research progress

---

## External Monitoring

**Dashboard URL**: http://147.47.200.154:3000/submit
**Local Monitor**: `/Users/ohkyungjin/Downloads/code_monitor`

**Test Objective**: Verify that this commit is properly tracked by the submission system

**Expected Behavior**:
1. Commit appears in dashboard within 24 hours
2. Contribution metrics updated
3. File changes recorded
4. Timestamp logged

**If Tracking Fails**:
- Integrate local code_monitor system
- Manual submission via web interface
- Verify git remote settings

---

## Contact Information

**Principal Investigator**: Kyungjin Oh
**Institution**: Seoul National University
**Email**: castella@snu.ac.kr

**Dataset Inquiries**:
- Spacetop: luke.j.chang@dartmouth.edu
- OpenNeuro: support@openneuro.org

---

**Document Status**: Complete
**Last Updated**: 2025-10-27 17:04
**Version**: 1.0
**Next Review**: Upon dataset access confirmation
