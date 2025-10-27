#!/usr/bin/env python3
"""
Test Implementation: HMM-Based Event Boundary Detection
Anxiety × Event Boundary Research Project

Purpose: Test code generation and commit tracking system
Author: Supervisor + Forge Pod Agents
Date: 2025-10-27
Status: Testing/Development

This script implements a basic Hidden Markov Model approach for detecting
event boundaries in resting-state fMRI data, as outlined in the traditional
methodology proposals document.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class EventBoundaryHMM:
    """
    Hidden Markov Model for event boundary detection in fMRI time series.

    This class implements the HMM-based approach proposed for detecting
    spontaneous event boundaries in resting-state fMRI data and testing
    whether anxiety modulates boundary frequency.
    """

    def __init__(self, n_states_range=(3, 11), random_state=42):
        """
        Initialize HMM boundary detector.

        Parameters
        ----------
        n_states_range : tuple
            Range of state numbers to test (min, max)
        random_state : int
            Random seed for reproducibility
        """
        self.n_states_range = n_states_range
        self.random_state = random_state
        self.optimal_n_states = None
        self.model = None

    def select_optimal_states(self, X, n_folds=5):
        """
        Select optimal number of HMM states via cross-validation.

        Uses Bayesian Information Criterion (BIC) and cross-validated
        log-likelihood to determine the best number of states.

        Parameters
        ----------
        X : array, shape (n_timepoints, n_features)
            Input time series (e.g., ROI signals)
        n_folds : int
            Number of cross-validation folds

        Returns
        -------
        optimal_n : int
            Optimal number of states
        scores : dict
            Cross-validation scores for each state number
        """
        print("Selecting optimal number of states via cross-validation...")

        state_numbers = range(*self.n_states_range)
        cv_scores = []
        bic_scores = []

        for n_states in state_numbers:
            print(f"  Testing {n_states} states...")

            # Cross-validation
            kf = KFold(n_splits=n_folds, shuffle=False)
            fold_scores = []

            for train_idx, test_idx in kf.split(X):
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type='full',
                    n_iter=1000,
                    random_state=self.random_state,
                    verbose=False
                )

                try:
                    model.fit(X[train_idx])
                    score = model.score(X[test_idx])
                    fold_scores.append(score)
                except:
                    fold_scores.append(-np.inf)

            cv_scores.append(np.mean(fold_scores))

            # BIC (lower is better)
            model_full = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=1000,
                random_state=self.random_state,
                verbose=False
            )
            model_full.fit(X)
            bic = model_full.bic(X)
            bic_scores.append(bic)

        # Select optimal based on CV score
        optimal_idx = np.argmax(cv_scores)
        self.optimal_n_states = state_numbers[optimal_idx]

        scores = {
            'n_states': list(state_numbers),
            'cv_score': cv_scores,
            'bic': bic_scores,
            'optimal': self.optimal_n_states
        }

        print(f"✓ Optimal number of states: {self.optimal_n_states}")
        return self.optimal_n_states, scores

    def fit(self, X, n_states=None):
        """
        Fit HMM model to time series data.

        Parameters
        ----------
        X : array, shape (n_timepoints, n_features)
            Input time series
        n_states : int, optional
            Number of states. If None, uses optimal from cross-validation

        Returns
        -------
        self : EventBoundaryHMM
            Fitted model
        """
        if n_states is None:
            if self.optimal_n_states is None:
                raise ValueError("Must run select_optimal_states first or provide n_states")
            n_states = self.optimal_n_states

        print(f"Fitting HMM with {n_states} states...")

        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=1000,
            random_state=self.random_state,
            verbose=False
        )

        self.model.fit(X)
        print("✓ Model fitted successfully")

        return self

    def detect_boundaries(self, X):
        """
        Detect event boundaries from fitted HMM.

        Event boundaries occur at timepoints where the hidden state changes.

        Parameters
        ----------
        X : array, shape (n_timepoints, n_features)
            Input time series

        Returns
        -------
        boundaries : array
            Indices of detected event boundaries
        state_sequence : array
            Decoded state sequence
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Decode state sequence
        state_sequence = self.model.predict(X)

        # Detect boundaries (state transitions)
        boundaries = []
        for t in range(1, len(state_sequence)):
            if state_sequence[t] != state_sequence[t-1]:
                boundaries.append(t)

        boundaries = np.array(boundaries)

        return boundaries, state_sequence

    def compute_metrics(self, boundaries, state_sequence, tr=2.0, duration_min=10.0):
        """
        Compute summary metrics for event boundaries.

        Parameters
        ----------
        boundaries : array
            Boundary indices
        state_sequence : array
            Hidden state sequence
        tr : float
            Repetition time in seconds
        duration_min : float
            Scan duration in minutes

        Returns
        -------
        metrics : dict
            Dictionary of boundary metrics
        """
        n_timepoints = len(state_sequence)
        n_boundaries = len(boundaries)

        # Boundary rate (per minute)
        boundary_rate = n_boundaries / duration_min

        # Mean state duration (in seconds)
        state_durations = np.diff(np.concatenate([[0], boundaries, [n_timepoints]]))
        mean_state_duration = np.mean(state_durations) * tr

        # State occupancy
        unique_states = np.unique(state_sequence)
        state_occupancy = {
            int(s): np.sum(state_sequence == s) / n_timepoints
            for s in unique_states
        }

        metrics = {
            'n_boundaries': n_boundaries,
            'boundary_rate_per_min': boundary_rate,
            'mean_state_duration_sec': mean_state_duration,
            'state_occupancy': state_occupancy,
            'n_states_observed': len(unique_states)
        }

        return metrics


def simulate_fmri_data(n_subjects=80, n_timepoints=300, n_rois=116,
                       anxiety_effect_size=0.35, seed=42):
    """
    Simulate fMRI time series data with anxiety effect on boundaries.

    This generates synthetic data where high-anxiety individuals have
    more frequent event boundaries (state transitions).

    Parameters
    ----------
    n_subjects : int
        Number of simulated subjects
    n_timepoints : int
        Number of time points (TRs)
    n_rois : int
        Number of brain regions
    anxiety_effect_size : float
        Correlation between anxiety and boundary count
    seed : int
        Random seed

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'timeseries': list of arrays (n_subjects × n_timepoints × n_rois)
        - 'anxiety': array of anxiety scores (n_subjects,)
        - 'demographics': DataFrame with subject info
    """
    np.random.seed(seed)

    print(f"Simulating data for {n_subjects} subjects...")

    # Generate anxiety scores (STAI-T distribution)
    anxiety_scores = np.random.normal(loc=42, scale=10, size=n_subjects)
    anxiety_scores = np.clip(anxiety_scores, 20, 80)  # STAI-T range

    # Generate demographics
    demographics = pd.DataFrame({
        'subject_id': [f'sub-{i+1:03d}' for i in range(n_subjects)],
        'age': np.random.randint(20, 40, n_subjects),
        'sex': np.random.choice(['M', 'F'], n_subjects),
        'STAI_T': anxiety_scores,
        'mean_fd': np.random.uniform(0.1, 0.4, n_subjects)  # Head motion
    })

    # Generate time series with anxiety-modulated boundaries
    timeseries = []

    for subj_idx in range(n_subjects):
        anxiety = anxiety_scores[subj_idx]

        # Anxiety modulates number of states (more anxiety → more states → more boundaries)
        base_states = 5
        anxiety_factor = (anxiety - 42) / 10  # Standardize around mean
        n_states = int(base_states + anxiety_factor * anxiety_effect_size * 3)
        n_states = np.clip(n_states, 3, 10)

        # Generate state sequence
        state_sequence = []
        current_state = 0

        for t in range(n_timepoints):
            # Transition probability increases with anxiety
            base_transition_prob = 0.05
            anxiety_modulated_prob = base_transition_prob * (1 + anxiety_factor * anxiety_effect_size)

            if np.random.rand() < anxiety_modulated_prob:
                # Transition to new state
                current_state = np.random.randint(0, n_states)

            state_sequence.append(current_state)

        state_sequence = np.array(state_sequence)

        # Generate ROI time series based on states
        ts = np.zeros((n_timepoints, n_rois))

        for state_id in range(n_states):
            # Each state has characteristic ROI pattern
            state_pattern = np.random.randn(n_rois) * 0.5

            # Assign pattern to timepoints in this state
            state_mask = (state_sequence == state_id)
            ts[state_mask, :] = state_pattern + np.random.randn(np.sum(state_mask), n_rois) * 0.2

        # Add temporal autocorrelation (AR(1) process)
        for roi in range(n_rois):
            for t in range(1, n_timepoints):
                ts[t, roi] += 0.3 * ts[t-1, roi]

        timeseries.append(ts)

    print("✓ Simulation complete")

    return {
        'timeseries': timeseries,
        'anxiety': anxiety_scores,
        'demographics': demographics
    }


def run_group_analysis(subject_metrics, demographics, output_dir='results'):
    """
    Perform group-level statistical analysis.

    Tests hypothesis that anxiety correlates with event boundary frequency.

    Parameters
    ----------
    subject_metrics : list of dict
        Boundary metrics for each subject
    demographics : DataFrame
        Subject demographics including anxiety scores
    output_dir : str
        Directory to save results

    Returns
    -------
    results : dict
        Statistical test results
    """
    print("\nPerforming group-level analysis...")

    # Combine metrics and demographics
    df = pd.DataFrame(subject_metrics)
    df = df.merge(demographics, on='subject_id')

    # Primary hypothesis: Anxiety × Boundary Count
    print("\n=== H1: Trait Anxiety × Event Boundary Count ===")
    r_boundary, p_boundary = stats.pearsonr(df['STAI_T'], df['n_boundaries'])
    print(f"Pearson r = {r_boundary:.3f}, p = {p_boundary:.4f}")

    if p_boundary < 0.05:
        print("✓ Significant positive correlation (supports H1)")
    else:
        print("✗ No significant correlation")

    # Secondary: Anxiety × Boundary Rate
    print("\n=== Secondary: Anxiety × Boundary Rate ===")
    r_rate, p_rate = stats.pearsonr(df['STAI_T'], df['boundary_rate_per_min'])
    print(f"Pearson r = {r_rate:.3f}, p = {p_rate:.4f}")

    # Group comparison (median split)
    print("\n=== Group Comparison (High vs Low Anxiety) ===")
    median_anxiety = df['STAI_T'].median()
    high_anxiety = df[df['STAI_T'] >= median_anxiety]
    low_anxiety = df[df['STAI_T'] < median_anxiety]

    t_stat, p_val = stats.ttest_ind(
        high_anxiety['n_boundaries'],
        low_anxiety['n_boundaries']
    )

    cohen_d = (high_anxiety['n_boundaries'].mean() - low_anxiety['n_boundaries'].mean()) / \
              np.sqrt((high_anxiety['n_boundaries'].std()**2 + low_anxiety['n_boundaries'].std()**2) / 2)

    print(f"High Anxiety: M = {high_anxiety['n_boundaries'].mean():.1f}, SD = {high_anxiety['n_boundaries'].std():.1f}")
    print(f"Low Anxiety:  M = {low_anxiety['n_boundaries'].mean():.1f}, SD = {low_anxiety['n_boundaries'].std():.1f}")
    print(f"t({len(df)-2}) = {t_stat:.2f}, p = {p_val:.4f}, d = {cohen_d:.2f}")

    # Multiple regression controlling for confounds
    print("\n=== Multiple Regression (Controlling Confounds) ===")
    from statsmodels.formula.api import ols

    model = ols('n_boundaries ~ STAI_T + age + C(sex) + mean_fd', data=df).fit()
    print(model.summary().tables[1])

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save results
    results = {
        'correlation': {
            'r': r_boundary,
            'p': p_boundary
        },
        'group_comparison': {
            't': t_stat,
            'p': p_val,
            'cohens_d': cohen_d,
            'high_anxiety_mean': high_anxiety['n_boundaries'].mean(),
            'low_anxiety_mean': low_anxiety['n_boundaries'].mean()
        },
        'regression': {
            'beta_anxiety': model.params['STAI_T'],
            'p_anxiety': model.pvalues['STAI_T'],
            'r_squared': model.rsquared
        }
    }

    # Generate visualizations
    create_visualizations(df, output_path)

    # Save DataFrame
    df.to_csv(output_path / 'subject_metrics.csv', index=False)

    print(f"\n✓ Results saved to {output_path}/")

    return results


def create_visualizations(df, output_dir):
    """
    Create publication-quality visualizations.

    Parameters
    ----------
    df : DataFrame
        Subject metrics and demographics
    output_dir : Path
        Directory to save figures
    """
    print("\nGenerating visualizations...")

    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)

    # Figure 1: Scatterplot - Anxiety × Boundary Count
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(
        x='STAI_T',
        y='n_boundaries',
        data=df,
        ax=ax,
        scatter_kws={'alpha': 0.6, 's': 50},
        line_kws={'color': 'red', 'lw': 2}
    )

    r, p = stats.pearsonr(df['STAI_T'], df['n_boundaries'])
    ax.text(
        0.05, 0.95,
        f'r = {r:.3f}, p = {p:.4f}',
        transform=ax.transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('Trait Anxiety (STAI-T Score)')
    ax.set_ylabel('Event Boundary Count')
    ax.set_title('Anxiety Modulation of Event Boundary Detection')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_anxiety_boundary_correlation.png', dpi=300)
    plt.close()

    # Figure 2: Group Comparison
    fig, ax = plt.subplots(figsize=(6, 6))

    median_anxiety = df['STAI_T'].median()
    df['anxiety_group'] = df['STAI_T'].apply(
        lambda x: 'High Anxiety' if x >= median_anxiety else 'Low Anxiety'
    )

    sns.boxplot(
        x='anxiety_group',
        y='n_boundaries',
        data=df,
        ax=ax,
        palette={'High Anxiety': 'salmon', 'Low Anxiety': 'skyblue'}
    )

    sns.swarmplot(
        x='anxiety_group',
        y='n_boundaries',
        data=df,
        ax=ax,
        color='black',
        alpha=0.5,
        size=4
    )

    ax.set_xlabel('Anxiety Group')
    ax.set_ylabel('Event Boundary Count')
    ax.set_title('Event Boundaries: High vs Low Anxiety')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_group_comparison.png', dpi=300)
    plt.close()

    print("✓ Figures saved")


def main():
    """
    Main analysis pipeline for testing HMM boundary detection.

    This demonstrates the complete workflow:
    1. Simulate fMRI data with anxiety effect
    2. Detect boundaries using HMM
    3. Test anxiety-boundary correlation
    4. Generate results and visualizations
    """
    print("="*60)
    print("HMM-Based Event Boundary Detection - Test Pipeline")
    print("="*60)

    # Parameters
    N_SUBJECTS = 80
    N_TIMEPOINTS = 300  # 10 minutes at TR=2s
    N_ROIS = 116  # AAL atlas
    TR = 2.0
    DURATION_MIN = N_TIMEPOINTS * TR / 60

    # Step 1: Simulate data
    print("\n[Step 1] Simulating fMRI data...")
    sim_data = simulate_fmri_data(
        n_subjects=N_SUBJECTS,
        n_timepoints=N_TIMEPOINTS,
        n_rois=N_ROIS,
        anxiety_effect_size=0.35,
        seed=42
    )

    # Step 2: Analyze each subject
    print("\n[Step 2] Detecting boundaries for each subject...")
    subject_metrics = []

    hmm_detector = EventBoundaryHMM(n_states_range=(3, 11), random_state=42)

    for subj_idx, ts in enumerate(sim_data['timeseries']):
        subj_id = sim_data['demographics'].loc[subj_idx, 'subject_id']

        if subj_idx == 0:
            # Optimize states on first subject (in practice, do group-level)
            optimal_n, scores = hmm_detector.select_optimal_states(ts)

        # Fit and detect
        hmm_detector.fit(ts, n_states=hmm_detector.optimal_n_states)
        boundaries, state_seq = hmm_detector.detect_boundaries(ts)
        metrics = hmm_detector.compute_metrics(boundaries, state_seq, TR, DURATION_MIN)

        metrics['subject_id'] = subj_id
        subject_metrics.append(metrics)

        if (subj_idx + 1) % 20 == 0:
            print(f"  Processed {subj_idx + 1}/{N_SUBJECTS} subjects")

    print(f"✓ Processed all {N_SUBJECTS} subjects")

    # Step 3: Group analysis
    print("\n[Step 3] Group-level statistical analysis...")
    results = run_group_analysis(
        subject_metrics,
        sim_data['demographics'],
        output_dir='results_test'
    )

    # Step 4: Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nPrimary Result:")
    print(f"  Anxiety × Boundary Count: r = {results['correlation']['r']:.3f}")
    print(f"  p-value: {results['correlation']['p']:.4f}")
    print(f"  {'✓ SIGNIFICANT' if results['correlation']['p'] < 0.05 else '✗ NOT SIGNIFICANT'}")

    print(f"\nGroup Comparison:")
    print(f"  High vs Low Anxiety: t = {results['group_comparison']['t']:.2f}")
    print(f"  Cohen's d = {results['group_comparison']['cohens_d']:.2f}")
    print(f"  {'✓ SIGNIFICANT' if results['group_comparison']['p'] < 0.05 else '✗ NOT SIGNIFICANT'}")

    print(f"\nRegression:")
    print(f"  Beta (Anxiety): {results['regression']['beta_anxiety']:.3f}")
    print(f"  R² = {results['regression']['r_squared']:.3f}")

    print("\n" + "="*60)
    print("Test implementation successful!")
    print("Ready for real data analysis with Spacetop/ds002748")
    print("="*60)


if __name__ == '__main__':
    main()
