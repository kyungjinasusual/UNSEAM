#!/usr/bin/env python3
"""
Yang et al. (2023) Replication Script

Validates standard GaussianHMM implementation by comparing with the approach from:
"The default network dominates neural responses to evolving movie stories"
Nature Communications, 14, 4400

Usage:
    python scripts/validate_yang.py --data-dir ~/data/studyforrest --n-states 4
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hmm_baseline.hmmlearn_wrapper import HMMLearnWrapper


def load_studyforrest_data(data_dir: str, run: int = 1) -> dict:
    """
    Load StudyForrest data.

    Args:
        data_dir: Directory containing StudyForrest data
        run: Which run to load (1-8)

    Returns:
        Dictionary with data and metadata
    """
    data_path = Path(data_dir)

    # Try different file naming conventions
    possible_patterns = [
        f"run{run}_*.npy",
        f"*run{run}*.npy",
        f"forrest_run{run}.npy",
        f"studyforrest_run{run}.npy",
    ]

    data_files = []
    for pattern in possible_patterns:
        data_files.extend(list(data_path.glob(pattern)))

    if not data_files:
        # Try to find any npy files
        all_npy = list(data_path.glob("*.npy"))
        if all_npy:
            print(f"Available .npy files in {data_path}:")
            for f in sorted(all_npy):
                print(f"  - {f.name}")
            raise FileNotFoundError(
                f"Could not find run {run} data. See available files above."
            )
        else:
            raise FileNotFoundError(
                f"No .npy files found in {data_path}"
            )

    # Load first matching file
    data_file = data_files[0]
    data = np.load(data_file)
    print(f"Loaded {data_file.name}: shape = {data.shape}")

    return {
        'data': data,
        'file': str(data_file),
        'run': run
    }


def run_yang_hmm(data: np.ndarray,
                 n_states: int = 4,
                 covariance_type: str = 'diag',
                 n_iter: int = 100,
                 n_init: int = 10) -> dict:
    """
    Run standard GaussianHMM (Yang et al. style).

    Args:
        data: fMRI data, shape (n_subjects, n_timepoints, n_features) or (n_timepoints, n_features)
        n_states: Number of hidden states
        covariance_type: Type of covariance parameters ('diag', 'full', etc.)
        n_iter: Number of EM iterations
        n_init: Number of initializations

    Returns:
        Dictionary with results
    """
    # Prepare data
    if data.ndim == 3:
        # Average across subjects for group-level analysis
        group_data = np.mean(data, axis=0)
    else:
        group_data = data

    print(f"\nRunning Yang-style GaussianHMM with {n_states} states...")
    print(f"  Data shape: {group_data.shape}")
    print(f"  Covariance type: {covariance_type}")
    print(f"  Max iterations: {n_iter}")
    print(f"  N initializations: {n_init}")

    model = HMMLearnWrapper(
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        n_init=n_init,
        random_seed=42
    )

    model.fit([group_data])

    # Get state sequence
    states = model.predict(group_data)

    # Get boundaries
    boundaries = model.get_event_boundaries(group_data)

    # Compute state occupancies
    state_counts = np.bincount(states, minlength=n_states)
    occupancies = state_counts / len(states)

    # Compute mean state durations
    state_changes = np.where(np.diff(states) != 0)[0]
    if len(state_changes) > 1:
        durations = np.diff(np.concatenate([[0], state_changes, [len(states)]]))
        mean_duration = np.mean(durations)
    else:
        mean_duration = len(states)

    return {
        'model': model,
        'states': states,
        'boundaries': boundaries,
        'n_states': n_states,
        'occupancies': occupancies,
        'mean_duration': mean_duration,
        'n_transitions': len(state_changes),
        'log_likelihood': model.model_.score(group_data) if model.model_ else None
    }


def select_optimal_n_states(data: np.ndarray,
                            state_range: list = [2, 3, 4, 5, 6, 8, 10],
                            criterion: str = 'bic') -> dict:
    """
    Select optimal number of states using BIC/AIC.

    Args:
        data: fMRI data
        state_range: List of state counts to try
        criterion: 'bic' or 'aic'

    Returns:
        Dictionary with model selection results
    """
    if data.ndim == 3:
        group_data = np.mean(data, axis=0)
    else:
        group_data = data

    print(f"\nSelecting optimal n_states using {criterion.upper()}...")

    results = {
        'n_states': [],
        'bic': [],
        'aic': [],
        'log_likelihood': []
    }

    for n_states in state_range:
        print(f"  Testing n_states = {n_states}...", end=" ")

        model = HMMLearnWrapper(
            n_states=n_states,
            covariance_type='diag',
            n_iter=100,
            n_init=5,
            random_seed=42
        )

        try:
            model.fit([group_data])

            # Compute BIC/AIC
            n_samples, n_features = group_data.shape
            ll = model.model_.score(group_data) * n_samples

            # Number of parameters
            n_params = (n_states * n_features +  # means
                       n_states * n_features +   # variances (diag)
                       n_states * (n_states - 1))  # transition matrix

            bic = -2 * ll + n_params * np.log(n_samples)
            aic = -2 * ll + 2 * n_params

            results['n_states'].append(n_states)
            results['bic'].append(bic)
            results['aic'].append(aic)
            results['log_likelihood'].append(ll)

            print(f"LL={ll:.1f}, BIC={bic:.1f}")

        except Exception as e:
            print(f"Failed: {e}")

    # Find optimal
    if criterion == 'bic':
        optimal_idx = np.argmin(results['bic'])
    else:
        optimal_idx = np.argmin(results['aic'])

    results['optimal_n_states'] = results['n_states'][optimal_idx]
    results['criterion'] = criterion

    print(f"\nOptimal n_states = {results['optimal_n_states']} (min {criterion.upper()})")

    return results


def cross_validate_model(data: np.ndarray,
                         n_states: int = 4,
                         n_folds: int = 5) -> dict:
    """
    Cross-validate HMM model.

    Args:
        data: fMRI data (n_timepoints, n_features)
        n_states: Number of states
        n_folds: Number of CV folds

    Returns:
        Dictionary with CV results
    """
    if data.ndim == 3:
        group_data = np.mean(data, axis=0)
    else:
        group_data = data

    n_samples = len(group_data)
    fold_size = n_samples // n_folds

    print(f"\n{n_folds}-fold Cross-Validation (n_states={n_states})...")

    train_scores = []
    test_scores = []

    for fold in range(n_folds):
        # Create train/test split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples

        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start),
                                    np.arange(test_end, n_samples)])

        train_data = group_data[train_idx]
        test_data = group_data[test_idx]

        # Train model
        model = HMMLearnWrapper(
            n_states=n_states,
            covariance_type='diag',
            n_iter=100,
            n_init=3,
            random_seed=42 + fold
        )

        model.fit([train_data])

        # Evaluate
        train_ll = model.model_.score(train_data)
        test_ll = model.model_.score(test_data)

        train_scores.append(train_ll)
        test_scores.append(test_ll)

        print(f"  Fold {fold+1}: Train LL={train_ll:.2f}, Test LL={test_ll:.2f}")

    return {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'mean_train': np.mean(train_scores),
        'mean_test': np.mean(test_scores),
        'std_train': np.std(train_scores),
        'std_test': np.std(test_scores),
        'n_folds': n_folds,
        'n_states': n_states
    }


def plot_yang_results(results: dict,
                      model_selection: dict = None,
                      output_dir: Path = None):
    """Generate visualization plots for Yang-style analysis."""

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. State sequence plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    states = results['states']
    n_timepoints = len(states)

    # State sequence
    ax = axes[0]
    ax.plot(states, 'b-', linewidth=0.8)
    ax.set_ylabel('State')
    ax.set_xlabel('TR')
    ax.set_title(f'HMM State Sequence ({results["n_states"]} states)')
    ax.set_xlim(0, n_timepoints)

    # Mark boundaries
    for b in results['boundaries']:
        ax.axvline(b, color='red', alpha=0.5, linewidth=1)

    # State occupancy
    ax = axes[1]
    bars = ax.bar(range(results['n_states']), results['occupancies'],
                  color='steelblue', alpha=0.7)
    ax.set_xlabel('State')
    ax.set_ylabel('Fractional Occupancy')
    ax.set_title('State Occupancies')
    ax.set_xticks(range(results['n_states']))

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / 'state_sequence.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Model selection plot
    if model_selection and len(model_selection['n_states']) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(model_selection['n_states'], model_selection['bic'],
                'bo-', label='BIC', linewidth=2, markersize=8)
        ax.plot(model_selection['n_states'], model_selection['aic'],
                'rs--', label='AIC', linewidth=2, markersize=8)

        # Mark optimal
        opt_idx = model_selection['n_states'].index(model_selection['optimal_n_states'])
        ax.axvline(model_selection['optimal_n_states'], color='green',
                   linestyle=':', alpha=0.7, label=f'Optimal ({model_selection["optimal_n_states"]})')

        ax.set_xlabel('Number of States')
        ax.set_ylabel('Information Criterion')
        ax.set_title('Model Selection: BIC/AIC vs Number of States')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'model_selection.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Transition matrix heatmap
    if results['model'].model_ is not None:
        fig, ax = plt.subplots(figsize=(8, 6))

        trans_mat = results['model'].model_.transmat_
        sns.heatmap(trans_mat, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=range(results['n_states']),
                   yticklabels=range(results['n_states']),
                   ax=ax)
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        ax.set_title('Transition Probability Matrix')

        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'transition_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate GaussianHMM implementation (Yang et al. 2023 style)'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing StudyForrest or test data')
    parser.add_argument('--run', type=int, default=1,
                        help='Which run to analyze (default: 1)')
    parser.add_argument('--n-states', type=int, default=4,
                        help='Number of HMM states (default: 4)')
    parser.add_argument('--covariance-type', type=str, default='diag',
                        choices=['diag', 'full', 'spherical', 'tied'],
                        help='Covariance type (default: diag)')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Number of EM iterations (default: 100)')
    parser.add_argument('--select-n-states', action='store_true',
                        help='Run model selection to find optimal n_states')
    parser.add_argument('--cross-validate', action='store_true',
                        help='Run cross-validation')
    parser.add_argument('--output-dir', type=str, default='results/validation/yang',
                        help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Yang 2023 Replication - GaussianHMM Validation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output: {output_dir}")

    # Load data
    try:
        data_info = load_studyforrest_data(args.data_dir, args.run)
        data = data_info['data']
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo download StudyForrest data:")
        print("  1. Visit: https://drive.google.com/drive/folders/1Fq0XzNU0qN6bIFVhH3pBwXqPKOWznx6t")
        print("  2. Download the preprocessed fMRI data")
        print("  3. Or use: pip install gdown && gdown --folder <URL>")
        sys.exit(1)

    model_selection = None
    cv_results = None

    # Model selection
    if args.select_n_states:
        model_selection = select_optimal_n_states(
            data,
            state_range=[2, 3, 4, 5, 6, 8, 10],
            criterion='bic'
        )
        args.n_states = model_selection['optimal_n_states']

    # Main HMM analysis
    results = run_yang_hmm(
        data,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        n_iter=args.n_iter
    )

    # Cross-validation
    if args.cross_validate:
        cv_results = cross_validate_model(data, n_states=args.n_states)

    # Print summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Number of states: {results['n_states']}")
    print(f"Number of transitions: {results['n_transitions']}")
    print(f"Mean state duration: {results['mean_duration']:.1f} TRs")
    print(f"Log-likelihood: {results['log_likelihood']:.2f}")
    print(f"\nState occupancies:")
    for i, occ in enumerate(results['occupancies']):
        print(f"  State {i}: {occ:.3f} ({occ*100:.1f}%)")

    if cv_results:
        print(f"\nCross-Validation Results:")
        print(f"  Mean Train LL: {cv_results['mean_train']:.2f} ± {cv_results['std_train']:.2f}")
        print(f"  Mean Test LL: {cv_results['mean_test']:.2f} ± {cv_results['std_test']:.2f}")

    # Generate plots
    plot_yang_results(results, model_selection, output_dir)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(args.data_dir),
        'data_file': data_info['file'],
        'n_states': results['n_states'],
        'covariance_type': args.covariance_type,
        'n_transitions': results['n_transitions'],
        'mean_duration': float(results['mean_duration']),
        'log_likelihood': float(results['log_likelihood']) if results['log_likelihood'] else None,
        'occupancies': results['occupancies'].tolist(),
        'boundaries': results['boundaries'].tolist() if isinstance(results['boundaries'], np.ndarray) else results['boundaries'],
    }

    if model_selection:
        summary['model_selection'] = {
            'n_states_tested': model_selection['n_states'],
            'bic_values': model_selection['bic'],
            'optimal_n_states': model_selection['optimal_n_states']
        }

    if cv_results:
        summary['cross_validation'] = {
            'mean_train_ll': cv_results['mean_train'],
            'mean_test_ll': cv_results['mean_test'],
            'std_train_ll': cv_results['std_train'],
            'std_test_ll': cv_results['std_test']
        }

    with open(output_dir / 'yang_validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Validation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
