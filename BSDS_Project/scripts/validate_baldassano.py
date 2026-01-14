#!/usr/bin/env python3
"""
Baldassano et al. (2017) Replication Script

Validates HMM implementation by replicating key findings from:
"Discovering Event Structure in Continuous Narrative Perception and Memory"
Neuron, 95(3), 709-721

Usage:
    python scripts/validate_baldassano.py --data-dir ~/data/sherlock --n-events 25
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hmm_baseline import HMMEventSegment, HMMConfig
from hmm_baseline.hmmlearn_wrapper import HMMLearnWrapper
from hmm_baseline.data_loaders import (
    load_sherlock_figshare,
    get_sherlock_human_boundaries,
    SHERLOCK_HUMAN_BOUNDARIES
)
from hmm_baseline.comparison import boundary_match_score


def load_sherlock_data(data_dir: str, roi: str = 'AG') -> np.ndarray:
    """Load Sherlock data for specified ROI."""
    data_path = Path(data_dir)

    # ROI name mapping
    roi_files = {
        'AG': 'AG_movie_1TR.npy',           # Angular Gyrus
        'PMC': 'PMC_movie_1TR.npy',         # Posterior Medial Cortex
        'EAC': 'EAC_movie_1TR.npy',         # Early Auditory Cortex
        'PPA': 'PPA_movie_1TR.npy',         # Parahippocampal Place Area
        'PCC': 'PCC_movie_1TR.npy',         # Posterior Cingulate Cortex
    }

    if roi not in roi_files:
        # Try to find file directly
        npy_file = data_path / f"{roi}_movie_1TR.npy"
        if not npy_file.exists():
            npy_file = data_path / f"{roi}.npy"
    else:
        npy_file = data_path / roi_files[roi]

    if not npy_file.exists():
        # List available files
        available = list(data_path.glob("*.npy"))
        raise FileNotFoundError(
            f"ROI file not found: {npy_file}\n"
            f"Available files: {[f.name for f in available]}"
        )

    data = np.load(npy_file)
    print(f"Loaded {roi} data: shape = {data.shape}")

    # Expected shape: (n_subjects, n_timepoints, n_voxels) or (n_timepoints, n_voxels)
    if data.ndim == 3:
        # Multiple subjects
        print(f"  - {data.shape[0]} subjects, {data.shape[1]} TRs, {data.shape[2]} voxels")
    elif data.ndim == 2:
        print(f"  - {data.shape[0]} TRs, {data.shape[1]} voxels")

    return data


def run_hmm_baldassano(data: np.ndarray, n_events: int, n_iter: int = 100) -> dict:
    """
    Run Event-Sequential HMM (Baldassano style).

    Args:
        data: fMRI data, shape (n_subjects, n_timepoints, n_voxels) or (n_timepoints, n_voxels)
        n_events: Number of events to detect
        n_iter: Number of EM iterations

    Returns:
        Dictionary with results
    """
    # Prepare data
    if data.ndim == 3:
        # Average across subjects for group-level analysis (Baldassano approach)
        group_data = np.mean(data, axis=0)  # (T, V)
        data_list = [group_data]
    else:
        data_list = [data]

    print(f"\nRunning Baldassano-style HMM with {n_events} events...")

    config = HMMConfig(
        n_events=n_events,
        n_iter=n_iter,
        tol=1e-4,
        TR=1.5,
        random_seed=42
    )

    model = HMMEventSegment(config)
    model.fit(data_list)

    # Get boundaries
    boundaries = model.get_event_boundaries(subject_idx=0)

    return {
        'model': model,
        'boundaries': boundaries,
        'n_events': n_events,
        'method': 'baldassano'
    }


def run_hmm_yang(data: np.ndarray, n_states: int, n_iter: int = 100) -> dict:
    """
    Run Standard GaussianHMM (Yang style).

    Args:
        data: fMRI data
        n_states: Number of hidden states
        n_iter: Number of EM iterations

    Returns:
        Dictionary with results
    """
    # Prepare data
    if data.ndim == 3:
        group_data = np.mean(data, axis=0)
    else:
        group_data = data

    print(f"\nRunning Yang-style HMM with {n_states} states...")

    model = HMMLearnWrapper(
        n_states=n_states,
        covariance_type='diag',
        n_iter=n_iter,
        n_init=10,
        random_seed=42
    )

    model.fit([group_data])

    # Get boundaries
    boundaries = model.get_event_boundaries(group_data)

    return {
        'model': model,
        'boundaries': boundaries,
        'n_states': n_states,
        'method': 'yang'
    }


def evaluate_boundaries(pred_boundaries: np.ndarray,
                        human_boundaries: np.ndarray,
                        tolerance: int = 3) -> dict:
    """
    Evaluate predicted boundaries against human annotations.

    Args:
        pred_boundaries: Predicted boundary TRs
        human_boundaries: Human-annotated boundary TRs
        tolerance: Tolerance window in TRs (default ±3)

    Returns:
        Dictionary with evaluation metrics
    """
    scores = boundary_match_score(pred_boundaries, human_boundaries, tolerance=tolerance)

    # Additional analysis
    if len(pred_boundaries) > 0 and len(human_boundaries) > 0:
        # Find closest matches
        matches = []
        for hb in human_boundaries:
            diffs = np.abs(pred_boundaries - hb)
            closest_idx = np.argmin(diffs)
            closest_diff = diffs[closest_idx]
            if closest_diff <= tolerance:
                matches.append({
                    'human_tr': int(hb),
                    'pred_tr': int(pred_boundaries[closest_idx]),
                    'diff': int(closest_diff)
                })

        scores['matches'] = matches
        scores['n_matched'] = len(matches)

    return scores


def plot_results(results: dict, human_boundaries: np.ndarray,
                n_timepoints: int, output_dir: Path):
    """Generate visualization plots."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Boundary comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    tr_times = np.arange(n_timepoints) * 1.5 / 60  # Convert to minutes

    # Human boundaries
    ax = axes[0]
    ax.set_title('Human-Annotated Event Boundaries', fontsize=12)
    for hb in human_boundaries:
        ax.axvline(hb * 1.5 / 60, color='green', alpha=0.7, linewidth=2)
    ax.set_ylabel('Human')
    ax.set_xlim(0, n_timepoints * 1.5 / 60)
    ax.text(0.02, 0.9, f"N = {len(human_boundaries)} boundaries",
            transform=ax.transAxes, fontsize=10)

    # Model boundaries
    ax = axes[1]
    method = results['method']
    pred_boundaries = results['boundaries']
    ax.set_title(f'Model-Predicted Boundaries ({method.capitalize()})', fontsize=12)
    for pb in pred_boundaries:
        ax.axvline(pb * 1.5 / 60, color='blue', alpha=0.7, linewidth=2)
    ax.set_ylabel(f'{method.capitalize()}')
    ax.set_xlabel('Time (minutes)')
    ax.text(0.02, 0.9, f"N = {len(pred_boundaries)} boundaries",
            transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'boundary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Match visualization
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot human boundaries
    for i, hb in enumerate(human_boundaries):
        ax.axvline(hb, color='green', alpha=0.5, linewidth=1.5,
                   label='Human' if i == 0 else '')

    # Plot predicted boundaries
    for i, pb in enumerate(pred_boundaries):
        ax.axvline(pb, color='blue', alpha=0.5, linewidth=1.5, linestyle='--',
                   label='Predicted' if i == 0 else '')

    ax.set_xlim(0, n_timepoints)
    ax.set_xlabel('TR')
    ax.set_ylabel('')
    ax.set_title('Boundary Alignment (Human vs Model)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'boundary_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Metrics bar plot
    if 'evaluation' in results:
        eval_results = results['evaluation']
        metrics = ['precision', 'recall', 'f1']
        values = [eval_results.get(m, 0) for m in metrics]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(metrics, values, color=['steelblue', 'darkorange', 'seagreen'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f'Boundary Detection Performance (tolerance=±3 TRs)')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=11)

        # Add reference line (Baldassano's reported ~35-40% match)
        ax.axhline(0.35, color='red', linestyle='--', alpha=0.5, label='Baldassano ~35%')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate HMM implementation using Sherlock data (Baldassano 2017)'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing Sherlock data')
    parser.add_argument('--roi', type=str, default='AG',
                        help='ROI to analyze (default: AG)')
    parser.add_argument('--n-events', type=int, default=25,
                        help='Number of events for Baldassano HMM (default: 25)')
    parser.add_argument('--n-states', type=int, default=25,
                        help='Number of states for Yang HMM (default: 25)')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Number of EM iterations (default: 100)')
    parser.add_argument('--tolerance', type=int, default=3,
                        help='Tolerance for boundary matching in TRs (default: 3)')
    parser.add_argument('--output-dir', type=str, default='results/validation/baldassano',
                        help='Output directory')
    parser.add_argument('--method', type=str, choices=['baldassano', 'yang', 'both'],
                        default='both', help='Which method to run')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Baldassano 2017 Replication - HMM Validation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"ROI: {args.roi}")
    print(f"Output: {output_dir}")

    # Load data
    try:
        data = load_sherlock_data(args.data_dir, args.roi)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo download Sherlock data:")
        print("  wget -O sherlock.zip 'https://figshare.com/ndownloader/files/9021937'")
        print("  unzip sherlock.zip")
        sys.exit(1)

    # Get human boundaries
    human_info = get_sherlock_human_boundaries(TR=1.5)
    human_boundaries = human_info['tr_indices']
    print(f"\nHuman boundaries: {len(human_boundaries)} events")
    print(f"  TRs: {human_boundaries[:5]}... (showing first 5)")

    # Determine n_timepoints
    n_timepoints = data.shape[1] if data.ndim == 3 else data.shape[0]

    all_results = {}

    # Run Baldassano-style HMM
    if args.method in ['baldassano', 'both']:
        results_b = run_hmm_baldassano(data, args.n_events, args.n_iter)

        # Evaluate
        eval_b = evaluate_boundaries(
            results_b['boundaries'],
            human_boundaries,
            tolerance=args.tolerance
        )
        results_b['evaluation'] = eval_b

        print(f"\n--- Baldassano HMM Results ---")
        print(f"Predicted boundaries: {len(results_b['boundaries'])}")
        print(f"Precision: {eval_b['precision']:.3f}")
        print(f"Recall: {eval_b['recall']:.3f}")
        print(f"F1 Score: {eval_b['f1']:.3f}")

        all_results['baldassano'] = results_b

        # Plot
        plot_results(results_b, human_boundaries, n_timepoints,
                    output_dir / 'baldassano')

    # Run Yang-style HMM
    if args.method in ['yang', 'both']:
        results_y = run_hmm_yang(data, args.n_states, args.n_iter)

        # Evaluate
        eval_y = evaluate_boundaries(
            results_y['boundaries'],
            human_boundaries,
            tolerance=args.tolerance
        )
        results_y['evaluation'] = eval_y

        print(f"\n--- Yang HMM Results ---")
        print(f"Predicted boundaries: {len(results_y['boundaries'])}")
        print(f"Precision: {eval_y['precision']:.3f}")
        print(f"Recall: {eval_y['recall']:.3f}")
        print(f"F1 Score: {eval_y['f1']:.3f}")

        all_results['yang'] = results_y

        # Plot
        plot_results(results_y, human_boundaries, n_timepoints,
                    output_dir / 'yang')

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(args.data_dir),
        'roi': args.roi,
        'n_timepoints': n_timepoints,
        'human_boundaries': human_boundaries.tolist(),
        'n_human_boundaries': len(human_boundaries),
        'tolerance': args.tolerance,
        'results': {}
    }

    for method, results in all_results.items():
        summary['results'][method] = {
            'n_predicted_boundaries': len(results['boundaries']),
            'boundaries': results['boundaries'].tolist() if isinstance(results['boundaries'], np.ndarray) else results['boundaries'],
            'evaluation': {k: v for k, v in results['evaluation'].items() if k != 'matches'}
        }

    with open(output_dir / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Validation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
