#!/usr/bin/env python3
"""
Comprehensive Comparison: BSDS vs HMM Baselines
==============================================

Compare three event segmentation methods:
1. BSDS (Bayesian Switching Dynamical Systems)
2. HMM-Baldassano (Event-sequential HMM)
3. HMM-Yang (Standard GaussianHMM)

Usage:
    python compare_methods.py --mode simulated
    python compare_methods.py --mode emofilm --task BigBuckBunny
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

# BSDS
from bsds_complete import BSDSModel, BSDSConfig

# HMM Baselines
from hmm_baseline import HMMEventSegment, HMMConfig
from hmm_baseline.hmmlearn_wrapper import HMMLearnWrapper
from hmm_baseline.data_loaders import simulate_event_data
from hmm_baseline.comparison import boundary_match_score


def run_comparison_simulated(n_subjects=5, n_timepoints=150, n_voxels=50,
                             n_events=6, seed=42, output_dir="results/comparison"):
    """Run comparison on simulated data with known ground truth."""

    print("="*70)
    print("Event Segmentation Methods Comparison")
    print("="*70)
    print()

    # Generate simulated data
    print("Generating simulated data...")
    data_list, true_bounds = simulate_event_data(
        n_subjects=n_subjects,
        n_timepoints=n_timepoints,
        n_voxels=n_voxels,
        n_events=n_events,
        random_seed=seed
    )

    # Convert to BSDS format (ROIs x Time per subject)
    bsds_data_list = [d.T for d in data_list]  # (V, T) format

    print(f"  Subjects: {n_subjects}")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  Voxels/ROIs: {n_voxels}")
    print(f"  True events: {n_events}")
    print(f"  True boundaries: {true_bounds.tolist()}")
    print()

    results = {
        'true_boundaries': true_bounds.tolist(),
        'n_subjects': n_subjects,
        'n_timepoints': n_timepoints,
        'n_voxels': n_voxels,
        'n_events': n_events,
        'methods': {}
    }

    # ============================================================
    # Method 1: BSDS
    # ============================================================
    print("-"*70)
    print("Method 1: BSDS (Bayesian Switching Dynamical Systems)")
    print("-"*70)

    bsds_config = BSDSConfig(
        n_states=n_events,
        max_ldim=min(10, n_voxels // 2),
        n_iter=50,
        TR=2.0
    )
    bsds_model = BSDSModel(bsds_config)

    try:
        bsds_model.fit(bsds_data_list)
        bsds_states = bsds_model.get_states()

        # Extract boundaries from BSDS states
        bsds_bounds_list = []
        bsds_scores = []

        for i in range(n_subjects):
            states = bsds_states[i] if i < len(bsds_states) else bsds_states[0]
            bounds = np.where(np.diff(states) != 0)[0]
            bsds_bounds_list.append(bounds)
            score = boundary_match_score(bounds, true_bounds, tolerance=3)
            bsds_scores.append(score)

            if i < 3:
                print(f"  Subject {i}: boundaries={bounds.tolist()}, F1={score['f1']:.3f}")

        avg_f1 = np.mean([s['f1'] for s in bsds_scores])
        print(f"  Average F1: {avg_f1:.3f}")

        results['methods']['BSDS'] = {
            'boundaries': [b.tolist() for b in bsds_bounds_list],
            'scores': bsds_scores,
            'avg_f1': avg_f1
        }

    except Exception as e:
        print(f"  BSDS failed: {e}")
        results['methods']['BSDS'] = {'error': str(e)}

    # ============================================================
    # Method 2: HMM-Baldassano (Event-Sequential)
    # ============================================================
    print()
    print("-"*70)
    print("Method 2: HMM-Baldassano (Event-Sequential HMM)")
    print("-"*70)

    hmm_config = HMMConfig(n_events=n_events, n_iter=100, verbose=False)
    hmm_bald = HMMEventSegment(hmm_config)
    hmm_bald.fit(data_list, use_brainiak=False)

    bald_scores = []
    for i in range(n_subjects):
        bounds = hmm_bald.get_event_boundaries(i)
        score = boundary_match_score(bounds, true_bounds, tolerance=3)
        bald_scores.append(score)

        if i < 3:
            print(f"  Subject {i}: boundaries={bounds.tolist()}, F1={score['f1']:.3f}")

    avg_f1 = np.mean([s['f1'] for s in bald_scores])
    print(f"  Average F1: {avg_f1:.3f}")

    results['methods']['HMM-Baldassano'] = {
        'boundaries': [hmm_bald.get_event_boundaries(i).tolist() for i in range(n_subjects)],
        'scores': bald_scores,
        'avg_f1': avg_f1
    }

    # ============================================================
    # Method 3: HMM-Yang (Standard GaussianHMM)
    # ============================================================
    print()
    print("-"*70)
    print("Method 3: HMM-Yang (Standard GaussianHMM)")
    print("-"*70)

    hmm_yang = HMMLearnWrapper(
        n_states=n_events,
        covariance_type='diag',
        n_iter=100,
        n_init=10,
        random_seed=seed,
        verbose=False
    )
    hmm_yang.fit(data_list)

    yang_scores = []
    yang_bounds_list = []
    for i in range(n_subjects):
        bounds = hmm_yang.get_event_boundaries(data_list[i])
        yang_bounds_list.append(bounds)
        score = boundary_match_score(bounds, true_bounds, tolerance=3)
        yang_scores.append(score)

        if i < 3:
            print(f"  Subject {i}: boundaries={bounds.tolist()}, F1={score['f1']:.3f}")

    avg_f1 = np.mean([s['f1'] for s in yang_scores])
    print(f"  Average F1: {avg_f1:.3f}")

    results['methods']['HMM-Yang'] = {
        'boundaries': [b.tolist() for b in yang_bounds_list],
        'scores': yang_scores,
        'avg_f1': avg_f1
    }

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"{'Method':<25} {'Avg F1':<10} {'Avg Precision':<15} {'Avg Recall':<15}")
    print("-"*70)

    for method_name, method_data in results['methods'].items():
        if 'error' in method_data:
            print(f"{method_name:<25} {'ERROR':<10}")
            continue

        scores = method_data['scores']
        avg_f1 = np.mean([s['f1'] for s in scores])
        avg_prec = np.mean([s['precision'] for s in scores])
        avg_rec = np.mean([s['recall'] for s in scores])
        print(f"{method_name:<25} {avg_f1:<10.3f} {avg_prec:<15.3f} {avg_rec:<15.3f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f"comparison_simulated_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    print()
    print(f"Results saved to: {results_file}")

    # Create visualization
    create_comparison_plot(results, output_path, timestamp)

    return results


def create_comparison_plot(results, output_path, timestamp):
    """Create visualization comparing the methods."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: F1 Score comparison (bar chart)
    ax1 = axes[0, 0]
    methods = []
    f1_scores = []
    for method_name, method_data in results['methods'].items():
        if 'error' not in method_data:
            methods.append(method_name)
            f1_scores.append(method_data['avg_f1'])

    colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(methods)]
    bars = ax1.bar(methods, f1_scores, color=colors, edgecolor='black')
    ax1.set_ylabel('Average F1 Score')
    ax1.set_title('Method Comparison: F1 Score')
    ax1.set_ylim(0, 1.1)
    for bar, score in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', fontsize=12)

    # Plot 2: Precision vs Recall
    ax2 = axes[0, 1]
    for i, (method_name, method_data) in enumerate(results['methods'].items()):
        if 'error' in method_data:
            continue
        scores = method_data['scores']
        precisions = [s['precision'] for s in scores]
        recalls = [s['recall'] for s in scores]
        ax2.scatter(recalls, precisions, label=method_name, s=100, alpha=0.7)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall (per subject)')
    ax2.legend()
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    # Plot 3: Boundary locations for Subject 0
    ax3 = axes[1, 0]
    true_bounds = results['true_boundaries']
    n_timepoints = results['n_timepoints']

    y_pos = 0
    ax3.barh(y_pos, n_timepoints, height=0.3, color='lightgray', label='Timeline')
    for b in true_bounds:
        ax3.axvline(b, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax3.scatter(true_bounds, [y_pos]*len(true_bounds), color='black', s=100,
               zorder=5, label='True boundaries')

    for i, (method_name, method_data) in enumerate(results['methods'].items()):
        if 'error' in method_data:
            continue
        y_pos = i + 1
        bounds = method_data['boundaries'][0]
        ax3.scatter(bounds, [y_pos]*len(bounds), s=80, marker='v',
                   label=f'{method_name}', alpha=0.8)

    ax3.set_yticks(range(len(results['methods']) + 1))
    ax3.set_yticklabels(['Ground Truth'] + list(results['methods'].keys()))
    ax3.set_xlabel('Timepoint (TR)')
    ax3.set_title('Event Boundaries Comparison (Subject 0)')
    ax3.set_xlim(0, n_timepoints)

    # Plot 4: Box plot of F1 scores
    ax4 = axes[1, 1]
    box_data = []
    box_labels = []
    for method_name, method_data in results['methods'].items():
        if 'error' not in method_data:
            box_data.append([s['f1'] for s in method_data['scores']])
            box_labels.append(method_name)

    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Distribution Across Subjects')
    ax4.set_ylim(0, 1.1)

    plt.tight_layout()

    plot_file = output_path / f"comparison_plot_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {plot_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare event segmentation methods')
    parser.add_argument('--mode', type=str, default='simulated',
                       choices=['simulated', 'emofilm'],
                       help='Data mode')
    parser.add_argument('--n-subjects', type=int, default=5)
    parser.add_argument('--n-timepoints', type=int, default=150)
    parser.add_argument('--n-voxels', type=int, default=50)
    parser.add_argument('--n-events', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/comparison')

    args = parser.parse_args()

    if args.mode == 'simulated':
        run_comparison_simulated(
            n_subjects=args.n_subjects,
            n_timepoints=args.n_timepoints,
            n_voxels=args.n_voxels,
            n_events=args.n_events,
            seed=args.seed,
            output_dir=args.output
        )
