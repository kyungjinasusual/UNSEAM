#!/usr/bin/env python3
"""
HMM Event Segmentation Baseline Runner
=======================================

Run HMM-based event segmentation on fMRI data.

This implements the approach from:
- Baldassano et al. (2017) Neuron - "Discovering event structure in continuous narrative perception and memory"

Features:
- Pure Python HMM implementation (no BrainIAK required)
- Support for Emo-Film and other datasets
- Comparison with BSDS results
- Validation against human-annotated boundaries (Sherlock)

Usage Examples:
    # Run on simulated data to verify implementation
    python run_hmm_baseline.py --mode test

    # Run on Emo-Film data
    python run_hmm_baseline.py --task BigBuckBunny --n-events 10

    # Run on Sherlock data (if available)
    python run_hmm_baseline.py --dataset sherlock --data-dir ./sherlock_data

    # Compare with BSDS results
    python run_hmm_baseline.py --task BigBuckBunny --compare-bsds results/bsds_model.pkl

Author: UNSEAM Project
Date: 2025-01-14
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from hmm_baseline import HMMConfig, HMMEventSegment
from hmm_baseline.data_loaders import (
    simulate_event_data,
    get_sherlock_human_boundaries,
    load_emofilm_data,
    DatasetInfo,
    SHERLOCK_HUMAN_BOUNDARIES
)
from hmm_baseline.comparison import (
    boundary_match_score,
    generate_comparison_report
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HMM Event Segmentation Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on simulated data
  python run_hmm_baseline.py --mode test

  # Run on Emo-Film data
  python run_hmm_baseline.py --task BigBuckBunny --n-events 10

  # Compare with BSDS
  python run_hmm_baseline.py --task BigBuckBunny --compare-bsds results/bsds_model.pkl
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='run',
                        choices=['run', 'test', 'compare', 'info'],
                        help='Operation mode')

    # Data source
    parser.add_argument('--dataset', type=str, default='emofilm',
                        choices=['sherlock', 'studyforrest', 'emofilm', 'simulated'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing data')
    parser.add_argument('--task', type=str, default='BigBuckBunny',
                        help='Task/movie name (for Emo-Film)')
    parser.add_argument('--roi-file', type=str, default=None,
                        help='Path to pre-extracted ROI data file')

    # Model parameters
    parser.add_argument('--n-events', '-K', type=int, default=10,
                        help='Number of events')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Maximum EM iterations')
    parser.add_argument('--tol', type=float, default=1e-4,
                        help='Convergence tolerance')
    parser.add_argument('--TR', type=float, default=2.0,
                        help='Repetition time in seconds')

    # Simulation parameters
    parser.add_argument('--sim-subjects', type=int, default=5,
                        help='Number of simulated subjects')
    parser.add_argument('--sim-timepoints', type=int, default=200,
                        help='Number of timepoints per subject')
    parser.add_argument('--sim-voxels', type=int, default=50,
                        help='Number of voxels/ROIs')

    # Comparison
    parser.add_argument('--compare-bsds', type=str, default=None,
                        help='Path to BSDS model for comparison')
    parser.add_argument('--human-boundaries', action='store_true',
                        help='Compare with human boundaries (Sherlock only)')
    parser.add_argument('--tolerance', type=int, default=3,
                        help='Boundary matching tolerance (TRs)')

    # Output
    parser.add_argument('--output-dir', '-o', type=str, default='results/hmm_baseline',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Verbose output')

    return parser.parse_args()


def run_test_mode(args):
    """Run on simulated data to verify implementation."""
    print("=" * 60)
    print("HMM Event Segmentation - Test Mode")
    print("=" * 60)
    print()

    # Generate simulated data with known boundaries
    true_n_events = 8
    print(f"Generating simulated data with {true_n_events} events...")
    data_list, true_boundaries = simulate_event_data(
        n_subjects=args.sim_subjects,
        n_timepoints=args.sim_timepoints,
        n_voxels=args.sim_voxels,
        n_events=true_n_events,
        random_seed=args.seed
    )

    print(f"  Subjects: {len(data_list)}")
    print(f"  Timepoints: {data_list[0].shape[0]}")
    print(f"  Voxels: {data_list[0].shape[1]}")
    print(f"  True boundaries: {true_boundaries.tolist()}")
    print()

    # Fit HMM with correct number of events
    print(f"Fitting HMM with {true_n_events} events...")
    config = HMMConfig(
        n_events=true_n_events,
        n_iter=args.n_iter,
        tol=args.tol,
        TR=args.TR,
        random_seed=args.seed,
        verbose=args.verbose
    )
    hmm = HMMEventSegment(config)
    hmm.fit(data_list, use_brainiak=False)  # Use Python implementation

    # Evaluate
    print()
    print("-" * 60)
    print("Results")
    print("-" * 60)

    for i in range(min(3, len(data_list))):
        pred_boundaries = hmm.get_event_boundaries(i)
        scores = boundary_match_score(pred_boundaries, true_boundaries, tolerance=args.tolerance)

        print(f"\nSubject {i}:")
        print(f"  Predicted boundaries: {pred_boundaries.tolist()}")
        print(f"  True boundaries:      {true_boundaries.tolist()}")
        print(f"  Precision: {scores['precision']:.3f}")
        print(f"  Recall:    {scores['recall']:.3f}")
        print(f"  F1 Score:  {scores['f1']:.3f}")

    print()
    print(hmm.summary())

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"hmm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    hmm.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    return hmm


def run_on_data(args):
    """Run HMM on real data."""
    print("=" * 60)
    print("HMM Event Segmentation")
    print("=" * 60)
    print()

    # Load data
    if args.dataset == 'simulated':
        print("Using simulated data...")
        data_list, true_boundaries = simulate_event_data(
            n_subjects=args.sim_subjects,
            n_timepoints=args.sim_timepoints,
            n_voxels=args.sim_voxels,
            n_events=args.n_events,
            random_seed=args.seed
        )
        human_bounds = true_boundaries if args.human_boundaries else None

    elif args.dataset == 'sherlock':
        print("Loading Sherlock data...")
        if args.data_dir is None:
            print("ERROR: --data-dir required for Sherlock dataset")
            print("Download from: https://figshare.com/articles/dataset/Sherlock_movie-watching_fMRI_atlas_data/5270695")
            sys.exit(1)

        from hmm_baseline.data_loaders import load_sherlock_figshare
        data_list = load_sherlock_figshare(args.data_dir)
        args.TR = 1.5  # Sherlock uses TR=1.5s
        human_bounds = SHERLOCK_HUMAN_BOUNDARIES if args.human_boundaries else None

    elif args.dataset == 'emofilm':
        print(f"Loading Emo-Film data (task: {args.task})...")
        if args.roi_file:
            data_list, metadata = load_emofilm_data(
                data_dir=args.data_dir or '.',
                task=args.task,
                roi_file=args.roi_file
            )
        else:
            # Try to find cached ROI data
            default_dirs = [
                'results',
                'code/results',
                '.'
            ]
            data_list = None
            for d in default_dirs:
                try:
                    data_list, metadata = load_emofilm_data(
                        data_dir=d,
                        task=args.task
                    )
                    break
                except FileNotFoundError:
                    continue

            if data_list is None:
                print(f"ERROR: No ROI data found for task {args.task}")
                print("Run ROI extraction first: python run_emofilm_bsds.py --mode extract --task {args.task}")
                sys.exit(1)

        human_bounds = None

    else:
        print(f"ERROR: Dataset '{args.dataset}' not yet fully supported")
        sys.exit(1)

    print(f"  Subjects: {len(data_list)}")
    print(f"  Timepoints: {[d.shape[0] for d in data_list]}")
    print(f"  Voxels/ROIs: {data_list[0].shape[1]}")
    print()

    # Fit HMM
    print(f"Fitting HMM with {args.n_events} events...")
    config = HMMConfig(
        n_events=args.n_events,
        n_iter=args.n_iter,
        tol=args.tol,
        TR=args.TR,
        random_seed=args.seed,
        verbose=args.verbose
    )
    hmm = HMMEventSegment(config)
    hmm.fit(data_list, use_brainiak=False)

    # Print results
    print()
    print(hmm.summary())

    # Compare with human boundaries if available
    if human_bounds is not None:
        print()
        print("-" * 60)
        print("Comparison with Human Boundaries")
        print("-" * 60)
        for i in range(min(3, len(data_list))):
            pred_boundaries = hmm.get_event_boundaries(i)
            scores = boundary_match_score(pred_boundaries, human_bounds, tolerance=args.tolerance)
            print(f"\nSubject {i}:")
            print(f"  F1 Score: {scores['f1']:.3f} (P={scores['precision']:.3f}, R={scores['recall']:.3f})")

    # Compare with BSDS if provided
    if args.compare_bsds:
        print()
        print("-" * 60)
        print("Comparison with BSDS")
        print("-" * 60)
        try:
            import pickle
            with open(args.compare_bsds, 'rb') as f:
                bsds_model = pickle.load(f)

            report = generate_comparison_report(
                hmm, bsds_model,
                subject_idx=0,
                human_boundaries=human_bounds,
                tolerance=args.tolerance,
                TR=args.TR
            )
            print(report)
        except Exception as e:
            print(f"Error loading BSDS model: {e}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"hmm_{args.dataset}_{args.n_events}events_{timestamp}"

    model_path = output_dir / f"{prefix}_model.pkl"
    hmm.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save boundaries as JSON
    boundaries_path = output_dir / f"{prefix}_boundaries.json"
    boundaries_data = {
        'n_events': args.n_events,
        'TR': args.TR,
        'dataset': args.dataset,
        'boundaries_tr': {i: hmm.get_event_boundaries(i).tolist() for i in range(len(data_list))},
        'boundaries_sec': {i: hmm.get_boundaries_timestamp(i).tolist() for i in range(len(data_list))},
    }
    with open(boundaries_path, 'w') as f:
        json.dump(boundaries_data, f, indent=2)
    print(f"Boundaries saved to: {boundaries_path}")

    return hmm


def show_dataset_info(args):
    """Show information about available datasets."""
    print("=" * 60)
    print("Available Datasets")
    print("=" * 60)
    print()

    for name, info in DatasetInfo.list_datasets().items():
        print(f"{info['name']}")
        print("-" * 40)
        print(f"  Description: {info['description']}")
        print(f"  TR: {info['TR']}s")
        print(f"  Subjects: {info['n_subjects']}")
        print(f"  Duration: {info['duration_minutes']} min")
        print(f"  Human boundaries: {'Yes' if info['human_boundaries'] else 'No'}")
        if info['download_url']:
            print(f"  Download: {info['download_url']}")
        if info['paper']:
            print(f"  Paper: {info['paper']}")
        print()

    print()
    print("Sherlock Human Boundaries (TR indices):")
    print(f"  {SHERLOCK_HUMAN_BOUNDARIES.tolist()}")
    print(f"  Total: {len(SHERLOCK_HUMAN_BOUNDARIES)} boundaries")


def main():
    args = parse_args()

    if args.mode == 'test':
        run_test_mode(args)
    elif args.mode == 'info':
        show_dataset_info(args)
    elif args.mode == 'run':
        run_on_data(args)
    elif args.mode == 'compare':
        if args.compare_bsds is None:
            print("ERROR: --compare-bsds required for compare mode")
            sys.exit(1)
        run_on_data(args)


if __name__ == '__main__':
    main()
