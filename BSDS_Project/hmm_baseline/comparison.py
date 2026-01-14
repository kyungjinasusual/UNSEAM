"""
Comparison utilities for HMM vs BSDS event segmentation

Provides metrics and visualizations for comparing different
event segmentation approaches.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt


def boundary_match_score(boundaries_pred: np.ndarray,
                         boundaries_true: np.ndarray,
                         tolerance: int = 3) -> Dict[str, float]:
    """
    Compute boundary matching score between predicted and ground truth.

    A predicted boundary is considered a match if it's within 'tolerance'
    TRs of a ground truth boundary.

    Args:
        boundaries_pred: Predicted boundary indices
        boundaries_true: Ground truth boundary indices
        tolerance: Maximum distance (in TRs) for a match

    Returns:
        Dictionary with precision, recall, and F1 score
    """
    if len(boundaries_pred) == 0 and len(boundaries_true) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if len(boundaries_pred) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    if len(boundaries_true) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Find matches
    matched_pred = set()
    matched_true = set()

    for i, bp in enumerate(boundaries_pred):
        for j, bt in enumerate(boundaries_true):
            if j not in matched_true and abs(bp - bt) <= tolerance:
                matched_pred.add(i)
                matched_true.add(j)
                break

    precision = len(matched_pred) / len(boundaries_pred)
    recall = len(matched_true) / len(boundaries_true)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_pred': len(boundaries_pred),
        'n_true': len(boundaries_true),
        'n_matched': len(matched_pred)
    }


def boundary_distance(boundaries1: np.ndarray,
                      boundaries2: np.ndarray) -> Dict[str, float]:
    """
    Compute distance metrics between two sets of boundaries.

    Args:
        boundaries1: First set of boundary indices
        boundaries2: Second set of boundary indices

    Returns:
        Dictionary with distance metrics
    """
    if len(boundaries1) == 0 or len(boundaries2) == 0:
        return {'mean_dist': np.nan, 'median_dist': np.nan, 'max_dist': np.nan}

    # For each boundary in set1, find distance to nearest in set2
    dists_1to2 = []
    for b1 in boundaries1:
        min_dist = np.min(np.abs(boundaries2 - b1))
        dists_1to2.append(min_dist)

    dists_2to1 = []
    for b2 in boundaries2:
        min_dist = np.min(np.abs(boundaries1 - b2))
        dists_2to1.append(min_dist)

    all_dists = dists_1to2 + dists_2to1

    return {
        'mean_dist': np.mean(all_dists),
        'median_dist': np.median(all_dists),
        'max_dist': np.max(all_dists),
        'mean_1to2': np.mean(dists_1to2),
        'mean_2to1': np.mean(dists_2to1)
    }


def compare_segmentations(hmm_model,
                          bsds_model,
                          subject_idx: int = 0,
                          human_boundaries: Optional[np.ndarray] = None,
                          tolerance: int = 3) -> Dict[str, Any]:
    """
    Compare HMM and BSDS event segmentations.

    Args:
        hmm_model: Fitted HMMEventSegment model
        bsds_model: Fitted BSDSModel
        subject_idx: Which subject to compare
        human_boundaries: Optional human-annotated boundaries (TR indices)
        tolerance: Tolerance for boundary matching (TRs)

    Returns:
        Dictionary with comparison metrics
    """
    results = {}

    # Get HMM boundaries
    hmm_bounds = hmm_model.get_event_boundaries(subject_idx)

    # Get BSDS state sequence and derive boundaries
    bsds_states = bsds_model.states_[subject_idx] if bsds_model.states_ is not None else None
    if bsds_states is not None:
        bsds_bounds = np.where(np.diff(bsds_states) != 0)[0]
    else:
        bsds_bounds = np.array([])

    results['hmm_boundaries'] = hmm_bounds
    results['bsds_boundaries'] = bsds_bounds
    results['hmm_n_boundaries'] = len(hmm_bounds)
    results['bsds_n_boundaries'] = len(bsds_bounds)

    # Compare HMM vs BSDS
    results['hmm_vs_bsds'] = boundary_match_score(hmm_bounds, bsds_bounds, tolerance)
    results['hmm_vs_bsds_dist'] = boundary_distance(hmm_bounds, bsds_bounds)

    # Compare to human boundaries if provided
    if human_boundaries is not None:
        results['human_boundaries'] = human_boundaries
        results['hmm_vs_human'] = boundary_match_score(hmm_bounds, human_boundaries, tolerance)
        results['bsds_vs_human'] = boundary_match_score(bsds_bounds, human_boundaries, tolerance)
        results['hmm_vs_human_dist'] = boundary_distance(hmm_bounds, human_boundaries)
        results['bsds_vs_human_dist'] = boundary_distance(bsds_bounds, human_boundaries)

    # Occupancy comparison
    hmm_occ = hmm_model.compute_occupancy(subject_idx)
    results['hmm_occupancy'] = hmm_occ

    if hasattr(bsds_model, 'gamma_list_') and bsds_model.gamma_list_ is not None:
        bsds_gamma = bsds_model.gamma_list_[subject_idx]
        bsds_occ = bsds_gamma.sum(axis=0) / bsds_gamma.sum()
        results['bsds_occupancy'] = bsds_occ

    return results


def plot_boundary_comparison(hmm_model,
                              bsds_model,
                              subject_idx: int = 0,
                              human_boundaries: Optional[np.ndarray] = None,
                              TR: float = 2.0,
                              title: str = "Event Boundary Comparison",
                              figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot event boundaries from different methods.

    Args:
        hmm_model: Fitted HMMEventSegment model
        bsds_model: Fitted BSDSModel
        subject_idx: Which subject to compare
        human_boundaries: Optional human-annotated boundaries (TR indices)
        TR: Repetition time in seconds
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Get data
    n_timepoints = hmm_model.n_timepoints_[subject_idx]
    time_axis = np.arange(n_timepoints) * TR

    # HMM segmentation
    ax1 = axes[0]
    hmm_states = hmm_model.get_state_sequence(subject_idx)
    ax1.plot(time_axis, hmm_states, 'b-', linewidth=0.5)
    ax1.fill_between(time_axis, 0, hmm_states, alpha=0.3)

    hmm_bounds = hmm_model.get_event_boundaries(subject_idx)
    for b in hmm_bounds:
        ax1.axvline(b * TR, color='blue', linestyle='--', alpha=0.7)

    ax1.set_ylabel('HMM State')
    ax1.set_title(f'{title} - HMM Event Segmentation')

    # BSDS segmentation
    ax2 = axes[1]
    if bsds_model.states_ is not None:
        bsds_states = bsds_model.states_[subject_idx]
        ax2.plot(time_axis[:len(bsds_states)], bsds_states, 'g-', linewidth=0.5)
        ax2.fill_between(time_axis[:len(bsds_states)], 0, bsds_states, alpha=0.3, color='green')

        bsds_bounds = np.where(np.diff(bsds_states) != 0)[0]
        for b in bsds_bounds:
            ax2.axvline(b * TR, color='green', linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, 'BSDS states not available',
                 transform=ax2.transAxes, ha='center', va='center')

    ax2.set_ylabel('BSDS State')
    ax2.set_title('BSDS Event Segmentation')

    # Boundary comparison
    ax3 = axes[2]

    y_pos = 0
    labels = []

    # HMM boundaries
    for b in hmm_bounds:
        ax3.plot(b * TR, y_pos, 'b|', markersize=20)
    labels.append(('HMM', 'blue'))
    y_pos += 1

    # BSDS boundaries
    if bsds_model.states_ is not None:
        for b in bsds_bounds:
            ax3.plot(b * TR, y_pos, 'g|', markersize=20)
    labels.append(('BSDS', 'green'))
    y_pos += 1

    # Human boundaries
    if human_boundaries is not None:
        for b in human_boundaries:
            ax3.plot(b * TR, y_pos, 'r|', markersize=20)
        labels.append(('Human', 'red'))
        y_pos += 1

    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels([l[0] for l in labels])
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Event Boundaries Comparison')
    ax3.set_ylim(-0.5, len(labels) - 0.5)

    plt.tight_layout()
    return fig


def plot_occupancy_comparison(hmm_model,
                               bsds_model,
                               subject_idx: int = 0,
                               figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
    """
    Compare state occupancy between HMM and BSDS.

    Args:
        hmm_model: Fitted HMMEventSegment model
        bsds_model: Fitted BSDSModel
        subject_idx: Which subject to compare
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # HMM occupancy
    hmm_occ = hmm_model.compute_occupancy(subject_idx)
    ax1 = axes[0]
    ax1.bar(range(len(hmm_occ)), hmm_occ, color='blue', alpha=0.7)
    ax1.set_xlabel('Event')
    ax1.set_ylabel('Fractional Occupancy')
    ax1.set_title(f'HMM Occupancy ({len(hmm_occ)} events)')

    # BSDS occupancy
    ax2 = axes[1]
    if hasattr(bsds_model, 'gamma_list_') and bsds_model.gamma_list_ is not None:
        bsds_gamma = bsds_model.gamma_list_[subject_idx]
        bsds_occ = bsds_gamma.sum(axis=0) / bsds_gamma.sum()
        ax2.bar(range(len(bsds_occ)), bsds_occ, color='green', alpha=0.7)
        ax2.set_title(f'BSDS Occupancy ({len(bsds_occ)} states)')
    else:
        ax2.text(0.5, 0.5, 'BSDS gamma not available',
                 transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('BSDS Occupancy')

    ax2.set_xlabel('State')
    ax2.set_ylabel('Fractional Occupancy')

    plt.tight_layout()
    return fig


def generate_comparison_report(hmm_model,
                                bsds_model,
                                subject_idx: int = 0,
                                human_boundaries: Optional[np.ndarray] = None,
                                tolerance: int = 3,
                                TR: float = 2.0) -> str:
    """
    Generate a text report comparing HMM and BSDS results.

    Args:
        hmm_model: Fitted HMMEventSegment model
        bsds_model: Fitted BSDSModel
        subject_idx: Which subject to compare
        human_boundaries: Optional human-annotated boundaries (TR indices)
        tolerance: Tolerance for boundary matching (TRs)
        TR: Repetition time in seconds

    Returns:
        Formatted report string
    """
    results = compare_segmentations(
        hmm_model, bsds_model, subject_idx, human_boundaries, tolerance
    )

    lines = [
        "=" * 60,
        "EVENT SEGMENTATION COMPARISON REPORT",
        "=" * 60,
        "",
        f"Subject: {subject_idx}",
        f"Tolerance: {tolerance} TRs ({tolerance * TR:.1f} seconds)",
        "",
        "-" * 60,
        "BOUNDARY COUNTS",
        "-" * 60,
        f"  HMM boundaries:  {results['hmm_n_boundaries']}",
        f"  BSDS boundaries: {results['bsds_n_boundaries']}",
    ]

    if human_boundaries is not None:
        lines.append(f"  Human boundaries: {len(human_boundaries)}")

    lines.extend([
        "",
        "-" * 60,
        "HMM vs BSDS COMPARISON",
        "-" * 60,
        f"  Precision: {results['hmm_vs_bsds']['precision']:.3f}",
        f"  Recall:    {results['hmm_vs_bsds']['recall']:.3f}",
        f"  F1 Score:  {results['hmm_vs_bsds']['f1']:.3f}",
        f"  Mean distance: {results['hmm_vs_bsds_dist']['mean_dist']:.2f} TRs",
    ])

    if human_boundaries is not None:
        lines.extend([
            "",
            "-" * 60,
            "HMM vs HUMAN COMPARISON",
            "-" * 60,
            f"  Precision: {results['hmm_vs_human']['precision']:.3f}",
            f"  Recall:    {results['hmm_vs_human']['recall']:.3f}",
            f"  F1 Score:  {results['hmm_vs_human']['f1']:.3f}",
            f"  Mean distance: {results['hmm_vs_human_dist']['mean_dist']:.2f} TRs",
            "",
            "-" * 60,
            "BSDS vs HUMAN COMPARISON",
            "-" * 60,
            f"  Precision: {results['bsds_vs_human']['precision']:.3f}",
            f"  Recall:    {results['bsds_vs_human']['recall']:.3f}",
            f"  F1 Score:  {results['bsds_vs_human']['f1']:.3f}",
            f"  Mean distance: {results['bsds_vs_human_dist']['mean_dist']:.2f} TRs",
        ])

    lines.extend([
        "",
        "-" * 60,
        "BOUNDARY LOCATIONS (TR indices)",
        "-" * 60,
        f"  HMM:  {results['hmm_boundaries'].tolist()}",
        f"  BSDS: {results['bsds_boundaries'].tolist()}",
    ])

    if human_boundaries is not None:
        lines.append(f"  Human: {human_boundaries.tolist()}")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
