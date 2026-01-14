"""
Data loaders for HMM baseline experiments

Includes loaders for:
- Sherlock dataset (Baldassano 2017)
- studyforrest dataset (Yang 2023)
- Emo-Film dataset (local)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


# Human-annotated event boundaries for Sherlock (Baldassano 2017)
# These are TR indices where human observers marked event boundaries
SHERLOCK_HUMAN_BOUNDARIES = np.array([
    26, 35, 56, 72, 86, 108, 131, 143, 157, 173,
    192, 204, 226, 313, 362, 398, 505, 526, 533,
    568, 616, 634, 678, 696, 747, 780, 870, 890
])


def get_sherlock_human_boundaries(TR: float = 1.5) -> Dict[str, np.ndarray]:
    """
    Get human-annotated event boundaries for Sherlock dataset.

    Args:
        TR: Repetition time in seconds (Sherlock uses TR=1.5s)

    Returns:
        Dictionary with 'tr_indices' and 'timestamps' arrays
    """
    return {
        'tr_indices': SHERLOCK_HUMAN_BOUNDARIES,
        'timestamps': SHERLOCK_HUMAN_BOUNDARIES * TR,
        'n_boundaries': len(SHERLOCK_HUMAN_BOUNDARIES),
        'TR': TR
    }


def load_sherlock_figshare(data_dir: str,
                            roi: str = 'angular_gyrus') -> List[np.ndarray]:
    """
    Load Sherlock data from Figshare download.

    The Sherlock movie-watching data can be downloaded from:
    https://figshare.com/articles/dataset/Sherlock_movie-watching_fMRI_atlas_data/5270695

    Args:
        data_dir: Directory containing downloaded Sherlock data
        roi: ROI to load ('angular_gyrus' or 'all')

    Returns:
        List of (T, V) arrays, one per subject
    """
    data_path = Path(data_dir)

    # Look for .npy files
    npy_files = sorted(data_path.glob('*.npy'))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in {data_dir}. "
            f"Download from: https://figshare.com/articles/dataset/Sherlock_movie-watching_fMRI_atlas_data/5270695"
        )

    data_list = []
    for f in npy_files:
        data = np.load(f)
        # Ensure shape is (T, V)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.shape[0] < data.shape[1]:
            # Transpose if needed (V, T) -> (T, V)
            data = data.T
        data_list.append(data)

    return data_list


def load_emofilm_data(data_dir: str,
                       task: str = 'BigBuckBunny',
                       subjects: List[str] = None,
                       roi_file: str = None) -> Tuple[List[np.ndarray], Dict]:
    """
    Load Emo-Film fMRI data (pre-extracted ROIs).

    Args:
        data_dir: Directory containing extracted ROI data
        task: Movie task name
        subjects: List of subject IDs or None for all
        roi_file: Path to specific ROI extraction cache file

    Returns:
        Tuple of (data_list, metadata)
    """
    data_path = Path(data_dir)

    # Look for ROI cache files
    if roi_file:
        roi_path = Path(roi_file)
    else:
        # Try to find in standard locations
        patterns = [
            f'roi_data_{task}*.npy',
            f'{task}*_roi_data.npy',
            f'*{task}*.npy'
        ]

        roi_path = None
        for pattern in patterns:
            matches = list(data_path.glob(pattern))
            if matches:
                roi_path = matches[0]
                break

    if roi_path is None or not roi_path.exists():
        raise FileNotFoundError(
            f"No ROI data found for task {task} in {data_dir}. "
            f"Run ROI extraction first using run_emofilm_bsds.py"
        )

    # Load data
    roi_data = np.load(roi_path, allow_pickle=True)

    if isinstance(roi_data, np.ndarray) and roi_data.dtype == object:
        # Dictionary stored as object array
        roi_dict = roi_data.item()
        data_list = list(roi_dict.values())
    else:
        # Direct array or list
        data_list = [roi_data] if roi_data.ndim == 2 else list(roi_data)

    # Filter subjects if specified
    if subjects:
        # This requires the data to have subject labels
        pass  # TODO: implement subject filtering

    metadata = {
        'task': task,
        'roi_file': str(roi_path),
        'n_subjects': len(data_list),
        'n_timepoints': [d.shape[0] for d in data_list],
        'n_voxels': data_list[0].shape[1] if data_list else 0
    }

    return data_list, metadata


def simulate_event_data(n_subjects: int = 5,
                        n_timepoints: int = 200,
                        n_voxels: int = 50,
                        n_events: int = 8,
                        event_var: float = 1.0,
                        noise_var: float = 0.5,
                        random_seed: int = 42) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Simulate fMRI data with known event structure.

    Useful for testing and validating event segmentation algorithms.

    Args:
        n_subjects: Number of subjects
        n_timepoints: Number of timepoints per subject
        n_voxels: Number of voxels/ROIs
        n_events: Number of events
        event_var: Variance of event patterns
        noise_var: Observation noise variance
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (data_list, true_boundaries)
    """
    np.random.seed(random_seed)

    # Generate event patterns
    event_patterns = np.random.randn(n_events, n_voxels) * np.sqrt(event_var)

    # Generate event boundaries (roughly equal length)
    base_length = n_timepoints // n_events
    boundaries = []
    current = 0
    for i in range(n_events - 1):
        # Add some variability to event lengths
        length = base_length + np.random.randint(-base_length // 4, base_length // 4)
        current += length
        if current < n_timepoints - base_length // 2:
            boundaries.append(current)

    true_boundaries = np.array(boundaries)

    # Generate data for each subject
    data_list = []
    for _ in range(n_subjects):
        data = np.zeros((n_timepoints, n_voxels))

        # Assign timepoints to events
        event_starts = np.concatenate([[0], true_boundaries, [n_timepoints]])
        for k in range(n_events):
            start = event_starts[k]
            end = event_starts[k + 1] if k + 1 < len(event_starts) else n_timepoints
            data[start:end, :] = event_patterns[k]

        # Add noise
        data += np.random.randn(n_timepoints, n_voxels) * np.sqrt(noise_var)

        data_list.append(data)

    return data_list, true_boundaries


class DatasetInfo:
    """Information about available datasets."""

    SHERLOCK = {
        'name': 'Sherlock',
        'description': 'BBC Sherlock episode viewing (Baldassano et al., 2017)',
        'TR': 1.5,
        'n_subjects': 17,
        'duration_minutes': 48,
        'human_boundaries': True,
        'download_url': 'https://figshare.com/articles/dataset/Sherlock_movie-watching_fMRI_atlas_data/5270695',
        'paper': 'Baldassano et al. (2017) Neuron'
    }

    STUDYFORREST = {
        'name': 'studyforrest',
        'description': 'Forrest Gump movie viewing (Yang et al., 2023)',
        'TR': 2.0,
        'n_subjects': 15,
        'duration_minutes': 120,
        'human_boundaries': False,
        'download_url': 'https://github.com/dblabs-mcgill-mila/hmm_forrest_nlp',
        'paper': 'Yang et al. (2023) Nature Communications'
    }

    EMOFILM = {
        'name': 'Emo-FilM',
        'description': 'Emotional film clips viewing',
        'TR': 2.0,
        'n_subjects': 'variable',
        'duration_minutes': 'variable',
        'human_boundaries': False,
        'download_url': None,
        'paper': None
    }

    @classmethod
    def list_datasets(cls) -> Dict[str, Dict]:
        """List all available datasets."""
        return {
            'sherlock': cls.SHERLOCK,
            'studyforrest': cls.STUDYFORREST,
            'emofilm': cls.EMOFILM
        }

    @classmethod
    def get_info(cls, dataset_name: str) -> Dict:
        """Get information about a specific dataset."""
        datasets = cls.list_datasets()
        if dataset_name.lower() not in datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
        return datasets[dataset_name.lower()]
