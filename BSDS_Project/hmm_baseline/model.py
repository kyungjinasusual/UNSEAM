"""
HMM Event Segmentation Model

Main interface for HMM-based event segmentation.
Uses BrainIAK if available, falls back to pure Python implementation.

Based on: Baldassano et al. (2017) Neuron
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import pickle
import json
from pathlib import Path

from .config import HMMConfig
from .event_segmentation import PythonEventSegment

# Try to import BrainIAK
try:
    from brainiak.eventseg.event import EventSegment as BrainIAKEventSegment
    HAS_BRAINIAK = True
except ImportError:
    HAS_BRAINIAK = False


class HMMEventSegment:
    """
    HMM Event Segmentation Model.

    This model segments continuous fMRI data into discrete events,
    following Baldassano et al. (2017) Neuron.

    Key features:
    - Events occur in sequence (no skipping/backtracking)
    - Gaussian emission model with shared variance
    - Extracts event boundaries in TR indices or timestamps

    Attributes:
        config: HMMConfig object
        event_pat_: Learned event patterns (V, K)
        event_var_: Learned variance
        segments_: Posterior probabilities per subject
        boundaries_: Event boundaries (TR indices) per subject
        ll_: Log-likelihood history
    """

    def __init__(self, config: Optional[HMMConfig] = None, **kwargs):
        """
        Initialize HMM Event Segmentation model.

        Args:
            config: HMMConfig object (optional)
            **kwargs: Override config parameters
        """
        if config is None:
            config = HMMConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config

        # Fitted attributes
        self.event_pat_ = None
        self.event_var_ = None
        self.segments_ = None
        self.boundaries_ = None
        self.ll_ = None

        # Data info
        self.n_voxels_ = None
        self.n_subjects_ = None
        self.n_timepoints_ = None

        # Internal model
        self._model = None
        self._use_brainiak = HAS_BRAINIAK

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def fit(self, data_list: List[np.ndarray],
            use_brainiak: Optional[bool] = None) -> 'HMMEventSegment':
        """
        Fit HMM event segmentation model.

        Args:
            data_list: List of (T, V) arrays, one per subject/run.
                       T = timepoints, V = voxels/ROIs
            use_brainiak: Force use of BrainIAK (True) or Python (False).
                          If None, uses BrainIAK if available.

        Returns:
            self
        """
        # Validate input
        if not isinstance(data_list, list):
            data_list = [data_list]

        for i, data in enumerate(data_list):
            if data.ndim != 2:
                raise ValueError(f"data_list[{i}] must be 2D (T, V), got shape {data.shape}")

        # Store data info
        self.n_subjects_ = len(data_list)
        self.n_voxels_ = data_list[0].shape[1]
        self.n_timepoints_ = [d.shape[0] for d in data_list]

        # Decide which backend to use
        if use_brainiak is None:
            use_brainiak = HAS_BRAINIAK
        self._use_brainiak = use_brainiak and HAS_BRAINIAK

        if self._use_brainiak:
            self._fit_brainiak(data_list)
        else:
            self._fit_python(data_list)

        # Extract boundaries
        self.boundaries_ = [self.get_event_boundaries(i) for i in range(self.n_subjects_)]

        return self

    def _fit_brainiak(self, data_list: List[np.ndarray]):
        """Fit using BrainIAK EventSegment."""
        if self.config.verbose:
            print(f"Fitting HMM with BrainIAK (n_events={self.config.n_events})")

        self._model = BrainIAKEventSegment(
            n_events=self.config.n_events,
            n_iter=self.config.n_iter,
            tol=self.config.tol
        )

        self._model.fit(data_list)

        # Store results
        self.event_pat_ = self._model.event_pat_
        self.event_var_ = self._model.event_var_
        self.segments_ = self._model.segments_
        self.ll_ = self._model.ll_

    def _fit_python(self, data_list: List[np.ndarray]):
        """Fit using pure Python implementation."""
        if self.config.verbose:
            print(f"Fitting HMM with Python implementation (n_events={self.config.n_events})")

        self._model = PythonEventSegment(
            n_events=self.config.n_events,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_seed=self.config.random_seed
        )

        self._model.fit(data_list)

        # Store results
        self.event_pat_ = self._model.event_pat_
        self.event_var_ = self._model.event_var_
        self.segments_ = self._model.segments_
        self.ll_ = self._model.ll_

    def get_event_boundaries(self, subject_idx: int = 0) -> np.ndarray:
        """
        Extract event boundaries as TR indices.

        The boundary occurs at the TR where the most likely state changes.

        Args:
            subject_idx: Which subject's segmentation to use

        Returns:
            Array of TR indices where event boundaries occur
        """
        if self.segments_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        gamma = self.segments_[subject_idx]
        state_sequence = np.argmax(gamma, axis=1)
        boundaries = np.where(np.diff(state_sequence) != 0)[0]

        return boundaries

    def get_boundaries_timestamp(self,
                                  subject_idx: int = 0,
                                  TR: Optional[float] = None) -> np.ndarray:
        """
        Extract event boundaries as timestamps (seconds).

        Args:
            subject_idx: Which subject's segmentation to use
            TR: Repetition time in seconds. If None, uses config.TR

        Returns:
            Array of timestamps (seconds) where event boundaries occur
        """
        if TR is None:
            TR = self.config.TR

        boundaries_tr = self.get_event_boundaries(subject_idx)
        return boundaries_tr * TR

    def get_state_sequence(self, subject_idx: int = 0) -> np.ndarray:
        """
        Get hard state (event) assignment for each timepoint.

        Args:
            subject_idx: Which subject's segmentation to use

        Returns:
            Array of state indices (length T)
        """
        if self.segments_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        gamma = self.segments_[subject_idx]
        return np.argmax(gamma, axis=1)

    def get_all_boundaries(self,
                           format: str = 'tr') -> Dict[int, np.ndarray]:
        """
        Get event boundaries for all subjects.

        Args:
            format: 'tr' for TR indices, 'timestamp' for seconds

        Returns:
            Dictionary mapping subject index to boundary array
        """
        result = {}
        for i in range(self.n_subjects_):
            if format == 'timestamp':
                result[i] = self.get_boundaries_timestamp(i)
            else:
                result[i] = self.get_event_boundaries(i)
        return result

    def segment_new_data(self, data: np.ndarray) -> np.ndarray:
        """
        Segment new data using learned event patterns.

        Args:
            data: (T, V) array of new observations

        Returns:
            gamma: (T, K) posterior state probabilities
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if hasattr(self._model, 'segment'):
            return self._model.segment(data)
        else:
            # BrainIAK doesn't have a segment method for new data
            # Use Python implementation's approach
            from .event_segmentation import (
                _compute_log_likelihood, forward_pass, backward_pass, compute_posteriors
            )
            event_means = self.event_pat_.T
            log_lik = _compute_log_likelihood(data, event_means, self.event_var_)
            log_alpha, _ = forward_pass(log_lik)
            log_beta = backward_pass(log_lik)
            return compute_posteriors(log_alpha, log_beta, log_lik)

    def compute_occupancy(self, subject_idx: int = 0) -> np.ndarray:
        """
        Compute fractional occupancy of each event.

        Args:
            subject_idx: Which subject's segmentation to use

        Returns:
            Array of fractional occupancies (length K)
        """
        if self.segments_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        gamma = self.segments_[subject_idx]
        return gamma.sum(axis=0) / gamma.sum()

    def compute_mean_lifetime(self,
                               subject_idx: int = 0,
                               TR: Optional[float] = None) -> np.ndarray:
        """
        Compute mean lifetime (duration) of each event in seconds.

        Args:
            subject_idx: Which subject's segmentation to use
            TR: Repetition time in seconds. If None, uses config.TR

        Returns:
            Array of mean lifetimes in seconds (length K)
        """
        if TR is None:
            TR = self.config.TR

        state_sequence = self.get_state_sequence(subject_idx)
        K = self.config.n_events

        lifetimes = []
        for k in range(K):
            # Find contiguous runs of state k
            in_state = (state_sequence == k).astype(int)
            changes = np.diff(np.concatenate([[0], in_state, [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            if len(starts) > 0:
                durations = ends - starts
                lifetimes.append(np.mean(durations) * TR)
            else:
                lifetimes.append(0.0)

        return np.array(lifetimes)

    def save(self, path: str):
        """
        Save fitted model to file.

        Args:
            path: Path to save file (pickle format)
        """
        save_dict = {
            'config': self.config.to_dict(),
            'event_pat_': self.event_pat_,
            'event_var_': self.event_var_,
            'segments_': self.segments_,
            'boundaries_': self.boundaries_,
            'll_': self.ll_,
            'n_voxels_': self.n_voxels_,
            'n_subjects_': self.n_subjects_,
            'n_timepoints_': self.n_timepoints_,
            'use_brainiak': self._use_brainiak
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        if self.config.verbose:
            print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'HMMEventSegment':
        """
        Load fitted model from file.

        Args:
            path: Path to saved model file

        Returns:
            Loaded HMMEventSegment model
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        config = HMMConfig.from_dict(save_dict['config'])
        model = cls(config)

        model.event_pat_ = save_dict['event_pat_']
        model.event_var_ = save_dict['event_var_']
        model.segments_ = save_dict['segments_']
        model.boundaries_ = save_dict['boundaries_']
        model.ll_ = save_dict['ll_']
        model.n_voxels_ = save_dict['n_voxels_']
        model.n_subjects_ = save_dict['n_subjects_']
        model.n_timepoints_ = save_dict['n_timepoints_']
        model._use_brainiak = save_dict.get('use_brainiak', False)

        return model

    def summary(self) -> str:
        """Generate summary string."""
        if self.segments_ is None:
            return "HMMEventSegment (not fitted)"

        lines = [
            "=" * 50,
            "HMM Event Segmentation Summary",
            "=" * 50,
            f"Backend: {'BrainIAK' if self._use_brainiak else 'Python'}",
            f"Number of events: {self.config.n_events}",
            f"Number of subjects: {self.n_subjects_}",
            f"Number of voxels/ROIs: {self.n_voxels_}",
            f"Variance: {self.event_var_:.4f}",
            f"Final log-likelihood: {self.ll_[-1] if self.ll_ else 'N/A':.2f}",
            "",
            "Boundaries per subject (TR indices):",
        ]

        for i, bounds in enumerate(self.boundaries_):
            lines.append(f"  Subject {i}: {bounds.tolist()}")

        lines.append("")
        lines.append("Mean occupancy:")
        for i in range(min(3, self.n_subjects_)):
            occ = self.compute_occupancy(i)
            lines.append(f"  Subject {i}: {occ.round(3).tolist()}")

        return "\n".join(lines)

    def __repr__(self):
        if self.segments_ is None:
            return f"HMMEventSegment(n_events={self.config.n_events}, fitted=False)"
        return f"HMMEventSegment(n_events={self.config.n_events}, n_subjects={self.n_subjects_})"
