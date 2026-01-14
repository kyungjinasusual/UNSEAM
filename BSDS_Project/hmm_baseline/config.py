"""
Configuration class for HMM Event Segmentation

Based on: Baldassano et al. (2017) Neuron
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class HMMConfig:
    """Configuration for HMM Event Segmentation model."""

    # Model structure
    n_events: int = 10  # Number of events (states)

    # Training parameters
    n_iter: int = 100  # Maximum EM iterations
    tol: float = 1e-4  # Convergence tolerance

    # Event model type
    # 'gaussian': Standard Gaussian emissions
    # 'split_merge': Allow split/merge operations (slower but better for uneven durations)
    event_model: str = 'gaussian'

    # Split-merge parameters (if event_model='split_merge')
    split_merge_proposals: int = 3  # Number of proposals per iteration

    # Data parameters
    TR: float = 2.0  # Repetition time in seconds

    # Initialization
    init_method: str = 'uniform'  # 'uniform', 'kmeans', or 'random'
    random_seed: Optional[int] = 42

    # Output
    verbose: bool = True

    # Variance constraints
    fix_variance: bool = False  # If True, fix variance at initial value

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'n_events': self.n_events,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'event_model': self.event_model,
            'split_merge_proposals': self.split_merge_proposals,
            'TR': self.TR,
            'init_method': self.init_method,
            'random_seed': self.random_seed,
            'verbose': self.verbose,
            'fix_variance': self.fix_variance
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'HMMConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'HMMConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def __str__(self):
        return (f"HMMConfig(n_events={self.n_events}, n_iter={self.n_iter}, "
                f"event_model={self.event_model}, TR={self.TR})")
