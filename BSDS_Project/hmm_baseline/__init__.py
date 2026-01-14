"""
HMM Baseline for Event Segmentation

Implements HMM-based event segmentation following:
- Baldassano et al. (2017) Neuron - Event-sequential HMM
- Yang et al. (2023) Nature Communications - Standard GaussianHMM
- BrainIAK eventseg module

This module provides:
- HMMEventSegment: Event-sequential HMM (Baldassano style)
- HMMLearnWrapper: Standard GaussianHMM (Yang style, requires hmmlearn)
- HMMConfig: Configuration dataclass
- Comparison utilities with BSDS results
"""

from .config import HMMConfig
from .model import HMMEventSegment
from .comparison import compare_segmentations, boundary_match_score

# Optional hmmlearn wrapper
try:
    from .hmmlearn_wrapper import HMMLearnWrapper, select_optimal_n_states
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

__all__ = [
    'HMMConfig',
    'HMMEventSegment',
    'compare_segmentations',
    'boundary_match_score',
    'HAS_HMMLEARN'
]

if HAS_HMMLEARN:
    __all__.extend(['HMMLearnWrapper', 'select_optimal_n_states'])
