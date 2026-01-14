"""
Pure Python HMM Event Segmentation

Implementation based on:
- Baldassano et al. (2017) Neuron
- BrainIAK eventseg module (https://github.com/brainiak/brainiak)

This provides a standalone implementation that doesn't require BrainIAK installation.
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def _log_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Log-normalize along an axis."""
    return x - logsumexp(x, axis=axis, keepdims=True)


def _compute_log_likelihood(data: np.ndarray,
                            event_means: np.ndarray,
                            event_var: float) -> np.ndarray:
    """
    Compute log-likelihood of data under Gaussian event model.

    Args:
        data: (T, V) array of observations
        event_means: (K, V) array of event mean patterns
        event_var: Scalar variance (isotropic)

    Returns:
        (T, K) array of log-likelihoods
    """
    T, V = data.shape
    K = event_means.shape[0]

    # Squared distances
    # data: (T, V), event_means: (K, V)
    # diff: (T, K, V)
    diff = data[:, np.newaxis, :] - event_means[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # (T, K)

    # Log-likelihood under isotropic Gaussian
    log_lik = -0.5 * V * np.log(2 * np.pi * event_var) - 0.5 * sq_dist / event_var

    return log_lik


def forward_pass(log_lik: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Forward pass for HMM with event structure.

    Unlike standard HMM, events must occur in order (state k can only
    transition to state k or k+1).

    Args:
        log_lik: (T, K) log-likelihood of observations

    Returns:
        log_alpha: (T, K) forward probabilities
        log_prob: Total log-probability
    """
    T, K = log_lik.shape
    log_alpha = np.full((T, K), -np.inf)

    # Initialize: must start in state 0
    log_alpha[0, 0] = log_lik[0, 0]

    # Forward recursion
    for t in range(1, T):
        for k in range(K):
            if k == 0:
                # Can only stay in state 0
                log_alpha[t, k] = log_alpha[t-1, k] + log_lik[t, k]
            else:
                # Can come from k-1 or k
                log_alpha[t, k] = logsumexp([log_alpha[t-1, k-1], log_alpha[t-1, k]]) + log_lik[t, k]

    # Total probability: must end in state K-1
    log_prob = log_alpha[-1, -1]

    return log_alpha, log_prob


def backward_pass(log_lik: np.ndarray) -> np.ndarray:
    """
    Backward pass for HMM with event structure.

    Args:
        log_lik: (T, K) log-likelihood of observations

    Returns:
        log_beta: (T, K) backward probabilities
    """
    T, K = log_lik.shape
    log_beta = np.full((T, K), -np.inf)

    # Initialize: must end in state K-1
    log_beta[-1, -1] = 0.0

    # Backward recursion
    for t in range(T - 2, -1, -1):
        for k in range(K):
            if k == K - 1:
                # Can only stay in state K-1
                log_beta[t, k] = log_beta[t+1, k] + log_lik[t+1, k]
            else:
                # Can go to k or k+1
                log_beta[t, k] = logsumexp([
                    log_beta[t+1, k] + log_lik[t+1, k],
                    log_beta[t+1, k+1] + log_lik[t+1, k+1]
                ])

    return log_beta


def compute_posteriors(log_alpha: np.ndarray,
                       log_beta: np.ndarray,
                       log_lik: np.ndarray) -> np.ndarray:
    """
    Compute posterior state probabilities (gamma).

    Args:
        log_alpha: (T, K) forward probabilities
        log_beta: (T, K) backward probabilities
        log_lik: (T, K) log-likelihoods

    Returns:
        gamma: (T, K) posterior state probabilities (soft segmentation)
    """
    log_gamma = log_alpha + log_beta
    gamma = np.exp(_log_normalize(log_gamma, axis=1))
    return gamma


def update_event_means(data: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Update event mean patterns (M-step).

    Args:
        data: (T, V) observations
        gamma: (T, K) posterior state probabilities

    Returns:
        event_means: (K, V) updated event means
    """
    # Weighted average
    # gamma: (T, K), data: (T, V)
    # event_means = (K, V)
    weights = gamma.sum(axis=0, keepdims=True).T  # (K, 1)
    weights = np.maximum(weights, 1e-10)  # Avoid division by zero
    event_means = (gamma.T @ data) / weights
    return event_means


def update_variance(data: np.ndarray,
                    event_means: np.ndarray,
                    gamma: np.ndarray) -> float:
    """
    Update event variance (M-step).

    Args:
        data: (T, V) observations
        event_means: (K, V) event mean patterns
        gamma: (T, K) posterior state probabilities

    Returns:
        event_var: Updated variance
    """
    T, V = data.shape
    K = event_means.shape[0]

    # Compute weighted squared residuals
    total_var = 0.0
    total_weight = 0.0

    for k in range(K):
        residuals = data - event_means[k]
        sq_residuals = np.sum(residuals ** 2, axis=1)
        total_var += np.sum(gamma[:, k] * sq_residuals)
        total_weight += np.sum(gamma[:, k]) * V

    event_var = total_var / total_weight
    return max(event_var, 1e-10)


class PythonEventSegment:
    """
    Pure Python HMM Event Segmentation.

    This implements the event segmentation model from Baldassano et al. (2017),
    where events must occur in sequence (no skipping or going backwards).

    Attributes:
        n_events: Number of events
        n_iter: Maximum EM iterations
        tol: Convergence tolerance
        event_pat_: Learned event patterns (V, K)
        event_var_: Learned variance
        segments_: Posterior state probabilities (T, K) for each dataset
        ll_: Log-likelihood history
    """

    def __init__(self,
                 n_events: int = 10,
                 n_iter: int = 100,
                 tol: float = 1e-4,
                 random_seed: Optional[int] = 42):
        self.n_events = n_events
        self.n_iter = n_iter
        self.tol = tol
        self.random_seed = random_seed

        # Fitted attributes
        self.event_pat_ = None  # (V, K)
        self.event_var_ = None
        self.segments_ = None  # List of (T, K) arrays
        self.ll_ = None

    def fit(self, data_list: List[np.ndarray]) -> 'PythonEventSegment':
        """
        Fit the event segmentation model.

        Args:
            data_list: List of (T, V) arrays, one per subject/run

        Returns:
            self
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Concatenate data for initialization
        all_data = np.vstack(data_list)  # (total_T, V)
        V = all_data.shape[1]
        K = self.n_events

        # Initialize event patterns using K-means
        kmeans = KMeans(n_clusters=K, random_state=self.random_seed, n_init=10)
        kmeans.fit(all_data)
        event_means = kmeans.cluster_centers_  # (K, V)

        # Initialize variance
        event_var = np.var(all_data)

        # EM iterations
        self.ll_ = []
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            total_ll = 0.0
            all_gamma = []

            # E-step for each dataset
            for data in data_list:
                log_lik = _compute_log_likelihood(data, event_means, event_var)
                log_alpha, log_prob = forward_pass(log_lik)
                log_beta = backward_pass(log_lik)
                gamma = compute_posteriors(log_alpha, log_beta, log_lik)

                all_gamma.append(gamma)
                total_ll += log_prob

            self.ll_.append(total_ll)

            # Check convergence
            if abs(total_ll - prev_ll) < self.tol:
                break
            prev_ll = total_ll

            # M-step: aggregate across datasets
            concat_data = np.vstack(data_list)
            concat_gamma = np.vstack(all_gamma)

            event_means = update_event_means(concat_data, concat_gamma)
            event_var = update_variance(concat_data, event_means, concat_gamma)

        # Store results
        self.event_pat_ = event_means.T  # (V, K) to match BrainIAK convention
        self.event_var_ = event_var
        self.segments_ = all_gamma

        return self

    def get_event_boundaries(self,
                             subject_idx: int = 0,
                             threshold: float = 0.5) -> np.ndarray:
        """
        Extract event boundaries as TR indices.

        Args:
            subject_idx: Which subject's segmentation to use
            threshold: Not used for argmax method, kept for API compatibility

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
                                  TR: float = 2.0) -> np.ndarray:
        """
        Extract event boundaries as timestamps (seconds).

        Args:
            subject_idx: Which subject's segmentation to use
            TR: Repetition time in seconds

        Returns:
            Array of timestamps where event boundaries occur
        """
        boundaries_tr = self.get_event_boundaries(subject_idx)
        return boundaries_tr * TR

    def get_state_sequence(self, subject_idx: int = 0) -> np.ndarray:
        """
        Get hard state assignment for each timepoint.

        Args:
            subject_idx: Which subject's segmentation to use

        Returns:
            Array of state indices (length T)
        """
        if self.segments_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        gamma = self.segments_[subject_idx]
        return np.argmax(gamma, axis=1)

    def segment(self, data: np.ndarray) -> np.ndarray:
        """
        Segment new data using learned event patterns.

        Args:
            data: (T, V) array of new observations

        Returns:
            gamma: (T, K) posterior state probabilities
        """
        if self.event_pat_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        event_means = self.event_pat_.T  # (K, V)
        log_lik = _compute_log_likelihood(data, event_means, self.event_var_)
        log_alpha, _ = forward_pass(log_lik)
        log_beta = backward_pass(log_lik)
        gamma = compute_posteriors(log_alpha, log_beta, log_lik)

        return gamma
