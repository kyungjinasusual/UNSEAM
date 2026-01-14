"""
hmmlearn-based HMM implementation

Wrapper around hmmlearn.GaussianHMM for standard HMM fitting.
This provides a different approach than the event-sequential HMM
from Baldassano (2017).

Key difference:
- Baldassano HMM: Events must occur in sequence (state k → k or k+1)
- Standard HMM: Any state transition allowed

Based on: Yang et al. (2023) Nature Communications
"The default network dominates neural responses to evolving movie stories"
Parameters: n_states=4, covariance_type='full', n_iter=500, n_init=100
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

# Try to import hmmlearn
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    warnings.warn("hmmlearn not installed. Install with: pip install hmmlearn")


class HMMLearnWrapper:
    """
    Wrapper for hmmlearn GaussianHMM.

    This implements standard HMM (any transition allowed) as used in
    Yang et al. (2023) Nature Communications.

    Attributes:
        n_states: Number of hidden states
        covariance_type: Type of covariance ('full', 'diag', 'spherical', 'tied')
        n_iter: Maximum EM iterations
        n_init: Number of random initializations
        model_: Fitted hmmlearn model
        best_score_: Best log-likelihood achieved
    """

    def __init__(self,
                 n_states: int = 4,
                 covariance_type: str = 'full',
                 n_iter: int = 500,
                 n_init: int = 10,
                 tol: float = 1e-4,
                 random_seed: Optional[int] = 42,
                 verbose: bool = True):
        """
        Initialize HMMLearn wrapper.

        Args:
            n_states: Number of hidden states (Yang 2023 used 4)
            covariance_type: Covariance type ('full' for Yang 2023)
            n_iter: Maximum EM iterations (Yang 2023 used 500)
            n_init: Number of random initializations (Yang 2023 used 100)
            tol: Convergence tolerance
            random_seed: Random seed for reproducibility
            verbose: Print progress
        """
        if not HAS_HMMLEARN:
            raise ImportError(
                "hmmlearn required. Install with: pip install hmmlearn"
            )

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_init = n_init
        self.tol = tol
        self.random_seed = random_seed
        self.verbose = verbose

        # Fitted attributes
        self.model_ = None
        self.best_score_ = None
        self.all_scores_ = []

        # Data info
        self.n_features_ = None
        self.n_subjects_ = None

    def fit(self, data_list: List[np.ndarray]) -> 'HMMLearnWrapper':
        """
        Fit HMM using multiple random initializations.

        Args:
            data_list: List of (T, V) arrays, one per subject/run

        Returns:
            self
        """
        if not isinstance(data_list, list):
            data_list = [data_list]

        # Concatenate data
        X = np.vstack(data_list)
        lengths = [d.shape[0] for d in data_list]

        self.n_features_ = X.shape[1]
        self.n_subjects_ = len(data_list)

        if self.verbose:
            print(f"Fitting GaussianHMM with {self.n_states} states")
            print(f"  Data shape: {X.shape}")
            print(f"  Covariance type: {self.covariance_type}")
            print(f"  Running {self.n_init} initializations...")

        best_model = None
        best_score = -np.inf
        self.all_scores_ = []

        for init_idx in range(self.n_init):
            # Set random seed for this initialization
            seed = self.random_seed + init_idx if self.random_seed else None

            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=seed,
                    verbose=False
                )

                model.fit(X, lengths)
                score = model.score(X, lengths)
                self.all_scores_.append(score)

                if score > best_score:
                    best_score = score
                    best_model = model

                if self.verbose and (init_idx + 1) % max(1, self.n_init // 10) == 0:
                    print(f"    Init {init_idx + 1}/{self.n_init}: score = {score:.2f}")

            except Exception as e:
                if self.verbose:
                    print(f"    Init {init_idx + 1} failed: {e}")
                self.all_scores_.append(-np.inf)

        self.model_ = best_model
        self.best_score_ = best_score

        if self.verbose:
            print(f"  Best score: {best_score:.2f}")

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict state sequence for new data.

        Args:
            data: (T, V) array

        Returns:
            State sequence of length T
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model_.predict(data)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Get state probabilities for each timepoint.

        Args:
            data: (T, V) array

        Returns:
            (T, K) array of state probabilities
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model_.predict_proba(data)

    def get_event_boundaries(self, data: np.ndarray) -> np.ndarray:
        """
        Extract event boundaries as TR indices.

        Args:
            data: (T, V) array

        Returns:
            Array of TR indices where state changes occur
        """
        states = self.predict(data)
        boundaries = np.where(np.diff(states) != 0)[0]
        return boundaries

    def get_boundaries_timestamp(self,
                                  data: np.ndarray,
                                  TR: float = 2.0) -> np.ndarray:
        """
        Get boundaries as timestamps.

        Args:
            data: (T, V) array
            TR: Repetition time in seconds

        Returns:
            Array of timestamps (seconds)
        """
        bounds_tr = self.get_event_boundaries(data)
        return bounds_tr * TR

    def compute_occupancy(self, data: np.ndarray) -> np.ndarray:
        """
        Compute fractional occupancy of each state.

        Args:
            data: (T, V) array

        Returns:
            Array of fractional occupancies
        """
        states = self.predict(data)
        occupancy = np.bincount(states, minlength=self.n_states) / len(states)
        return occupancy

    def compute_dwell_times(self,
                            data: np.ndarray,
                            TR: float = 2.0) -> Dict[int, List[float]]:
        """
        Compute dwell times (duration in each state visit).

        Args:
            data: (T, V) array
            TR: Repetition time in seconds

        Returns:
            Dictionary mapping state index to list of dwell times
        """
        states = self.predict(data)
        dwell_times = {k: [] for k in range(self.n_states)}

        current_state = states[0]
        current_duration = 1

        for t in range(1, len(states)):
            if states[t] == current_state:
                current_duration += 1
            else:
                dwell_times[current_state].append(current_duration * TR)
                current_state = states[t]
                current_duration = 1

        # Don't forget last segment
        dwell_times[current_state].append(current_duration * TR)

        return dwell_times

    def compute_transition_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute empirical transition matrix from data.

        Args:
            data: (T, V) array

        Returns:
            (K, K) transition probability matrix
        """
        states = self.predict(data)
        trans_matrix = np.zeros((self.n_states, self.n_states))

        for t in range(len(states) - 1):
            trans_matrix[states[t], states[t+1]] += 1

        # Normalize rows
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans_matrix = trans_matrix / row_sums

        return trans_matrix

    def get_state_means(self) -> np.ndarray:
        """
        Get learned state means.

        Returns:
            (K, V) array of state means
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        return self.model_.means_

    def get_state_covariances(self) -> np.ndarray:
        """
        Get learned state covariances.

        Returns:
            Covariance parameters (shape depends on covariance_type)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        return self.model_.covars_

    def summary(self, data_list: Optional[List[np.ndarray]] = None) -> str:
        """
        Generate summary string.

        Args:
            data_list: Optional data for computing statistics
        """
        if self.model_ is None:
            return "HMMLearnWrapper (not fitted)"

        lines = [
            "=" * 50,
            "hmmlearn GaussianHMM Summary",
            "=" * 50,
            f"Number of states: {self.n_states}",
            f"Covariance type: {self.covariance_type}",
            f"Number of features: {self.n_features_}",
            f"Best log-likelihood: {self.best_score_:.2f}",
            f"Initializations: {self.n_init}",
            "",
        ]

        if data_list:
            lines.append("Per-subject statistics:")
            for i, data in enumerate(data_list[:3]):
                bounds = self.get_event_boundaries(data)
                occ = self.compute_occupancy(data)
                lines.append(f"  Subject {i}:")
                lines.append(f"    Boundaries: {bounds.tolist()}")
                lines.append(f"    Occupancy: {occ.round(3).tolist()}")

        return "\n".join(lines)


def select_optimal_n_states(data_list: List[np.ndarray],
                            state_range: Tuple[int, int] = (2, 8),
                            covariance_type: str = 'full',
                            n_iter: int = 100,
                            n_init: int = 10,
                            criterion: str = 'bic',
                            verbose: bool = True) -> Dict:
    """
    Select optimal number of states using information criteria.

    Args:
        data_list: List of (T, V) arrays
        state_range: (min_states, max_states) to evaluate
        covariance_type: Covariance type for HMM
        n_iter: EM iterations per model
        n_init: Initializations per model
        criterion: 'bic', 'aic', or 'likelihood'
        verbose: Print progress

    Returns:
        Dictionary with results for each n_states
    """
    if not HAS_HMMLEARN:
        raise ImportError("hmmlearn required")

    X = np.vstack(data_list)
    lengths = [d.shape[0] for d in data_list]
    n_samples = X.shape[0]

    results = {}

    for n_states in range(state_range[0], state_range[1] + 1):
        if verbose:
            print(f"Evaluating n_states = {n_states}...")

        best_score = -np.inf
        best_model = None

        for init_idx in range(n_init):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=42 + init_idx
                )
                model.fit(X, lengths)
                score = model.score(X, lengths)

                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                continue

        if best_model is None:
            continue

        # Compute number of free parameters
        n_features = X.shape[1]
        n_params = (n_states - 1)  # Initial state probs
        n_params += n_states * (n_states - 1)  # Transition matrix
        n_params += n_states * n_features  # Means

        if covariance_type == 'full':
            n_params += n_states * n_features * (n_features + 1) // 2
        elif covariance_type == 'diag':
            n_params += n_states * n_features
        elif covariance_type == 'spherical':
            n_params += n_states
        elif covariance_type == 'tied':
            n_params += n_features * (n_features + 1) // 2

        # Information criteria
        bic = -2 * best_score + n_params * np.log(n_samples)
        aic = -2 * best_score + 2 * n_params

        results[n_states] = {
            'log_likelihood': best_score,
            'bic': bic,
            'aic': aic,
            'n_params': n_params,
            'model': best_model
        }

        if verbose:
            print(f"  LL={best_score:.2f}, BIC={bic:.2f}, AIC={aic:.2f}")

    # Find optimal
    if criterion == 'bic':
        optimal = min(results.keys(), key=lambda k: results[k]['bic'])
    elif criterion == 'aic':
        optimal = min(results.keys(), key=lambda k: results[k]['aic'])
    else:
        optimal = max(results.keys(), key=lambda k: results[k]['log_likelihood'])

    results['optimal_n_states'] = optimal
    results['criterion_used'] = criterion

    if verbose:
        print(f"\nOptimal n_states = {optimal} (by {criterion})")

    return results
