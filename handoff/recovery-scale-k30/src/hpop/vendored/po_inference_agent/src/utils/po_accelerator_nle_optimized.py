import numpy as np
import pandas as pd
import os, sys, random, math, time, json, yaml, logging
import heapq
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm
from tqdm import tqdm
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
# Project‑local imports
from src.utils.po_fun import BasicUtils, StatisticalUtils
from src.utils.po_accelerator_eta_accelerated import (
    get_ultra_fast_eta_matrix
)

import threading
from concurrent.futures import ThreadPoolExecutor
from src.utils.linext_direct import get_linext_direct  
from src.utils.po_fun import BasicUtils  # Fallback when C++ not available
###############################################################################
#                               LogLikelihoodCache
###############################################################################
class LogLikelihoodCache:
    """ Thread‑safe, cached NLE computations for *flat* (single‑layer) models. """

    # --- shared state -----------------------------------------------------------------
    _nle_cache: Dict[bytes, int] = {}
    _nle_first_cache: Dict[Tuple[bytes, int], int] = {}
    _cache_lock = threading.Lock()
    _pool = ThreadPoolExecutor(max_workers=16)

    # --- helpers ----------------------------------------------------------------------
    @staticmethod
    def _mat_key(mat: np.ndarray) -> bytes:
        return mat.tobytes()

    # --- public API -------------------------------------------------------------------
    @classmethod
    def nle(cls, mat: np.ndarray) -> int:
        """Cached BasicUtils.nle (executed in the thread‑pool)."""
        key = cls._mat_key(mat)
        with cls._cache_lock:
            if key in cls._nle_cache:
                return cls._nle_cache[key]
        val = cls._pool.submit(BasicUtils.nle, mat).result()
        with cls._cache_lock:
            cls._nle_cache[key] = val
        return val

    @classmethod
    def nle_first(cls, mat: np.ndarray, idx: int) -> int:
        key = (cls._mat_key(mat), idx)
        with cls._cache_lock:
            if key in cls._nle_first_cache:
                return cls._nle_first_cache[key]
        val = cls._pool.submit(BasicUtils.num_extensions_with_first, mat, idx).result()
        with cls._cache_lock:
            cls._nle_first_cache[key] = val
        return val

    # --- small utility ----------------------------------------------------------------
    @staticmethod
    def bar_eta_vectorised(indices: List[int], U0: np.ndarray) -> np.ndarray:
        """Mean of latent vector for a *set* of global indices (fully vectorised)."""
        if not indices:
            return np.empty(0)
        subset = U0[np.array(indices)]  # (|set|, K)
        return subset.mean(axis=1)      # (|set|,)

###############################################################################
#                           HPO_LogLikelihoodCache
###############################################################################
class HPO_LogLikelihoodCache_Optimized:
    """ Like `LogLikelihoodCache`, but for hierarchical‑partial‑order (HPO) models. """

    _nle_cache: Dict[bytes, int] = {}
    _nle_first_cache: Dict[Tuple[bytes, int], int] = {}
    _cache_lock = threading.Lock()
    _pool = ThreadPoolExecutor(max_workers=4)
    # Removed: _linext (subprocess overhead was too high)
    SOFTMAX_FRONTIER_EPS = 0.001

    @classmethod
    def nle(cls, mat: np.ndarray) -> int:
        """⭐ FAST: LinextDirect C++ acceleration. Returns -2 on overflow."""
        key = cls._mat_key(mat)
        
        # Check cache without lock
        if key in cls._nle_cache:
            return cls._nle_cache[key]
        
        # Use C++ library directly
        linext = get_linext_direct(quiet=True)
        val = linext.nle(mat)
        
        # Store result
        with cls._cache_lock:
            cls._nle_cache[key] = val
        
        return val

    @classmethod
    def log_nle(cls, mat: np.ndarray) -> float:
        """LOG-SPACE NLE computation - never overflows."""
        key = ("log", cls._mat_key(mat))
        
        if key in cls._nle_cache:
            return cls._nle_cache[key]
        
        linext = get_linext_direct(quiet=True)
        val = linext.log_nle(mat)
        
        with cls._cache_lock:
            cls._nle_cache[key] = val
        
        return val

    @classmethod
    def nle_first(cls, mat: np.ndarray, idx: int) -> int:
        """num_extensions_with_first using C++. Returns -2 on overflow."""
        key = (cls._mat_key(mat), idx)
        
        if key in cls._nle_first_cache:
            return cls._nle_first_cache[key]
        
        tops = BasicUtils.find_tops(mat)
        if idx not in tops:
            val = 0
        else:
            mat_reduced = np.delete(np.delete(mat, idx, axis=0), idx, axis=1)
            linext = get_linext_direct(quiet=True)
            val = linext.nle(mat_reduced)
        
        cls._nle_first_cache[key] = val
        return val

    @classmethod
    def log_nle_first(cls, mat: np.ndarray, idx: int) -> float:
        """LOG-SPACE num_extensions_with_first - never overflows."""
        key = ("log", cls._mat_key(mat), idx)
        
        if key in cls._nle_first_cache:
            return cls._nle_first_cache[key]
        
        tops = BasicUtils.find_tops(mat)
        if idx not in tops:
            val = -math.inf  # log(0) = -inf
        else:
            mat_reduced = np.delete(np.delete(mat, idx, axis=0), idx, axis=1)
            linext = get_linext_direct(quiet=True)
            val = linext.log_nle(mat_reduced)
        
        cls._nle_first_cache[key] = val
        return val

    @classmethod
    def nle_linext(cls, mat: np.ndarray) -> int:
        return cls.nle(mat)  # Redirect to the pure Python nle method

    # -------------------------------------------------------------------------
    @staticmethod
    def _mat_key(mat: np.ndarray) -> bytes:
        return mat.tobytes()

    # -------------------------------------------------------------------------
    @staticmethod
    def bar_eta_vectorised(indices: List[int], U0: np.ndarray) -> np.ndarray:
        """Vectorised bar‑eta; identical to flat case but kept separate for clarity."""
        if not indices:
            return np.empty(0)
        return U0[np.array(indices)].mean(axis=1)


    # -------------------------------------------------------------------------
    @classmethod
    def _adjacency_from_U0(cls, U0: np.ndarray, items: List[int]) -> np.ndarray:
        """Fast adjacency construction for a set of *global* items using only U0."""
        if not items:
            return np.empty((0, 0), dtype=int)
        subset = U0[np.array(items)]
        return BasicUtils.generate_partial_order(subset)

    # -------------------------------------------------------------------------
    @classmethod
    def calculate_log_likelihood_po_optimized(
        cls,
        U: np.ndarray,
        h: np.ndarray,
        observed_orders: List[List[int]],
        choice_sets: List[List[int]],
        items: List[int],
        item_to_index: Dict[int, int],
        prob_noise: float,
        softmax_params: Optional[Dict[str, float]],
        noise_option: str = "queue_jump",
    ) -> float:
        """
        Log-likelihood calculation for a SINGLE partial order.
        
        Args:
            U: Latent utility matrix (n_items x K)
            h: Partial order adjacency matrix (n_items x n_items)
            observed_orders: List of observed total orders
            choice_sets: List of choice sets corresponding to each observed order
            items: List of item IDs
            item_to_index: Mapping from item ID to matrix index
            prob_noise: Noise probability (epsilon for softmax)
            softmax_params: Optional softmax parameters {"beta": ..., "epsilon": ...}
            noise_option: "queue_jump", "weighted_queue_jump", or "log_successors_queue_jump"
            
        Returns:
            Log-likelihood value
        """
        if noise_option not in {"queue_jump", "log_successors_queue_jump", "frontier_softmax_uni_utility"}:
            raise ValueError(f"Unsupported noise_option: {noise_option}")

        total_ll = 0.0

        # Create local index mapping for items
        g2l = {g: i for i, g in enumerate(items)}

        for task_idx, order in enumerate(observed_orders):
            choice_set = choice_sets[task_idx]
            # Filter choice_set to only items in g2l and create local mapping
            choice_set_filtered = [g for g in choice_set if g in g2l]
            g_to_local_idx = {g: i for i, g in enumerate(choice_set_filtered)}
            local_ids = [g2l[g] for g in choice_set_filtered]
            if not local_ids:
                continue

            # Sub-matrix extraction (indices into U/h)
            H_sub = h[np.ix_(local_ids, local_ids)]
            # obs_local must be indices into H_sub (same as choice_set_filtered order)
            obs_local = [g_to_local_idx[g] for g in order if g in g_to_local_idx]
            if not obs_local:
                continue

            # Compute likelihood
            if noise_option == "queue_jump":
                total_ll += cls._queue_jump_ll(obs_local, H_sub, prob_noise)
            elif noise_option == "log_successors_queue_jump":
                if isinstance(softmax_params, dict):
                    beta_val = softmax_params.get("beta", 1.0)
                    epsilon_val = softmax_params.get("epsilon", 0.01)
                else:
                    beta_val = 1.0 / max(prob_noise, 0.01)
                    epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
                total_ll += cls._log_successors_queue_jump_ll(
                    obs_local, H_sub, beta=beta_val, epsilon=epsilon_val
                )
            elif noise_option == "frontier_softmax_uni_utility":
                if isinstance(softmax_params, dict):
                    epsilon_val = softmax_params.get("epsilon", 0.01)
                else:
                    epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
                total_ll += cls._frontier_softmax_uni_utility_ll(
                    obs_local, H_sub, beta=1.0, epsilon=epsilon_val
                )
        
        return total_ll

    @classmethod
    def calculate_log_likelihood_hpo_with_pointwise(
        cls,
        U: Dict[str, Any],                     # {"U0": …, "U_a_dict": …}
        h_U: Dict[int, np.ndarray],            # assessor → local partial order
        observed_orders: Dict[int, List[List[int]]],
        M_a_dict: Dict[int, List[int]],        # assessor → list of global items
        O_a_i_dict: Dict[int, List[List[int]]],# assessor → list of choice‑sets
        item_to_index: Dict[int, int],
        prob_noise: float,
        softmax_params: Optional[Dict[str, float]],
        noise_option: str = "queue_jump"
    ) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """
        Calculate log-likelihood with pointwise contributions for WAIC.
        
        Returns:
            total_ll: Total log-likelihood (sum of all pointwise contributions)
            pointwise_ll: Dict mapping (assessor_id, order_idx) -> log-likelihood contribution
        """
        from typing import Dict, Tuple
        
        if noise_option not in {"queue_jump", "weighted_queue_jump", "log_successors_queue_jump", "frontier_softmax_uni_utility"}:
            raise ValueError(f"Unknown noise_option: {noise_option}")

        total_ll = 0.0
        pointwise_ll = {}
        MISSING_DATA_PENALTY = -1e10
        
        # If h_U is empty, build it using the accelerator
        if not h_U:
            from .po_fun import build_hierarchical_partial_orders_optimized
            h_U = build_hierarchical_partial_orders_optimized(
                M0=list(item_to_index.keys()),
                assessors=list(observed_orders.keys()),
                M_a_dict=M_a_dict,
                U0=U["U0"],
                U_a_dict=U["U_a_dict"],
            )
        
        for assessor, orders_a in observed_orders.items():
            if assessor not in h_U:
                continue
                
            H_a = h_U[assessor]
            g2l = {g: i for i, g in enumerate(M_a_dict[assessor])}
            bar_eta_full = None
            if noise_option == "weighted_queue_jump":
                Ua = U["U_a_dict"][assessor]
                eta_full = get_ultra_fast_eta_matrix(
                    assessor=assessor,
                    Ua=Ua,
                    Ma=M_a_dict[assessor],
                )
                bar_eta_full = eta_full.mean(axis=1)

            for order_idx, order in enumerate(orders_a):
                order_ll = 0.0  # Likelihood for this specific order
                
                choice_set = O_a_i_dict[assessor][order_idx]
                choice_set_filtered = [g for g in choice_set if g in g2l]
                g_to_local_idx = {g: i for i, g in enumerate(choice_set_filtered)}
                local_ids = [g2l[g] for g in choice_set_filtered]
                if not local_ids:
                    pointwise_ll[(assessor, order_idx)] = MISSING_DATA_PENALTY
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Sub‑matrix for this choice‑set
                idx_array = np.array(local_ids)
                H_sub = H_a[idx_array[:, None], idx_array]

                # Compute bar-eta for this choice set
                bar_eta = bar_eta_full[np.array(local_ids)] if bar_eta_full is not None else None

                # Observed order as indices within choice_set
                obs_local = [g_to_local_idx[g] for g in order if g in g_to_local_idx]
                if not obs_local:
                    pointwise_ll[(assessor, order_idx)] = MISSING_DATA_PENALTY
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Likelihood contribution for this order
                if noise_option == "queue_jump":
                    order_ll = cls._queue_jump_ll(obs_local, H_sub, prob_noise)
                elif noise_option == "weighted_queue_jump":
                    order_ll = cls._wqueue_jump_ll(obs_local, H_sub, prob_noise, bar_eta)
                elif noise_option == "log_successors_queue_jump":
                    if isinstance(softmax_params, dict):
                        beta_val = softmax_params.get("beta", 1.0)
                        epsilon_val = softmax_params.get("epsilon", 0.01)
                    else:
                        beta_val = 1.0 / max(prob_noise, 0.01)
                        epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
                    order_ll = cls._log_successors_queue_jump_ll(
                        obs_local,
                        H_sub,
                        beta=beta_val,
                        epsilon=epsilon_val
                    )
                elif noise_option == "frontier_softmax_uni_utility":
                    if isinstance(softmax_params, dict):
                        epsilon_val = softmax_params.get("epsilon", 0.01)
                    else:
                        epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
                    order_ll = cls._frontier_softmax_uni_utility_ll(
                        obs_local,
                        H_sub,
                        beta=1.0,
                        epsilon=epsilon_val
                    )
                
                pointwise_ll[(assessor, order_idx)] = order_ll
                total_ll += order_ll
                
        return total_ll, pointwise_ll




    # Flag to control Numba usage (set to False if issues arise)
    USE_NUMBA = True
    _numba_checked = False
    _numba_available = False
    
    @classmethod
    def _check_numba(cls):
        """Check if Numba is available (only once)."""
        if not cls._numba_checked:
            try:
                from src.utils.numba_accelerated import NUMBA_AVAILABLE
                cls._numba_available = NUMBA_AVAILABLE and cls.USE_NUMBA
            except ImportError:
                cls._numba_available = False
            cls._numba_checked = True
        return cls._numba_available


    @staticmethod
    def _logsumexp(values: List[float]) -> float:
        if not values:
            return -math.inf
        max_v = max(values)
        if max_v == -math.inf:
            return -math.inf
        return max_v + math.log(sum(math.exp(v - max_v) for v in values))

    @classmethod
    def _queue_jump_ll(
        cls,
        order_idx: List[int],
        H: np.ndarray,
        prob_noise: float,
    ) -> float:
        """
        NLE-based queue-jump likelihood (Nicholls-style):
          P(y_j | remaining) = (1-ε) * (NLE_first / NLE_total) + ε / |remaining|

        Uses log-space computation for large graphs to avoid overflow.

        Args:
            order_idx: Observed trace as indices [y_1, ..., y_T]
            H: Partial order adjacency (cover)
            prob_noise: Queue-jump probability ε

        Returns:
            Log-likelihood log P(Y | H, ε)
        """
        m = len(order_idx)
        if m <= 0:
            return 0.0
        n = H.shape[0]
        if n == 0:
            return 0.0
        eps = max(min(prob_noise, 0.99), 1e-10)
        logp = 0.0
        
        # Check if we need log-space (detect on first step)
        use_log_space = False
        if m > 0:
            h_test = H[np.ix_(order_idx, order_idx)]
            test_nle = cls.nle(h_test)
            if test_nle < 0:  # -2 indicates overflow
                use_log_space = True
        
        for j, y_j in enumerate(order_idx):
            remaining = order_idx[j:]
            if not remaining:
                break
            h_rem = H[np.ix_(remaining, remaining)]
            local_idx = remaining.index(y_j)
            
            if use_log_space:
                # Log-space computation for large graphs
                log_num_le = cls.log_nle(h_rem)
                log_num_first = cls.log_nle_first(h_rem, local_idx)
                
                if log_num_first == -math.inf:
                    # Item is not a top element, only noise term
                    prob_total = eps * (1.0 / (m - j))
                else:
                    # log(ratio) = log(num_first) - log(num_le)
                    log_ratio = log_num_first - log_num_le
                    ratio = math.exp(log_ratio) if log_ratio > -700 else 0.0
                    prob_no_jump = (1.0 - eps) * ratio
                    prob_jump = eps * (1.0 / (m - j))
                    prob_total = prob_no_jump + prob_jump
            else:
                # Standard computation for small graphs
                num_le = cls.nle(h_rem)
                if num_le <= 0:
                    return -1e10
                num_first = cls.nle_first(h_rem, local_idx)
                prob_no_jump = (1.0 - eps) * (num_first / num_le)
                prob_jump = eps * (1.0 / (m - j))
                prob_total = prob_no_jump + prob_jump
            
            logp += math.log(max(prob_total, 1e-300))
        return logp

    @classmethod
    def _log_successors_queue_jump_ll(
        cls,
        order_idx: List[int],
        H: np.ndarray,
        beta: float = 1.0,
        epsilon: float = 0.01,
    ) -> float:
        """
        Log-successors-based frontier-softmax likelihood (Eq. 3-4 in paper):
          Q(a) = log(1 + S_t(a))  where S_t(a) = |{b ∈ R_t \ {a} : a ≻ b}|
          P(y_t) = (1-ε)·exp(β·Q(y_t)) / Σ exp(β·Q(a)) + ε/|R_t|   if y_t in F_t
                 = ε/|R_t|                                         otherwise
        
        NO canonical ordering L* is used - this is a plan-free likelihood that
        maximizes future choice by preferring actions with more remaining successors.
        
        Args:
            order_idx: Observed trace as indices [y_1, y_2, ..., y_T]
            H: Partial order matrix (transitive closure)
            beta: Inverse temperature (higher = more deterministic)
            epsilon: Trembling-hand ε ∈ (0,1), probability of uniform slip
        
        Returns:
            Log-likelihood log P(Y | H, β, ε)
        """
        m = len(order_idx)
        if m <= 1:
            return 0.0
        if beta <= 0:
            return -1e10

        n = H.shape[0]
        if n == 0:
            return 0.0
        if any(y < 0 or y >= n for y in order_idx):
            return -1e10

        H_bool = H.astype(bool)
        successors = [np.flatnonzero(H_bool[u]).tolist() for u in range(n)]
        indegrees = H_bool.sum(axis=0).astype(int).tolist()

        # Trembling-hand epsilon
        eps = max(min(epsilon, 0.99), 1e-10)  # Clamp to (0, 1)

        # Initialize: all nodes remaining, frontier = nodes with no unmet predecessors
        remaining = set(range(n))
        unmet = indegrees[:]  # unmet prerequisites among remaining nodes
        frontier = {i for i in remaining if unmet[i] == 0}

        if not frontier:
            return -1e10  # No valid starting point

        logp = 0.0

        for y_t in order_idx:
            if y_t not in remaining:
                return -1e10
            if not frontier:
                return -1e10

            # Compute Q = log(1 + successors) for each node in frontier
            # S_t(a) = count of successors of 'a' that are still in remaining
            scores = []
            for a in frontier:
                num_successors = sum(1 for v in successors[a] if v in remaining and v != a)
                Q = math.log(num_successors + 1)  # Q = log(1 + S_t(a))
                scores.append(beta * Q)
            
            denom = cls._logsumexp(scores)
            if denom == -math.inf:
                return -1e10

            # Trembling-hand mixture
            p_noise = eps / float(len(remaining))
            p_total = p_noise

            if y_t in frontier:
                # Compute Q for y_t
                num_successors_y = sum(1 for v in successors[y_t] if v in remaining and v != y_t)
                Q_y = math.log(num_successors_y + 1)
                score_y = beta * Q_y
                p_rational = math.exp(score_y - denom)
                p_total += (1.0 - eps) * p_rational

            logp += math.log(max(p_total, 1e-300))

            # Update state after executing y_t
            remaining.remove(y_t)
            frontier.discard(y_t)

            # Add new frontier nodes (successors of y_t whose prerequisites are now all met)
            for v in successors[y_t]:
                if v in remaining:
                    unmet[v] -= 1
                    if unmet[v] == 0:
                        frontier.add(v)

        return logp

    @classmethod
    def _frontier_softmax_uni_utility_ll(
        cls,
        order_idx: List[int],
        H: np.ndarray,
        beta: float = 1.0,
        epsilon: float = 0.01,
    ) -> float:
        """
        Uniform-utility frontier-softmax likelihood (baseline for ablation):
          Q(a) = 0  for all a ∈ F_t  (uniform over frontier)
          P(y_t) = (1-ε) · 1/|F_t| + ε/|R_t|   if y_t in F_t
                 = ε/|R_t|                      otherwise
        
        This is the same as log_successors_queue_jump but with Q=0 (no successor preference).
        Beta parameter is ignored since Q=0 for all nodes.
        
        Args:
            order_idx: Observed trace as indices [y_1, y_2, ..., y_T]
            H: Partial order matrix (transitive closure)
            beta: Inverse temperature (ignored - kept for API compatibility)
            epsilon: Trembling-hand ε ∈ (0,1), probability of uniform slip
        
        Returns:
            Log-likelihood log P(Y | H, ε)
        """
        m = len(order_idx)
        if m <= 1:
            return 0.0

        n = H.shape[0]
        if n == 0:
            return 0.0
        if any(y < 0 or y >= n for y in order_idx):
            return -1e10

        H_bool = H.astype(bool)
        successors = [np.flatnonzero(H_bool[u]).tolist() for u in range(n)]
        indegrees = H_bool.sum(axis=0).astype(int).tolist()

        # Trembling-hand epsilon
        eps = max(min(epsilon, 0.99), 1e-10)  # Clamp to (0, 1)

        # Initialize: all nodes remaining, frontier = nodes with no unmet predecessors
        remaining = set(range(n))
        unmet = indegrees[:]  # unmet prerequisites among remaining nodes
        frontier = {i for i in remaining if unmet[i] == 0}

        if not frontier:
            return -1e10  # No valid starting point

        logp = 0.0

        for y_t in order_idx:
            if y_t not in remaining:
                return -1e10
            if not frontier:
                return -1e10

            # Uniform utility: all frontier nodes have equal probability
            # P(y_t | frontier) = 1/|F_t| for y_t in F_t
            frontier_size = len(frontier)
            
            # Trembling-hand mixture
            p_noise = eps / float(len(remaining))
            p_total = p_noise

            if y_t in frontier:
                # Uniform probability over frontier
                p_rational = 1.0 / frontier_size
                p_total += (1.0 - eps) * p_rational

            logp += math.log(max(p_total, 1e-300))

            # Update state after executing y_t
            remaining.remove(y_t)
            frontier.discard(y_t)

            # Add new frontier nodes (successors of y_t whose prerequisites are now all met)
            for v in successors[y_t]:
                if v in remaining:
                    unmet[v] -= 1
                    if unmet[v] == 0:
                        frontier.add(v)

        return logp
    
