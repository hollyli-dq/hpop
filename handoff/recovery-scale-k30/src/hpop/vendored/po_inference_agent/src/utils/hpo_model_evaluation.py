# src/assessment/hpo_model_evaluation.py
"""
Model-evaluation utilities for HPO / HCPO:

- Compute pointwise log-likelihood matrix ℓ_{i,s}
- WAIC (Watanabe, 2010)
- PSIS–LOO using ArviZ (Vehtari et al., 2017)

Requirements
------------
pip install arviz numpy pandas

ArviZ's az.loo implements PSIS–LOO as in:
Vehtari, A., Gelman, A., & Gabry, J. (2017).
"Practical Bayesian model evaluation using leave-one-out
cross-validation and WAIC."
"""

from typing import Dict, List, Any, Optional, Tuple, Hashable, Union

import math
import numpy as np
import pandas as pd
import pickle
try:
    import arviz as az
except Exception:
    az = None

from src.utils.po_accelerator_eta_accelerated import get_ultra_fast_eta_matrix
from src.utils.po_accelerator_nle_optimized import HPO_LogLikelihoodCache_Optimized
from src.utils.po_fun import BasicUtils


# ------------------------------------------------------------
#  Utility: log p(y_i | parameters) for a single list
# ------------------------------------------------------------
def single_list_loglik(
    e_list,
    cluster_id: int,
    U0: np.ndarray,
    U_a_dict: Dict[int, np.ndarray],
    M_a_dict: Dict[int, List[int]],
    h_U: Dict[int, np.ndarray],
    item_to_index: Dict[int, int],
    prob_noise: float,
    softmax_params: Optional[Dict[str, float]],
    noise_option: str,
    bar_eta_by_cluster: Dict[int, np.ndarray],
) -> float:
    """
    Compute log p(y_i | θ, U, c) using the optimized suffix NLE
    implementation already used by the sampler.
    """
    if not e_list:
        return 0.0

    choice_set = sorted(e_list)
    Ma = M_a_dict.get(cluster_id, [])
    if cluster_id not in h_U:
        return -np.inf

    h_a = h_U[cluster_id]
    local_map = {item: idx for idx, item in enumerate(choice_set)}
    sub_size = len(choice_set)
    h_sub = np.zeros((sub_size, sub_size), dtype=int)
    for r, item_r in enumerate(choice_set):
        if item_r in Ma:
            local_r = Ma.index(item_r)
            for c, item_c in enumerate(choice_set):
                if item_c in Ma:
                    local_c = Ma.index(item_c)
                    if local_r < len(h_a) and local_c < len(h_a):
                        h_sub[r, c] = h_a[local_r, local_c]

    obs_local = [local_map[item] for item in e_list if item in local_map]
    if not obs_local:
        return -np.inf

    if noise_option == "queue_jump":
        return HPO_LogLikelihoodCache_Optimized._queue_jump_ll(obs_local, h_sub, prob_noise)
    if noise_option == "weighted_queue_jump":
        bar_eta_full = bar_eta_by_cluster.get(cluster_id, None)
        if bar_eta_full is None:
            return -np.inf
        ma_index = {item: idx for idx, item in enumerate(Ma)}
        bar_eta = np.array([bar_eta_full[ma_index[item]] for item in choice_set])
        return HPO_LogLikelihoodCache_Optimized._wqueue_jump_ll(obs_local, h_sub, prob_noise, bar_eta)
    if noise_option == "softmax_queue_jump":
        if isinstance(softmax_params, dict):
            beta_val = softmax_params.get("beta", 1.0)
            epsilon_val = softmax_params.get("epsilon", 0.01)
        else:
            beta_val = 1.0 / max(prob_noise, 0.01)
            epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
        return HPO_LogLikelihoodCache_Optimized._softmax_queue_jump_ll(
            obs_local, h_sub, beta=beta_val, epsilon=epsilon_val
        )
    if noise_option == "log_successors_queue_jump":
        if isinstance(softmax_params, dict):
            beta_val = softmax_params.get("beta", 1.0)
            epsilon_val = softmax_params.get("epsilon", 0.01)
        else:
            beta_val = 1.0 / max(prob_noise, 0.01)
            epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
        return HPO_LogLikelihoodCache_Optimized._log_successors_queue_jump_ll(
            obs_local, h_sub, beta=beta_val, epsilon=epsilon_val
        )
    if noise_option == "frontier_softmax_uni_utility":
        if isinstance(softmax_params, dict):
            epsilon_val = softmax_params.get("epsilon", 0.01)
        else:
            epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
        return HPO_LogLikelihoodCache_Optimized._frontier_softmax_uni_utility_ll(
            obs_local, h_sub, beta=1.0, epsilon=epsilon_val
        )
    raise ValueError(f"Invalid noise_option: {noise_option}.")


# ------------------------------------------------------------
#  Build bar_eta (η̄) for each cluster from latent U
# ------------------------------------------------------------
def compute_bar_eta_cluster(
    U0: np.ndarray,
    U_a_dict: Dict[int, np.ndarray],
    M_a_dict: Dict[int, List[int]]
) -> Dict[int, np.ndarray]:
    """
    Compute bar_eta[k] = mean_j eta_{k,j} over the K latent coords,
    using the ultra-fast η matrix builder.
    """
    bar_eta: Dict[int, np.ndarray] = {}
    for k, Ua in U_a_dict.items():
        if k not in M_a_dict:
            # Skip cluster IDs that don't exist in M_a_dict
            continue
        
        Ma = M_a_dict[k]
        
        # Verify shapes match: Ua should have one row per item in Ma
        if Ua.shape[0] != len(Ma):
            raise ValueError(
                f"Shape mismatch for cluster {k}: Ua has {Ua.shape[0]} rows but M_a_dict has {len(Ma)} items. "
                "Ua and M_a_dict should have the same number of items. This indicates a data consistency issue."
            )
            
        eta_full = get_ultra_fast_eta_matrix(k, Ua, Ma)
        bar_eta[k] = eta_full.mean(axis=1)
    return bar_eta

# ------------------------------------------------------------
#  Main: Compute ℓ_{i,s} matrix  (n_units × S)
# ------------------------------------------------------------
def compute_pointwise_loglik_matrix(
    results: Dict[str, Any],
    units: List[List[int]],
    item_to_index: Dict[int, int],
    noise_option: str,
) -> np.ndarray:
    """
    Compute ℓ_{i,s} = log p(y_i | θ^{(s)}, U^{(s)}, c^{(s)}) for every
    predictive unit i and posterior draw s.

    Parameters
    ----------
    results : dict
        Full MCMC result dict (the *_full.pkl).
        Must contain:
          - "U0_trace", "Ua_trace", "H_trace"
          - "prob_noise_trace"
          - "c_vec_trace" (cluster assignments per iteration)
        Optional:
          - "M_a_dict_trace" (preferred): per-iteration M_a_dict
          - "M_a_final" (fallback): final M_a_dict, used to reconstruct per-iteration if trace not available
    units : list[list[int]]
        Predictive units y_i (e.g., observed lists), each as a list of item IDs.
    item_to_index : dict[int, int]
        Mapping from item ID to row index in U0.
    noise_option : str
        Noise model ("queue_jump", "weighted_queue_jump", or "softmax_queue_jump").

    Returns
    -------
    np.ndarray
        Array of shape (n_units, S) with ℓ_{i,s}.
    """
    import time

    # Basic sanity checks
    required_keys = ["U0_trace", "Ua_trace", "H_trace", "c_vec_trace"]
    if noise_option in ("softmax_queue_jump", "log_successors_queue_jump", "frontier_softmax_uni_utility"):
        required_keys += ["softmax_beta_trace", "epsilon_trace"]
    else:
        required_keys += ["prob_noise_trace", "epsilon_trace"]
    for k in required_keys:
        if k not in results:
            raise KeyError(f"results dict must contain key '{k}' for evaluation.")

    S = len(results["U0_trace"])
    n_units = len(units)
    
    # Verify c_vec_trace has correct length
    if len(results["c_vec_trace"]) != S:
        raise ValueError(f"c_vec_trace length ({len(results['c_vec_trace'])}) must match U0_trace length ({S})")
    
    # Verify M_a_dict_trace exists and has correct length
    if "M_a_dict_trace" not in results:
        raise KeyError(
            "M_a_dict_trace not found in results. "
            "Please re-run MCMC with the updated code that stores M_a_dict_trace."
        )
    if len(results["M_a_dict_trace"]) != S:
        raise ValueError(f"M_a_dict_trace length ({len(results['M_a_dict_trace'])}) must match U0_trace length ({S})")
    
    # Verify each c_vec has correct length
    for s in range(S):
        if len(results["c_vec_trace"][s]) != n_units:
            raise ValueError(f"c_vec_trace[{s}] length ({len(results['c_vec_trace'][s])}) must match units length ({n_units})")

    print(f"Computing log-likelihood matrix: {n_units} units × {S} samples = {n_units * S:,} evaluations")
    
    # Try to use tqdm for progress bar, fallback to simple prints
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("(Install 'tqdm' for progress bar: pip install tqdm)")

    # ℓ_{i,s}
    ell = np.zeros((n_units, S), dtype=float)

    # Progress tracking
    start_time = time.time()
    last_print_time = start_time
    print_interval = 5.0  # Print progress every 5 seconds if not using tqdm
    
    iterator = tqdm(range(S), desc="Processing samples") if use_tqdm else range(S)
    
    for s in iterator:
        U0 = np.asarray(results["U0_trace"][s])
        Ua_dict = results["Ua_trace"][s]         # dict[int, np.ndarray]
        H = results["H_trace"][s]               # dict[int, np.ndarray]
        
        if noise_option in ("softmax_queue_jump", "log_successors_queue_jump", "frontier_softmax_uni_utility"):
            softmax_beta = float(results["softmax_beta_trace"][s])
            epsilon_val = float(results["epsilon_trace"][s])
            prob_noise = epsilon_val
            softmax_params = {"beta": softmax_beta, "epsilon": epsilon_val}
        else:
            prob_noise = float(results["prob_noise_trace"][s])
            softmax_params = None
        c_vec_s = results["c_vec_trace"][s]     # cluster assignments for iteration s
        M_a_dict = results["M_a_dict_trace"][s]  # dict[int, list[int]] - per iteration
        
        # Validate M_a_dict type - handle case where it might be a list (trace)
        if isinstance(M_a_dict, list):
            # If we accidentally got a list of dicts, try to get the first one if it matches s?
            # But s is the index. If results["M_a_dict_trace"][s] returned a list, then the trace is nested?
            # Assuming it's a single dict for this iteration
            raise TypeError(f"M_a_dict for iteration {s} is a list, expected dict. Check trace structure.")

        bar_eta = compute_bar_eta_cluster(U0, Ua_dict, M_a_dict)

        for i, y_i in enumerate(units):
            c = c_vec_s[i]  # Use cluster assignment from iteration s
            # Check if cluster exists in M_a_dict and bar_eta for this iteration
            if c not in M_a_dict or c not in bar_eta:
                # Cluster doesn't exist in this iteration - set log-likelihood to -inf
                ell[i, s] = -np.inf
                continue
            ell[i, s] = single_list_loglik(
                e_list=y_i,
                cluster_id=c,
                U0=U0,
                U_a_dict=Ua_dict,
                M_a_dict=M_a_dict,
                h_U=H,
                item_to_index=item_to_index,
                prob_noise=prob_noise,
                softmax_params=softmax_params,
                noise_option=noise_option,
                bar_eta_by_cluster=bar_eta,
            )
        
        # Print progress if not using tqdm
        if not use_tqdm:
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                elapsed = current_time - start_time
                progress = (s + 1) / S
                rate = (s + 1) / elapsed if elapsed > 0 else 0
                eta = (S - s - 1) / rate if rate > 0 else 0
                print(f"  Progress: {s+1}/{S} samples ({progress*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                      f"Rate: {rate:.2f} samples/s")
                last_print_time = current_time

    total_time = time.time() - start_time
    print(f"✅ Completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Average rate: {S/total_time:.2f} samples/s")

    return ell


# ------------------------------------------------------------
#  Graph recovery / evaluation utilities
# ------------------------------------------------------------
def structural_hamming_distance(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    rev = (pred_adj == 1) & (true_adj == 0) & (true_adj.T == 1)
    additions = (pred_adj == 1) & (true_adj == 0) & ~(true_adj.T == 1)
    deletions = (pred_adj == 0) & (true_adj == 1) & ~(pred_adj.T == 1)
    return int(additions.sum() + deletions.sum() + rev.sum())


def _incomparable_pairs(closure: np.ndarray) -> set:
    """Extract incomparable pairs from a transitive closure matrix."""
    n = closure.shape[0]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if closure[i, j] or closure[j, i]:
                continue
            pairs.add((i, j))
    return pairs


def precision_recall(true_adj: np.ndarray, pred_adj: np.ndarray) -> Tuple[float, float]:
    tp = ((pred_adj == 1) & (true_adj == 1)).sum()
    fp = ((pred_adj == 1) & (true_adj == 0)).sum()
    fn = ((pred_adj == 0) & (true_adj == 1)).sum()
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    return precision, recall


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def edge_accuracy(true_adj: np.ndarray, pred_adj: np.ndarray) -> float:
    mask = ~np.eye(true_adj.shape[0], dtype=bool)
    return float((true_adj[mask] == pred_adj[mask]).mean())


def identifiability_metrics(
    edge_probs: np.ndarray, entropy_threshold: float
) -> Tuple[float, bool]:
    mask = ~np.eye(edge_probs.shape[0], dtype=bool)
    p = np.clip(edge_probs[mask], 1e-9, 1.0 - 1e-9)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    mean_entropy = float(entropy.mean()) if entropy.size else 0.0
    not_identifiable = mean_entropy >= entropy_threshold
    return mean_entropy, not_identifiable


def log_mean_exp(values: np.ndarray) -> float:
    max_v = float(np.max(values))
    return max_v + math.log(float(np.mean(np.exp(values - max_v))))


# ------------------------------------------------------------
#  WAIC calculation
# ------------------------------------------------------------
def compute_waic(loglik_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute WAIC and related quantities from ℓ_{i,s}.

    Parameters
    ----------
    loglik_matrix : np.ndarray
        Shape (n_units, S), where rows are units i and columns are draws s.

    Returns
    -------
    dict
        {
          "WAIC": ...,
          "elpd_WAIC": ...,
          "p_WAIC": ...,
          "se_elpd": ...,
        }
    """
    n_units, S = loglik_matrix.shape

    # 1. lppd_i  (stable log-sum-exp over s)
    m_i = np.max(loglik_matrix, axis=1)
    lppd_i = m_i + np.log(
        np.mean(np.exp(loglik_matrix - m_i[:, None]), axis=1)
    )

    # 2. Variance term for p_WAIC
    var_ddof = 1 if S > 1 else 0
    v_i = np.var(loglik_matrix, axis=1, ddof=var_ddof)  # Var_s(ℓ_{is})

    p_waic = float(np.sum(v_i))
    elpd_waic = float(np.sum(lppd_i) - p_waic)
    waic = float(-2.0 * elpd_waic)

    # Per-unit elpd contributions: elpd_i = lppd_i - v_i
    elpd_i = lppd_i - v_i
    se_elpd = float(np.sqrt(n_units * np.var(elpd_i, ddof=var_ddof)))

    return {
        "WAIC": waic,
        "elpd_WAIC": elpd_waic,
        "p_WAIC": p_waic,
        "se_elpd": se_elpd,
    }


def compute_waic_from_results(
    results: Dict[str, Any],
    observed_orders: Dict[int, List[List[int]]],
    O_a_i_dict: Dict[int, List[List[int]]],
    M_a_dict: Dict[int, List[int]],
    item_to_index: Dict[int, int],
    *,
    burn_in_fraction: float,
    noise_option: str = "softmax_queue_jump",
) -> Tuple[float, float, float]:
    h_trace = results.get("H_trace", [])
    U0_trace = results.get("U0_trace", [])
    Ua_trace = results.get("Ua_trace", [])
    if not h_trace or len(h_trace) != len(U0_trace) or len(h_trace) != len(Ua_trace):
        raise ValueError("Trace lengths are inconsistent; cannot compute WAIC.")

    if noise_option in ("softmax_queue_jump", "log_successors_queue_jump", "frontier_softmax_uni_utility"):
        softmax_beta_trace = results.get("softmax_beta_trace", [])
        epsilon_trace = results.get("epsilon_trace", [])
        if len(softmax_beta_trace) != len(h_trace) or len(epsilon_trace) != len(h_trace):
            raise ValueError("Softmax traces do not match H_trace length.")
    else:
        prob_noise_trace = results.get("prob_noise_trace", [])
        epsilon_trace = results.get("epsilon_trace", [])
        if len(prob_noise_trace) != len(h_trace) or len(epsilon_trace) != len(h_trace):
            raise ValueError("Noise traces do not match H_trace length.")

    assessors = sorted(observed_orders.keys())
    obs_keys = []
    for assessor in assessors:
        for order_idx in range(len(observed_orders[assessor])):
            obs_keys.append((assessor, order_idx))

    burn_in = int(len(h_trace) * burn_in_fraction)
    post_len = len(h_trace) - burn_in
    if post_len <= 0:
        raise ValueError("No posterior samples after burn-in for WAIC.")

    log_lik = np.zeros((len(obs_keys), post_len), dtype=float)

    for offset, s in enumerate(range(burn_in, len(h_trace))):
        h_U = h_trace[s]
        U0 = U0_trace[s]
        U_a_dict = Ua_trace[s]
        if noise_option in ("softmax_queue_jump", "log_successors_queue_jump", "frontier_softmax_uni_utility"):
            softmax_beta = float(softmax_beta_trace[s])
            epsilon_val = float(epsilon_trace[s])
            prob_noise = epsilon_val
            softmax_params = {"beta": softmax_beta, "epsilon": epsilon_val}
        else:
            prob_noise = float(prob_noise_trace[s])
            softmax_params = None

        _, pointwise = HPO_LogLikelihoodCache_Optimized.calculate_log_likelihood_hpo_with_pointwise(
            U={"U0": U0, "U_a_dict": U_a_dict},
            h_U=h_U,
            observed_orders=observed_orders,
            M_a_dict=M_a_dict,
            O_a_i_dict=O_a_i_dict,
            item_to_index=item_to_index,
            prob_noise=prob_noise,
            softmax_params=softmax_params,
            noise_option=noise_option,
        )
        for idx, key in enumerate(obs_keys):
            log_lik[idx, offset] = float(pointwise.get(key, 0.0))

    stats = compute_waic(log_lik)
    waic = float(stats["WAIC"])
    p_waic = float(stats["p_WAIC"])
    lppd = float(stats["elpd_WAIC"] + p_waic)
    return waic, lppd, p_waic


# ------------------------------------------------------------
#  PSIS–LOO via ArviZ
# ------------------------------------------------------------
def compute_psis_loo(loglik_matrix: np.ndarray,
                     results: Dict[str, Any]):
    """
    Compute PSIS–LOO using ArviZ from:
      - loglik_matrix: shape (n_units, S) = (obs, draws)
      - results: MCMC result dict containing e.g. 'prob_noise_trace'

    We build a real posterior group from prob_noise_trace.
    """
    if az is None:
        raise ImportError("arviz is required for PSIS-LOO; install with `pip install arviz`.")
    # 1) Shapes for ArviZ
    # loglik_matrix: (n_units, S) -> (chain=1, draw=S, obs=n_units)
    ll = loglik_matrix.T[None, :, :]   # shape (1, S, n_units)
    n_draws = ll.shape[1]

    # 2) Use a real parameter as posterior, e.g. prob_noise_trace
    if "prob_noise_trace" not in results:
        raise KeyError("results must contain 'prob_noise_trace' to build posterior")

    prob_noise = np.asarray(results["prob_noise_trace"])
    if prob_noise.shape[0] != n_draws:
        raise ValueError(
            f"prob_noise_trace has length {prob_noise.shape[0]} but we have {n_draws} draws"
        )

    # posterior: one chain, S draws, scalar per draw
    posterior = {"prob_noise": prob_noise[None, :]}  # shape (1, S)

    # 3) Build InferenceData
    idata = az.from_dict(
        posterior=posterior,
        log_likelihood={"y": ll},
    )

    # 4) PSIS–LOO
    loo = az.loo(idata, pointwise=True)
    return loo


# ------------------------------------------------------------
#  Baseline Methods for Comparison
# ------------------------------------------------------------

def baseline_always_before_intersection(
    observed_orders: List[List[int]],
    n_items: int
) -> np.ndarray:
    """
    Baseline A: Always-Before (Intersection)
    
    Add edge i→j if and only if i precedes j in ALL traces.
    Then apply transitive reduction.
    
    Args:
        observed_orders: List of observed traces, each as a list of item indices
        n_items: Total number of items (items should be 0-indexed from 0 to n_items-1)
    
    Returns:
        Binary adjacency matrix (transitively reduced)
    """
    if not observed_orders:
        return np.zeros((n_items, n_items), dtype=np.int8)
    
    # Count how many times i precedes j across all traces
    # For intersection, we need i to precede j in ALL traces
    precedes_count = np.zeros((n_items, n_items), dtype=int)
    
    for order in observed_orders:
        # Build position mapping for this trace
        pos = {item: idx for idx, item in enumerate(order) if 0 <= item < n_items}
        
        # For each pair (i, j) where both appear in this trace
        for i in range(n_items):
            if i not in pos:
                continue
            for j in range(n_items):
                if i == j:
                    continue
                if j not in pos:
                    continue
                # Check if i precedes j in this trace
                if pos[i] < pos[j]:
                    precedes_count[i, j] += 1
    
    # Intersection: i→j only if i precedes j in ALL traces
    num_traces = len(observed_orders)
    adj = (precedes_count == num_traces).astype(np.int8)
    
    # Ensure transitivity (closure) before reduction
    closure = BasicUtils.transitive_closure(adj.astype(np.int8))
    
    # Apply transitive reduction
    reduced = BasicUtils.transitive_reduction(closure.astype(int))
    
    return reduced.astype(np.int8)


def _has_cycles(adj: np.ndarray) -> bool:
    """Check if directed graph has cycles using DFS."""
    n = adj.shape[0]
    visited = [False] * n
    
    def dfs(node, rec_stack):
        """DFS helper that returns True if cycle found."""
        visited[node] = True
        rec_stack.add(node)
        
        # Check all neighbors
        neighbors = np.where(adj[node] == 1)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                if dfs(neighbor, rec_stack):
                    return True
            elif neighbor in rec_stack:
                # Found back edge - cycle exists
                return True
        
        rec_stack.remove(node)
        return False
    
    # Check each connected component
    for i in range(n):
        if not visited[i]:
            rec_stack = set()
            if dfs(i, rec_stack):
                return True
    
    return False


def _break_cycles_greedy(adj: np.ndarray, prob_matrix: np.ndarray) -> np.ndarray:
    """
    Break cycles by removing edges with lowest probabilities.
    Uses a greedy approach: repeatedly remove lowest-probability edge until acyclic.
    """
    adj_copy = adj.copy()
    max_iterations = adj.shape[0] ** 2  # Safety limit
    
    for iteration in range(max_iterations):
        if not _has_cycles(adj_copy):
            break
        
        # Find edge with minimum probability among all edges
        n = adj_copy.shape[0]
        min_prob = float('inf')
        min_edge = None
        
        for i in range(n):
            for j in range(n):
                if adj_copy[i, j] == 1:
                    prob = prob_matrix[i, j]
                    if prob < min_prob:
                        min_prob = prob
                        min_edge = (i, j)
        
        if min_edge is not None:
            # Remove this edge
            adj_copy[min_edge[0], min_edge[1]] = 0
        else:
            # No edges left, break
            break
    
    return adj_copy


def baseline_majority_projection(
    observed_orders: List[List[int]],
    n_items: int,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Baseline B: Majority + Projection
    
    Add edge i→j if p(i before j) > threshold.
    Then project to acyclic graph via transitive closure/reduction.
    
    Args:
        observed_orders: List of observed traces, each as a list of item indices
        n_items: Total number of items (items should be 0-indexed from 0 to n_items-1)
        threshold: Probability threshold (default 0.5 for majority)
    
    Returns:
        Binary adjacency matrix (transitively reduced, acyclic)
    """
    if not observed_orders:
        return np.zeros((n_items, n_items), dtype=np.int8)
    
    # Count how many times i precedes j across all traces
    precedes_count = np.zeros((n_items, n_items), dtype=int)
    pair_count = np.zeros((n_items, n_items), dtype=int)  # How many traces contain both i and j
    
    for order in observed_orders:
        # Build position mapping for this trace
        pos = {item: idx for idx, item in enumerate(order) if 0 <= item < n_items}
        items_in_trace = set(pos.keys())
        
        # For each pair (i, j) where both appear in this trace
        for i in items_in_trace:
            for j in items_in_trace:
                if i == j:
                    continue
                pair_count[i, j] += 1
                # Check if i precedes j in this trace
                if pos[i] < pos[j]:
                    precedes_count[i, j] += 1
    
    # Compute probability: p(i before j) = count(i before j) / count(both appear)
    # Avoid division by zero
    prob_matrix = np.zeros((n_items, n_items), dtype=float)
    mask = pair_count > 0
    prob_matrix[mask] = precedes_count[mask] / pair_count[mask]
    
    # Majority rule: add edge i→j if p(i before j) > threshold
    adj = (prob_matrix > threshold).astype(np.int8)
    
    # Project to acyclic: ensure no cycles
    # First, resolve bidirectional edges by keeping only the direction with higher probability
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            if adj[i, j] == 1 and adj[j, i] == 1:
                # Both directions present, keep only the one with higher probability
                if prob_matrix[i, j] > prob_matrix[j, i]:
                    adj[j, i] = 0
                elif prob_matrix[j, i] > prob_matrix[i, j]:
                    adj[i, j] = 0
                else:
                    # Tie: break by lexicographic order (keep i→j if i < j)
                    if i < j:
                        adj[j, i] = 0
                    else:
                        adj[i, j] = 0
    
    # Break any remaining cycles (e.g., 3-cycles or longer)
    adj = _break_cycles_greedy(adj, prob_matrix)
    
    # Ensure transitivity (closure) before reduction
    closure = BasicUtils.transitive_closure(adj.astype(np.int8))
    
    # Apply transitive reduction
    reduced = BasicUtils.transitive_reduction(closure.astype(int))
    
    return reduced.astype(np.int8)


def _dedup_preserve_order(xs: List[int]) -> List[int]:
    """Deduplicate list while preserving order of first occurrence."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _to_cover(adj: np.ndarray) -> np.ndarray:
    """Convert adjacency matrix to cover (transitive closure then reduction).
    
    This ensures the matrix is in canonical cover form (transitively reduced).
    Matches the approach used by baseline_always_before_intersection and baseline_majority_projection.
    """
    closure = BasicUtils.transitive_closure(adj.astype(np.int8))
    # Use same method as baseline functions for consistency
    cover = BasicUtils.transitive_reduction(closure.astype(int))
    return cover.astype(np.int8)


def evaluate_baselines(
    observed_orders: Dict[Hashable, List[List[int]]],
    M_a_dict: Dict[Hashable, List[int]],
    true_cover: Dict[Hashable, np.ndarray],
    baseline_method: str = "both",
    majority_threshold: float = 0.5,
    store_matrices: bool = False,
    strict_checks: bool = True
) -> Dict[Hashable, Dict[str, Any]]:
    """
    Evaluate baseline methods against true partial orders.
    
    Args:
        observed_orders: Dict mapping assessor_id -> list of observed traces (global indices)
        M_a_dict: Dict mapping assessor_id -> list of items (global indices) in that assessor.
                  The order of items defines the local index space: item at position i maps to local index i.
        true_cover: Dict mapping assessor_id -> true cover matrix (local indices, n_items × n_items).
                    Must match the ordering in M_a_dict[assessor_id].
        baseline_method: "always_before", "majority", or "both"
        majority_threshold: Threshold for majority baseline (default 0.5)
        store_matrices: If True, include baseline matrices in results (default False to save memory)
        strict_checks: If True, perform strict correctness checks (default True)
    
    Returns:
        Dictionary mapping assessor_id -> {baseline_name -> {metrics}} for each baseline
        
    Correctness Requirements:
        - M_a_dict[assessor_id] must list items in the SAME order used to build true_cover[assessor_id]
        - true_cover[assessor_id].shape must equal (len(M_a_dict[assessor_id]), len(M_a_dict[assessor_id]))
    """
    results = {}
    
    for assessor_id, orders in observed_orders.items():
        if assessor_id not in M_a_dict:
            continue
        
        items = M_a_dict[assessor_id]
        n_items = len(items)

        # Map global indices to local indices for orders
        # CRITICAL: This assumes M_a_dict[assessor_id] order matches true_cover ordering
        global_to_local = {item: idx for idx, item in enumerate(items)}
        local_orders = []
        for order in orders:
            # Deduplicate while preserving order (removes consecutive duplicates)
            order_dedup = _dedup_preserve_order(order)
            local_order = [global_to_local[g] for g in order_dedup if g in global_to_local]
            if local_order:
                local_orders.append(local_order)
        
        if not local_orders:
            continue
        
        true_h = true_cover.get(assessor_id)
        if true_h is None:
            continue
        
        # CORRECTNESS CHECK #1: Size matching
        if strict_checks:
            assert true_h.shape == (n_items, n_items), (
                f"Size mismatch for assessor {assessor_id}: "
                f"true_cover shape {true_h.shape} != expected {(n_items, n_items)}. "
                f"This indicates M_a_dict[{assessor_id}] ordering does not match true_cover[{assessor_id}] ordering."
            )
        
        # Ensure true_h is a cover matrix (no diagonal, no transitive redundancy)
        # Most implementations already provide cover, but this ensures consistency
        true_h_clean = true_h.copy()
        np.fill_diagonal(true_h_clean, 0)
        # Always convert to cover for fair comparison (idempotent if already cover)
        true_h_cover = _to_cover(true_h_clean)
        
        assessor_results = {}
        
        # Baseline A: Always-Before (Intersection)
        if baseline_method in ("always_before", "both"):
            baseline_a = baseline_always_before_intersection(local_orders, n_items)
            # Both baseline functions already return cover (transitively reduced)
            # Additional safety: ensure it's a cover (idempotent if already cover)
            baseline_a_cover = _to_cover(baseline_a)
            
            precision_a, recall_a = precision_recall(true_h_cover, baseline_a_cover)
            f1_a = f1_score(precision_a, recall_a)
            shd_a = structural_hamming_distance(true_h_cover, baseline_a_cover)
            
            assessor_results["always_before"] = {
                "precision": precision_a,
                "recall": recall_a,
                "f1": f1_a,
                "shd": shd_a,
            }
            if store_matrices:
                assessor_results["always_before"]["matrix"] = baseline_a_cover
        
        # Baseline B: Majority + Projection
        if baseline_method in ("majority", "both"):
            baseline_b = baseline_majority_projection(
                local_orders, n_items, threshold=majority_threshold
            )
            # Already returns cover, but ensure consistency (idempotent if already cover)
            baseline_b_cover = _to_cover(baseline_b)
            
            precision_b, recall_b = precision_recall(true_h_cover, baseline_b_cover)
            f1_b = f1_score(precision_b, recall_b)
            shd_b = structural_hamming_distance(true_h_cover, baseline_b_cover)
            
            assessor_results["majority"] = {
                "precision": precision_b,
                "recall": recall_b,
                "f1": f1_b,
                "shd": shd_b,
                "threshold": majority_threshold
            }
            if store_matrices:
                assessor_results["majority"]["matrix"] = baseline_b_cover
        
        results[assessor_id] = assessor_results
    
    return results


# ------------------------------------------------------------
#  High-level pipeline: from *_full.pkl → WAIC / LOO
# ------------------------------------------------------------
def evaluate_model(full_pkl_path: str, noise_option: str = "queue_jump", use_psis_loo: bool = True):
    """
    Main entry point:
    - Loads a *_full.pkl results dict
    - Reconstructs ℓ_{i,s}
    - Computes WAIC and (optionally) PSIS–LOO.

    Parameters
    ----------
    full_pkl_path : str
        Path to the saved *_full.pkl file.
    noise_option : str
        Noise model string used in the original fit.
    use_psis_loo : bool
        If True, also compute PSIS–LOO via ArviZ.

    Returns
    -------
    dict
        {
          "WAIC": {...},
          "LOO": arviz.ELPDData  # present only if use_psis_loo=True
        }
    """
    print(f"📦 Loading MCMC results from: {full_pkl_path}")
    with open(full_pkl_path, "rb") as f:
        res = pickle.load(f)

    # You might need to adapt this depending on how you stored the data.
    # Here we assume:
    #   - res["observed_orders"] is a list of lists of item IDs (one per list)
    #   - res["c_vec_final"]      is the cluster assignment for each list
    units = res["observed_orders"]
    c_vec = res["c_vec_final"]

    items = res["items"]
    item_to_index = {item: i for i, item in enumerate(items)}

    # ℓ_{i,s}
    ell = compute_pointwise_loglik_matrix(
        results=res,
        units=units,
        item_to_index=item_to_index,
        noise_option=noise_option,
    )

    print("📊 Computing WAIC …")
    waic = compute_waic(ell)
    output: Dict[str, Any] = {"WAIC": waic}

    if use_psis_loo:
        print("📊 Computing PSIS–LOO …")
        loo = compute_psis_loo(ell)
        output["LOO"] = loo

    return output


def critical_pairs_from_closure(closure: np.ndarray) -> set[tuple[int, int]]:
    """Critical pairs = incomparable pairs under the TRUE poset closure."""
    n = closure.shape[0]
    cp = set()
    for i in range(n):
        for j in range(i + 1, n):
            if closure[i, j] == 0 and closure[j, i] == 0:
                cp.add((i, j))
    return cp


def build_probability_matrices(h_trace: list) -> dict:
    """Build probability matrices from MCMC traces by averaging."""
    if not h_trace:
        return {}

    # h_trace is a list of dicts, each dict maps assessor_id to matrix
    assessors = list(h_trace[0].keys())
    result = {}

    for assessor in assessors:
        # Get all matrices for this assessor
        matrices = [sample[assessor] for sample in h_trace if assessor in sample]
        if not matrices:
            continue

        # Stack and average to get probabilities
        matrix_stack = np.stack(matrices, axis=0)
        prob_matrix = np.mean(matrix_stack, axis=0)

        result[assessor] = prob_matrix

    return result


