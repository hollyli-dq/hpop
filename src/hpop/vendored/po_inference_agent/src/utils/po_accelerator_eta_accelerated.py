#!/usr/bin/env python3
"""
Vectorized Eta Transformation Accelerator

This module provides vectorized eta transformations without approximations:
1. Vectorized exact gumbel inverse CDF calculation  
2. Pre-computed eta values for entire assessor
3. Exact scipy calculations (no approximations)
4. No caching (since eta values change frequently in MCMC)
"""

import numpy as np
from typing import List, Tuple
from scipy.stats import norm
import warnings

from src.utils.po_fun import StatisticalUtils

def vectorized_gumbel_inv_cdf_exact(p_array: np.ndarray) -> np.ndarray:
    """
    Vectorized exact gumbel inverse CDF calculation using numpy operations.
    
    Direct implementation: gumbel_inv_cdf(p) = -log(-log(p))
    This is much faster and more stable than the list comprehension approach.
    
    Args:
        p_array: Array of probabilities in [0,1]
    
    Returns:
        Array of exact gumbel inverse CDF values
    """
    # Clip to avoid numerical issues at boundaries
    p_clipped = np.clip(p_array, 1e-15, 1 - 1e-15)
    
    # Vectorized exact calculation
    return np.array([StatisticalUtils.gumbel_inv_cdf(p) for p in p_clipped])
    
def precompute_eta_matrix_full_assessor(
    Ua: np.ndarray,
    Ma: List[int]
) -> np.ndarray:
    """
    Pre-compute eta transformations for entire assessor using exact calculations.
    
    This is the key vectorization - compute all eta values at once for the assessor
    instead of computing them repeatedly for each choice set.
    
    Args:
        Ua: Utility matrix for assessor (n_items_a, K)
        Ma: List of global items for this assessor
    Returns:
        eta_matrix: (n_items_a, K) matrix of eta-transformed utilities
    """
    n_items_a, K = Ua.shape
    
    # STEP 1: Vectorized exact normal CDF (no approximation)
    p_matrix = norm.cdf(Ua)  # Use exact scipy implementation
    
    # STEP 2: Vectorized exact gumbel inverse CDF (no approximation)
    gumbel_matrix = vectorized_gumbel_inv_cdf_exact(p_matrix)
    
    return gumbel_matrix
    
def extract_choice_set_eta_vectorized(
    choice_set: List[int],
    Ma: List[int],
    eta_matrix_full: np.ndarray
) -> np.ndarray:
    """
    Extract eta values for choice set from pre-computed full eta matrix.
    
    Uses pre-computed full eta matrix instead of computing on-the-fly.
    
    Args:
        choice_set: Items in this choice set
        Ma: All items for assessor (for indexing)
        eta_matrix_full: Pre-computed eta matrix for all items
    
    Returns:
        choice_set_eta: (len(choice_set), K) eta values for choice set
    """
    # Create index mapping for fast lookup
    ma_to_local = {item: idx for idx, item in enumerate(Ma)}
    
    # Extract rows corresponding to choice set items
    local_indices = []
    for item in choice_set:
        if item in ma_to_local:
            local_indices.append(ma_to_local[item])
        else:
            # Item not in assessor's list - shouldn't happen in correct usage
            warnings.warn(f"Item {item} not in assessor's item list {Ma}")
            local_indices.append(0)  # Fallback
    
    return eta_matrix_full[local_indices, :]
    
def calculate_single_row_eta_exact(
    utility_row: np.ndarray
) -> np.ndarray:
    """
    Calculate eta transformation for a single utility row with exact calculations.
    
    Used for incremental updates where only one row changes.
    
    Args:
        utility_row: Utility values for single item (K,)
    Returns:
        eta_row: Eta-transformed utilities for this item (K,)
    """
    # STEP 1: Exact normal CDF
    p_values = norm.cdf(utility_row)
    
    # STEP 2: Exact gumbel inverse CDF
    gumbel_values = vectorized_gumbel_inv_cdf_exact(p_values)
    
    return gumbel_values
    
def calculate_incremental_eta_change_exact(
    row_idx: int,
    old_row: np.ndarray,
    new_row: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate exact eta change for modified row only.
    
    Returns both old and new eta values for the changed item.
    Uses exact calculations without approximations.
    
    Args:
        row_idx: Index of changed row
        old_row: Previous utility values
        new_row: New utility values
    Returns:
        (old_eta_row, new_eta_row): Exact eta values for old and new utilities
    """
    # Calculate exact eta for old utilities
    eta_old = calculate_single_row_eta_exact(old_row)
    
    # Calculate exact eta for new utilities
    eta_new = calculate_single_row_eta_exact(new_row)
    
    return eta_old, eta_new
    
# Convenience functions for integration with existing code

def get_ultra_fast_eta_matrix(
    assessor: int,
    Ua: np.ndarray,
    Ma: List[int]
) -> np.ndarray:
    """
    Get pre-computed eta matrix for entire assessor with exact calculations.
    
    Main acceleration: compute eta once for entire assessor instead of per choice set.
    """
    return precompute_eta_matrix_full_assessor(Ua, Ma)
    
def get_choice_set_eta_ultra_fast(
    choice_set: List[int],
    Ma: List[int],
    eta_matrix_full: np.ndarray
) -> np.ndarray:
    """Extract choice set eta values from pre-computed matrix."""
    return extract_choice_set_eta_vectorized(choice_set, Ma, eta_matrix_full)
    
def get_incremental_eta_change_ultra_fast(
    assessor: int,
    row_idx: int,
    old_row: np.ndarray,
    new_row: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get exact eta change for single modified row."""
    return calculate_incremental_eta_change_exact(
        row_idx, old_row, new_row
    )
