import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from scipy.stats import norm
import networkx as nx
from tqdm import tqdm
import time
import random
import math
import json
import yaml
import logging
from datetime import datetime

# Import from the package
from src.utils.po_fun import BasicUtils, StatisticalUtils

import threading
from concurrent.futures import ThreadPoolExecutor

##########################################################
#                   LogLikelihoodCache
##########################################################
class LogLikelihoodCache:
    """
    Caches and parallelizes the computation of number of linear extensions (nle) and
    number of extensions with a specific first item (nle_first).
    """

    # Class-level dictionaries for caching
    nle_cache = {}
    nle_first_cache = {}

    # Thread pool for parallel computations
    _pool = ThreadPoolExecutor(max_workers=16)

    # A lock to ensure thread-safe reads/writes to the caches
    _cache_lock = threading.Lock()

    @staticmethod
    def _matrix_key(adj_matrix: np.ndarray) -> bytes:
        """Convert adjacency matrix to a bytes object as a cache key."""
        return adj_matrix.tobytes()

    @classmethod
    def _get_nle(cls, adj_matrix: np.ndarray) -> int:
        """
        Retrieve or compute the number of linear extensions with caching,
        offloading the BasicUtils.nle(...) computation to the thread pool.
        """
        key = cls._matrix_key(adj_matrix)

        # Check the cache under the lock
        with cls._cache_lock:
            if key in cls.nle_cache:
                return cls.nle_cache[key]

        # If not cached, compute in a worker thread

            future = cls._pool.submit(BasicUtils.nle, adj_matrix)
            val = future.result()  # Wait for worker to finish

        # Store result back to cache
        with cls._cache_lock:
            cls.nle_cache[key] = val
        return val

    @classmethod
    def _get_nle_first(cls, adj_matrix: np.ndarray, local_idx: int) -> int:
        """
        Retrieve or compute the number of linear extensions with a specific first item,
        using caching and parallel computation.
        """
        matrix_key = cls._matrix_key(adj_matrix)
        cache_key = (matrix_key, local_idx)

        # Check the cache
        with cls._cache_lock:
            if cache_key in cls.nle_first_cache:
                return cls.nle_first_cache[cache_key]


        future = cls._pool.submit(BasicUtils.num_extensions_with_first, adj_matrix, local_idx)
        val = future.result()

        # Update the cache
        with cls._cache_lock:
            cls.nle_first_cache[cache_key] = val

        return val
    @classmethod
    def _compute_bar_eta(cls, j, U0):
        """
        Suppose 'U0' is shape (n_global, K).
        For item 'j', we have a latent vector: U0[j,:].
        We define bar(eta_j) = average(U0[j,:]).
        """
        # If item j is an integer index, then:
        latent_vec = U0[j,:]  # shape (K,)
        bar_eta_j = np.mean(latent_vec)
        return bar_eta_j
    
    @classmethod  
    def _compute_bar_eta_from_local(cls, local_idx, Ua):
        """
        Compute bar_eta using assessor-specific utilities.
        
        Args:
            local_idx: Local index within the assessor's item list
            Ua: Assessor-specific utilities, shape (n_local, K)
        
        Returns:
            bar_eta_j = average(Ua[local_idx, :])
        """
        latent_vec = Ua[local_idx, :]  # shape (K,)
        bar_eta_j = np.mean(latent_vec)
        return bar_eta_j
    
    @classmethod
    def calculate_log_likelihood(
        cls,
        Z,
        h_Z,
        observed_orders_idx,
        choice_sets,
        item_to_index,
        prob_noise,
        softmax_params,
        noise_option
    ):
        if noise_option not in ["queue_jump"]:
            raise ValueError("Invalid noise_option. Valid options are ['queue_jump'].")

        log_likelihood = 0.0

        for idx, y_i in enumerate(observed_orders_idx):
            O_i = choice_sets[idx]
            O_i_indices = sorted([item_to_index[item] for item in O_i])
            m = len(y_i)

            if noise_option == "queue_jump":
                for j, y_j in enumerate(y_i):
                    remaining_indices = y_i[j:]
                    h_Z_remaining = h_Z[np.ix_(remaining_indices, remaining_indices)]

                    num_le = cls._get_nle(h_Z_remaining)  # parallel call
                    local_idx = remaining_indices.index(y_j)
                    num_first_item = cls._get_nle_first(h_Z_remaining, local_idx)

                    prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)
                    prob_jump = prob_noise * (1 / (m - j))
                    prob_observed = prob_no_jump + prob_jump
                    log_likelihood += math.log(prob_observed)

        return log_likelihood


##########################################################
#                   HPO_LogLikelihoodCache
##########################################################
class HPO_LogLikelihoodCache:
    """
    Similar to LogLikelihoodCache but used for hierarchical partial orders.
    Uses the same approach of parallelizing nle computations and caching results.
    """

    # Class-level dictionaries for caching
    nle_cache = {}
    nle_first_cache = {}

    # Thread pool and lock for concurrency
    _pool = ThreadPoolExecutor(max_workers=4)
    _cache_lock = threading.Lock()

    @staticmethod
    def _matrix_key(adj_matrix: np.ndarray) -> bytes:
        """Convert adjacency matrix to a bytes object as a cache key."""
        return adj_matrix.tobytes()

    @classmethod
    def _get_nle(cls, adj_matrix: np.ndarray) -> int:
        """Retrieve or compute the number of linear extensions with caching & parallel."""
        key = cls._matrix_key(adj_matrix)   ### ADDINGT THE LOCK TO READ THE CACHE 
        with cls._cache_lock:
            if key in cls.nle_cache:
                return cls.nle_cache[key]

        future = cls._pool.submit(BasicUtils.nle, adj_matrix)
        val = future.result()

        with cls._cache_lock:
            cls.nle_cache[key] = val
        return val

    @classmethod
    def _get_nle_first(cls, adj_matrix: np.ndarray, local_idx: int) -> int:
        """Retrieve or compute the number of extensions with a specific first item, in parallel."""
        matrix_key = cls._matrix_key(adj_matrix)
        cache_key = (matrix_key, local_idx)

        with cls._cache_lock:
            if cache_key in cls.nle_first_cache:
                return cls.nle_first_cache[cache_key]

 
        future = cls._pool.submit(BasicUtils.num_extensions_with_first, adj_matrix, local_idx)
        val = future.result()

        with cls._cache_lock:
            cls.nle_first_cache[cache_key] = val
        return val
    @classmethod
    def _compute_bar_eta(cls, j, U0):
        """
        Suppose 'U0' is shape (n_global, K).
        For item 'j', we have a latent vector: U0[j,:].
        We define bar(eta_j) = average(U0[j,:]).
        """
        # If item j is an integer index, then:
        latent_vec = U0[j,:]  # shape (K,)
        bar_eta_j = np.mean(latent_vec)
        return bar_eta_j
    
    @classmethod  
    def _compute_bar_eta_from_local(cls, local_idx, Ua):
        """
        Compute bar_eta using assessor-specific utilities.
        
        Args:
            local_idx: Local index within the assessor's item list
            Ua: Assessor-specific utilities, shape (n_local, K)
        
        Returns:
            bar_eta_j = average(Ua[local_idx, :])
        """
        latent_vec = Ua[local_idx, :]  # shape (K,)
        bar_eta_j = np.mean(latent_vec)
        return bar_eta_j
    

    @staticmethod
    def _bar_eta_choice_from_Ua(
        choice_set: List[int],
        Ma: List[int],                # assessor's items (global IDs) in local order
        Ua: np.ndarray               # (n_local, K)
    ) -> np.ndarray:
        """
        Return bar-eta per item in `choice_set` using Ua:
          η_{i,k} = gumbel_inv_cdf( Phi( Ua[local_i,k] ) )
          bar_eta_i = mean_k η_{i,k}
        The output is aligned to `choice_set` order. If an item is not in `Ma`,
        it falls back to using zeros for missing utilities.
        """
        if not choice_set:
            return np.empty(0)

        # Map choice_set global IDs to local indices in Ua; -1 for missing
        local_idx = np.array([Ma.index(g) if g in Ma else -1 for g in choice_set], dtype=int)

        # Build a Ua submatrix aligned with choice_set; for missing, use zeros
        K = Ua.shape[1] if Ua.size > 0 else 0
        Ua_sub = np.zeros((len(choice_set), K))
        present_mask = local_idx >= 0
        if K > 0 and present_mask.any():
            Ua_sub[present_mask, :] = Ua[local_idx[present_mask], :]

        p = norm.cdf(Ua_sub)
        g = StatisticalUtils.gumbel_inv_cdf(p)

        return g.mean(axis=1)
      
    @classmethod
    def calculate_log_likelihood_hpo(
        cls,
        U,                # global + local latents
        h_U,              # { assessor : adjacency_matrix_local_items }
        observed_orders,  # { assessor : [observed_orders], ... }
        M_a_dict,
        O_a_i_dict,       # { assessor : [choice_set for each task], ... }
        item_to_index,    # item→int map
        prob_noise,
        softmax_params,
        noise_option
    ):
        if noise_option not in ["queue_jump", "weighted_queue_jump"]:
            raise ValueError(f"Invalid noise_option: {noise_option}.")
        if len(observed_orders) == 0:
            return 0.0

        log_likelihood = 0.0
        U0 = U.get("U0", np.array([]))
        U_a_dict = U.get("U_a_dict", {})

        # For each assessor, we go through the tasks
        for a in O_a_i_dict.keys():
            tasks_choice_sets = O_a_i_dict[a]
            tasks_observed = observed_orders.get(a, [])
            Ma = M_a_dict.get(a, [])
            
            # Get assessor-specific utilities (needed for weighted_queue_jump)
            Ua = U_a_dict.get(a, np.array([]))

            tasks_h = h_U[a]


            for i_task, choice_set in enumerate(tasks_choice_sets):
                if i_task >= len(tasks_observed):
                    continue
                    
                sub_size = len(choice_set)
                h_sub = np.zeros((sub_size, sub_size), dtype=int)
                local_map = {item: idx for idx, item in enumerate(choice_set)}

                # Build the adjacency among only the chosen items
                for r, item_r in enumerate(choice_set):
                    if item_r in Ma:
                        local_r = Ma.index(item_r)
                        for c, item_c in enumerate(choice_set):
                            if item_c in Ma:
                                local_c = Ma.index(item_c)
                                if local_r < len(tasks_h) and local_c < len(tasks_h):
                                    h_sub[r, c] = tasks_h[local_r, local_c]

                y_i = tasks_observed[i_task]
                y_i_local = [local_map[item] for item in y_i if item in local_map]
                m = len(y_i_local)

                if noise_option == "queue_jump":
                    for j, y_j in enumerate(y_i_local):
                        remaining_indices = y_i_local[j:]
                        h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)]

                        num_le = cls._get_nle(h_remaining)  # parallel call
                        local_idx = remaining_indices.index(y_j)
                        num_first_item = cls._get_nle_first(h_remaining, local_idx)

                        prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)
                        prob_jump = prob_noise * (1.0 / (m - j))
                        log_likelihood += math.log(max(prob_no_jump + prob_jump, 1e-20))

                elif noise_option=="weighted_queue_jump":
                    # Non-vectorized: compute bar-eta per remaining item using Ua (invariants guaranteed)
                    for j, y_j in enumerate(y_i_local):
                        remaining_indices = y_i_local[j:]
                        h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)]

                        num_le = cls._get_nle(h_remaining)
                        local_idx = remaining_indices.index(y_j)
                        num_first_item = cls._get_nle_first(h_remaining, local_idx)
                        prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)

                        sum_w = 0.0
                        weight_map = {}
                        for local_id in remaining_indices:
                            item_name = choice_set[local_id]
                            # Compute bar-eta directly from Ua
                            local_idx_in_Ma = Ma.index(item_name)
                            ua_row = Ua[local_idx_in_Ma, :]
                            p_vec = norm.cdf(ua_row)
                            g_vec = StatisticalUtils.gumbel_inv_cdf(p_vec)
                            bar_eta_item = float(np.mean(g_vec))

                            w_val = math.exp(bar_eta_item)
                            weight_map[local_id] = w_val
                            sum_w += w_val

                        w_j = weight_map[y_j]
                        prob_jump_for_yj = prob_noise * (w_j / sum_w)

                        total_p = prob_no_jump + prob_jump_for_yj
                        log_likelihood += math.log(max(total_p, 1e-20))


        return log_likelihood
        
    @classmethod
    def calculate_single_list_likelihood(
        cls,
        list_idx,         # ID of the list/order to evaluate
        e_list,           # The observed order for this list/task
        cluster_id,       # ID of the cluster to consider for this list
        U0,               # Global latent positions
        U_a_dict,         # Assessor-specific latent positions
        observed_orders,  # All observed orders
        M_a_dict,         # Mapping from assessor to item indices
        O_a_i_dict,       # Choice sets for each assessor
        item_to_index,    # Map from item to index
        prob_noise,       # Noise probability
        softmax_params,     # Unused (reserved for future noise models)
        noise_option,     # Noise model type
        h_U         # Pre-computed hierarchical partial orders      
    ):

        if noise_option not in ["queue_jump", "weighted_queue_jump"]:
            raise ValueError(f"Invalid noise_option: {noise_option}.")
            
        # Get the observed order for this list
        y_i = e_list
        # If no order observed, return 0
        if not y_i:
            return 0.0
            
        # Get the choice set for this list (all elements in the observed order)
        choice_set = sorted(e_list)

        # If not provided, we need to compute it based on U0 and U_a
        Ma   = M_a_dict[cluster_id]
        U_a  = U_a_dict[cluster_id]
        h_a  = h_U[cluster_id]
    
        # Subset to only the items in the choice set
        sub_size = len(choice_set)
        h_sub = np.zeros((sub_size, sub_size), dtype=int)
        local_map = {item: idx for idx, item in enumerate(choice_set)}
        
        # Build the adjacency matrix for only the chosen items
        for r, item_r in enumerate(choice_set):
            if item_r in Ma:
                local_r = Ma.index(item_r)
                for c, item_c in enumerate(choice_set):
                    if item_c in Ma:
                        local_c = Ma.index(item_c)
                        if local_r < len(h_a) and local_c < len(h_a):
                            h_sub[r, c] = h_a[local_r, local_c]
        
        # Convert observed order to local indices 
        y_i_local = [local_map[item] for item in y_i if item in local_map]
        m = len(y_i_local)
        
        # Calculate log-likelihood based on noise model
        log_likelihood = 0.0
        
        if noise_option == "queue_jump":
            for j, y_j in enumerate(y_i_local):
                remaining_indices = y_i_local[j:]
                h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices) ]              
                
                num_le = cls._get_nle(h_remaining)
                local_idx = remaining_indices.index(y_j)
                num_first_item = cls._get_nle_first(h_remaining, local_idx)
                
                prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)
                prob_jump = prob_noise * (1.0 / (m - j))
                log_likelihood += math.log(max(prob_no_jump + prob_jump, 1e-20))
        elif noise_option=="weighted_queue_jump":
            # Non-vectorized: compute bar-eta per remaining item using Ua (invariants guaranteed)
            for j, y_j in enumerate(y_i_local):
                remaining_indices = y_i_local[j:]
                h_remaining = h_sub[np.ix_(remaining_indices, remaining_indices)]

                num_le = cls._get_nle(h_remaining)
                local_idx = remaining_indices.index(y_j)
                num_first_item = cls._get_nle_first(h_remaining, local_idx)
                prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)

                sum_w = 0.0
                weight_map = {}
                for local_id in remaining_indices:
                    item_name = choice_set[local_id]
                    # Compute bar-eta directly from Ua
                    local_idx_in_Ma = Ma.index(item_name)
                    ua_row = U_a[local_idx_in_Ma, :]
                    p_vec = norm.cdf(ua_row)
                    g_vec = StatisticalUtils.gumbel_inv_cdf(p_vec)
                    bar_eta_item = float(np.mean(g_vec))

                    w_val = math.exp(bar_eta_item)
                    weight_map[local_id] = w_val
                    sum_w += w_val

                w_j = weight_map[y_j]
                prob_jump_for_yj = prob_noise * (w_j / sum_w)

                total_p = prob_no_jump + prob_jump_for_yj
                log_likelihood += math.log(max(total_p, 1e-20))
        return log_likelihood
        
