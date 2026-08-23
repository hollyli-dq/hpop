import numpy as np
import math
import sys
from scipy.stats import multivariate_normal,norm
from scipy.special import betaln, gammaln
from tabulate import tabulate
from typing import List, Dict, Tuple, Union

import networkx as nx
import random
import seaborn as sns
import pandas as pd  
from collections import Counter
import itertools
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from collections import defaultdict
from math import log, inf
from scipy.stats import beta,expon
from scipy.stats import gamma 


_COV_CACHE: Dict[Tuple[int, float, float], Tuple[np.ndarray, float, float]] = {}


class ConversionUtils:
    """
    Utility class for converting sequences and orders to different representations.
    """

    @staticmethod
    def seq2dag(seq: List[int], n: int) -> np.ndarray:
        """
        Converts a sequence to a directed acyclic graph (DAG) represented as an adjacency matrix.

        Parameters:
        - seq: A sequence (list) of integers representing a total order.
        - n: Total number of elements.

        Returns:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the DAG.
        """
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(len(seq)):
            u = seq[i] - 1  # Convert to 0-based index
            for j in range(i + 1, len(seq)):
                v = seq[j] - 1  # Convert to 0-based index
                adj_matrix[u, v] = 1
        return adj_matrix

    @staticmethod
    def order2partial(v: List[List[int]], n: Optional[int] = None) -> np.ndarray:
        """
        Computes the intersection of the transitive closures of a list of total orders.

        Parameters:
        - v: List of sequences, where each sequence is a list of integers representing a total order.
        - n: Total number of elements (optional).

        Returns:
        - result_matrix: An n x n numpy array representing the adjacency matrix of the partial order.
        """
        if n is None:
            n = max(max(seq) for seq in v)
        z = np.zeros((n, n), dtype=int)
        for seq in v:
            dag_matrix = ConversionUtils.seq2dag(seq, n)
            closure_matrix = BasicUtils.transitive_closure(dag_matrix)
            z += closure_matrix
        result_matrix = (z == len(v)).astype(int)
        return result_matrix



    
class GenerationUtils:
    """
    Utility class for generating latent positions, partial orders, random partial orders, 
    linear extensions, total orders, and subsets.
    """
    @staticmethod
    def generate_choice_sets_for_assessors(
        M_a_dict: Dict[int, List[int]],
        min_tasks: int = 1,
        min_size: int = 2
    ) -> Dict[int, List[List[int]]]:
        """
        Generate a dictionary of choice sets O_{a,i} for each assessor.
        
        Each assessor a is assigned a random number of tasks (choice sets) between
        min_tasks and max_tasks. For each task, a random subset of items (of size at least
        min_size and at most the total number of items in M_a) is selected from the assessor's M_a.
        
        Parameters:
            M_a_dict : Dict[int, List[int]]
                Dictionary mapping assessor IDs to their overall list of item IDs.
            min_tasks : int, optional
                Minimum number of tasks per assessor (default is 1).
            max_tasks : int, optional
                Maximum number of tasks per assessor (default is 3).
            min_size : int, optional
                Minimum number of items in a choice set (default is 2).
        
        Returns:
            Dict[int, List[List[int]]]: Dictionary where each key is an assessor ID and each value is a list
                                        of choice sets (each choice set is a list of item IDs).
        """
        O_a_i_dict = {}
        for assessor, items in M_a_dict.items():
            num_items = len(items)
            max_tasks = 10*num_items
            
            # FIXED: Ensure max_tasks >= min_tasks to avoid ValueError
            if max_tasks < min_tasks:
                max_tasks = min_tasks  # Set max_tasks to at least min_tasks
                
            # Determine the number of tasks for this assessor.
            num_tasks = random.randint(min_tasks, max_tasks)
            tasks = []
            for _ in range(num_tasks):
                # Choose a task size: at least min_size, at most all available items.
                task_size = random.randint(min_size, num_items)
                task = sorted(random.sample(items, task_size))
                tasks.append(task)
            O_a_i_dict[assessor] = tasks
        return O_a_i_dict

    @staticmethod
    def generate_latent_positions(n: int, K: int, rho: float) -> np.ndarray:
        """
        Generates latent positions Z for n items in K dimensions with correlation rho.

        Parameters:
        - n: Number of items.
        - K: Number of dimensions.
        - rho: Correlation coefficient between dimensions.

        Returns:
        - Z: An n x K numpy array of latent positions.
        """
        Sigma = BasicUtils.build_Sigma_rho(K,rho)
        mu = np.zeros(K)
        rv = multivariate_normal(mean=mu, cov=Sigma)
        Z = rv.rvs(size=n)
        if K == 1:
            Z = Z.reshape(n, 1)
        return Z

    @staticmethod
    def generate_random_PO(n: int) -> nx.DiGraph:
        """
        Generates a random partial order (directed acyclic graph) with `n` nodes.
        Ensures there are no cycles in the generated graph.

        Parameters:
        - n: Number of nodes in the partial order.

        Returns:
        - h: A NetworkX DiGraph representing the partial order.
        """
        h = nx.DiGraph()
        h.add_nodes_from(range(n))
        possible_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(possible_edges)
        for u, v in possible_edges:
            if random.choice([True, False]):
                h.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(h):
                    h.remove_edge(u, v)
        return h
    @staticmethod
    def generate_U(n: int, K: int, rho_val: float) -> np.ndarray:
        """
        Generate a latent variable matrix U of size n x K from a multivariate normal distribution
        with zero mean and a covariance matrix based on the given correlation rho_val.

        Parameters:
        - n: Number of observations.
        - K: Number of features.
        - rho_val: Correlation value for constructing the covariance matrix.

        Returns:
        - U: An n x K numpy array of latent positions.
        """
        K=int(K)
        cov = BasicUtils.build_Sigma_rho(K, rho_val)
        mean = np.zeros(K)
        U = np.random.multivariate_normal(mean, cov, size=n)
        return U
    # Cache for Numba availability check
    _numba_unifLE = None
    _numba_checked = False
    
    @staticmethod
    def unifLE(tc: np.ndarray, elements: List[int], le: Optional[List[int]] = None) -> List[int]:
        """
        Sample a linear extension uniformly at random from the given partial order matrix `tc`.
        
        OPTIMIZED v3: Uses Numba JIT when available (20x speedup), falls back to vectorized NumPy.

        Parameters:
        - tc: Transitive closure matrix representing the partial order (numpy 2D array).
        - elements: List of elements corresponding to the current `tc` matrix.
        - le: List to build the linear extension (default: None).

        Returns:
        - le: A linear extension (list of elements in the original subset).
        """
        n = len(elements)
        if n == 0:
            return le if le is not None else []
        
        # Try Numba version (20x faster)
        if not GenerationUtils._numba_checked:
            try:
                from src.utils.numba_accelerated import _unifLE_numba, NUMBA_AVAILABLE
                if NUMBA_AVAILABLE:
                    GenerationUtils._numba_unifLE = _unifLE_numba
            except ImportError:
                pass
            GenerationUtils._numba_checked = True
        
        if GenerationUtils._numba_unifLE is not None:
            tc_bool = tc.astype(np.bool_)
            result_indices = GenerationUtils._numba_unifLE(tc_bool)
            elements_arr = np.array(elements)
            return [elements_arr[i] for i in result_indices]
        
        # Fallback: Pure Python vectorized version
        result = []
        active = np.ones(n, dtype=bool)
        elements_arr = np.array(elements)
        tc_bool = tc.astype(bool)
        
        for _ in range(n):
            indegrees = tc_bool[active, :].sum(axis=0)
            minimal_mask = active & (indegrees == 0)
            minimal_indices = np.where(minimal_mask)[0]
            
            if len(minimal_indices) == 0:
                if not active.any():
                    break
                raise ValueError("No minimal elements found. The partial order might contain cycles.")
            
            chosen_idx = random.choice(minimal_indices)
            result.append(elements_arr[chosen_idx])
            active[chosen_idx] = False
        
        return result

    @staticmethod
    def sample_total_order(h: np.ndarray, subset: List[int]) -> List[int]:
        """
        Sample a total order (linear extension) for a restricted partial order.

        Parameters:
        - h: The original partial order adjacency matrix.
        - subset: List of node indices to sample a linear extension for.

        Returns:
        - sampled_order: A list representing the sampled linear extension.
        """
        # Restrict the matrix to the given subset
        restricted_matrix = BasicUtils.restrict_partial_order(h, subset)

        # Initialize elements as the elements in the subset
        elements = subset.copy()
        restricted_matrix_tc = BasicUtils.transitive_closure(restricted_matrix)

        # Sample one linear extension using the `unifLE` function
        sampled_order = GenerationUtils.unifLE(restricted_matrix_tc, elements)

        return sampled_order

    @staticmethod
    def sample_linear_extension(h: np.ndarray) -> List[int]:
        """
        Sample a linear extension uniformly at random from partial order h.
        
        Args:
            h: Partial order adjacency matrix (n x n)
            
        Returns:
            List of indices representing a linear extension
        """
        n = h.shape[0]
        if n == 0:
            return []
        
        elements = list(range(n))
        tc = BasicUtils.transitive_closure(h)
        return GenerationUtils.unifLE(tc, elements)
    
    @staticmethod
    def enumerate_linear_extensions(h: np.ndarray, max_extensions: int = 1000) -> List[List[int]]:
        """
        Enumerate all linear extensions of a partial order.
        
        Uses recursive enumeration over minimal elements.
        For efficiency, limits to max_extensions.
        
        Args:
            h: Partial order adjacency matrix (n x n)
            max_extensions: Maximum number of extensions to enumerate
            
        Returns:
            List of linear extensions, each as a list of indices
        """
        n = h.shape[0]
        if n == 0:
            return [[]]
        if n == 1:
            return [[0]]
        
        tc = BasicUtils.transitive_closure(h)
        elements = list(range(n))
        
        results = []
        GenerationUtils._enumerate_le_recursive(tc, elements, [], results, max_extensions)
        return results
    
    @staticmethod
    def _enumerate_le_recursive(tc: np.ndarray, elements: List[int], 
                                  current_le: List[int], results: List[List[int]], 
                                  max_extensions: int) -> None:
        """Recursive helper for enumerate_linear_extensions."""
        if len(results) >= max_extensions:
            return
        
        if len(elements) == 0:
            results.append(current_le.copy())
            return
        
        # Find minimal elements (no incoming edges)
        indegrees = np.sum(tc, axis=0)
        minimal_indices = np.where(indegrees == 0)[0]
        
        if len(minimal_indices) == 0:
            return  # Cycle detected
        
        # For each minimal element, recurse
        for idx_in_tc in minimal_indices:
            if len(results) >= max_extensions:
                return
            
            element = elements[idx_in_tc]
            
            # Remove this element and recurse
            tc_new = np.delete(np.delete(tc, idx_in_tc, axis=0), idx_in_tc, axis=1)
            elements_new = [e for i, e in enumerate(elements) if i != idx_in_tc]
            
            current_le.append(element)
            GenerationUtils._enumerate_le_recursive(tc_new, elements_new, current_le, results, max_extensions)
            current_le.pop()

    @staticmethod
    def topological_sort(adj_matrix: np.ndarray):
        """
        Returns one valid topological ordering of nodes in a DAG
        represented by an adjacency matrix.

        Parameters:
        - adj_matrix: n x n adjacency matrix (0/1),
                    where edge i->j means adj_matrix[i, j] == 1.

        Returns:
        - ordering: A list of node indices in topological order.
        
        Raises:
        - ValueError if the graph has a cycle or is not a DAG.
        """
        n = adj_matrix.shape[0]
        # in_degree[i] = number of incoming edges for node i
        in_degree = np.sum(adj_matrix, axis=0)

        # start with nodes that have no incoming edges
        queue = [i for i in range(n) if in_degree[i] == 0]
        ordering = []

        while queue:
            node = queue.pop()
            ordering.append(node)

            # "Remove" node from the graph => 
            # For each edge node->v, reduce in_degree[v] by 1
            for v in range(n):
                if adj_matrix[node, v] == 1:
                    in_degree[v] -= 1
                    # If v becomes a node with no incoming edges => add to queue
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(ordering) != n:
            # A cycle must exist, or something prevented us from ordering all nodes
            raise ValueError("The adjacency matrix contains a cycle (not a DAG).")

        return ordering

    @staticmethod
    def generate_subsets(N: int, n: int) -> List[List[int]]:
        """
        Generate N subsets O1, O2, ..., ON where:
        - N is the number of subsets.
        - n is the size of the universal set {0, 1, ..., n-1}.
        
        Each subset Oi is created by:
        - Determining the subset size ni by uniformly sampling from [2, n].
        - Randomly selecting ni distinct elements from the set {0, 1, ..., n-1}.

        Parameters:
        - N: Number of subsets to generate.
        - n: Size of the universal set.

        Returns:
        - subsets: A list of subsets, each subset is a list of distinct integers.
        """
        subsets = []
        universal_set = list(range(n))  # Universal set from 0 to n-1

        for _ in range(N):
            # Randomly sample the subset size ni from [2, n]
            ni = random.randint(2, n)
            # Randomly select ni distinct elements from the universal set
            subset = random.sample(universal_set, ni)
            subset =sorted(subset)
            subsets.append(subset)

        return subsets
    @staticmethod
    def generate_total_orders_for_assessor(
        h_dict: Dict[int, np.ndarray],
        M_a_dict: Dict[int, List[int]],
        O_a_i_dict: Dict[int, List[List[int]]],
        prob_noise: float
    ) -> Dict[int, List[List[int]]]:
        """
        For each assessor, generate total orders (linear extensions) from their local partial order.
        
        Parameters:
        h_dict: Dictionary mapping assessor IDs to local partial order matrices (each of shape (|Mₐ|,|Mₐ|)).
        M_a_dict: Dictionary mapping assessor IDs to their ordered list of global item IDs.
                The order corresponds to the rows/columns in the local partial order matrix.
        O_a_i_dict: Dictionary mapping assessor IDs to a list of choice sets.
                    Each choice set is a list of global item IDs.
        prob_noise: The noise (jump) probability.
        
        Returns:
        Dict[int, List[List[int]]]: Mapping from assessor IDs to a list of total orders.
                                    Each total order is expressed as a list of global item IDs.
        """
        total_orders_dict = {}
        
        for a, choice_sets in O_a_i_dict.items():
            # Retrieve local partial order matrix.
            h_local = h_dict.get(a)
            if h_local is None:
                print(f"Warning: No partial order matrix found for assessor {a}. Skipping.")
                continue
            # Retrieve assessor's ordered global items.
            M_a = M_a_dict.get(a)
            if M_a is None:
                print(f"Warning: No item set found for assessor {a}. Skipping.")
                continue
            
            assessor_orders = []
            for subset in choice_sets:
                # Generate total order for this choice set.
                total_order = StatisticalUtils.generate_total_order_for_choice_set_with_queue_jump(subset, M_a, h_local, prob_noise)
                if total_order:
                    assessor_orders.append(total_order)
            total_orders_dict[a] = assessor_orders
        
        return total_orders_dict
    @staticmethod
    def generate_cluster_data(
        n_items: int = 10,
        n_lists: int = 20,
        K_true: int = 3,
        rho_true: float = 0.8,
        tau_true: float = 0.7,
        Discount_d:float = 0.0, 
        Concentration_theta: float = 1.0,
        prob_noise_true: float = 0.1,
        min_tasks_per_list: int = 3,
        max_tasks_per_list: int = 5,
        min_choice_set_size: int = 2,
        
        rng=None
    ) -> Dict[str, Any]:
        """
        Generate synthetic data for cluster-based HPO model using utility functions.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        
        print(f"🎯 Generating cluster-based HPO data...")
        print(f"   Items: {n_items}, Lists: {n_lists}, Clusters: CRP(α={Concentration_theta})")
        print(f"   Latent dim: {K_true}, ρ={rho_true}, τ={tau_true}")

        # 1. Generate items and global structure
        M0 = list(range(n_items))
        
        # 2. Generate cluster assignments using CRP
        c_vec, na_dict = StatisticalUtils.generate_cluster_assignments_pitman_yor(
            n_lists=n_lists,
            discount_d= Discount_d,         # <-- discount 'd'
            concentration_theta=Concentration_theta, # <-- concentration 'theta'
            rng=rng,
            start_label=1,
        )
        cluster_ids = sorted(na_dict.keys())
        
        print(f"   Generated {len(cluster_ids)} clusters: {na_dict}")
        
        # 3. Generate global latent utilities (U0)
        U0_true = GenerationUtils.generate_U(n_items, K_true, rho_true)
        
        # 4. Generate cluster-specific latent utilities and item assignments
        U_a_dict_true = {}
        M_a_dict = {}
        
        for k in cluster_ids:
            # Each cluster works with a subset of items
            n_items_in_cluster = rng.integers(max(2, n_items//2), n_items + 1)
            cluster_items = rng.choice(M0, size=n_items_in_cluster, replace=False)
            M_a_dict[k] = sorted(cluster_items)
            
            # Generate cluster-specific latent utilities
            n_a = len(cluster_items)
            Ua = np.zeros((n_a, K_true), dtype=float)
            
            for i_loc, item_id in enumerate(cluster_items):
                mean_vec = tau_true * U0_true[item_id, :]
                cov_mat = (1.0 - tau_true**2) * BasicUtils.build_Sigma_rho(K_true, rho_true)
                Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_mat)
            
            U_a_dict_true[k] = Ua
        
        # 5. Build hierarchical partial orders
        h_U_true = StatisticalUtils.build_hierarchical_partial_orders(
            M0=M0,
            assessors=cluster_ids,
            M_a_dict=M_a_dict,
            U0=U0_true,
            U_a_dict=U_a_dict_true
        )
        
        # 6. Generate data for each LIST individually (correct approach)
        O_a_i_dict = []  # List of choice sets for each list
        observed_orders = []  # List of orders for each list
        
        for list_idx in range(n_lists):
            cluster_id = c_vec[list_idx]
            cluster_items = M_a_dict[cluster_id]
            h_cluster = h_U_true[cluster_id]
            
            # Generate multiple tasks for this list
            num_tasks = rng.integers(min_tasks_per_list, max_tasks_per_list + 1)
            
            # For this list, generate one choice set and one order
            # (Each list contributes exactly one observation to the MCMC)
            choice_set_size = rng.integers(min_choice_set_size, min(len(cluster_items) + 1, 6))
            choice_set = sorted(rng.choice(cluster_items, size=choice_set_size, replace=False))
            
            # Generate total order for this choice set
            total_order = StatisticalUtils.generate_total_order_for_choice_set_with_queue_jump(
                subset=choice_set,
                M_a=cluster_items,
                h_local=h_cluster,
                prob_noise=prob_noise_true
            )
            
            # Store the single choice set and order for this list
            O_a_i_dict.append(choice_set)
            observed_orders.append(total_order if total_order else choice_set)
        
        # 7. Create cluster-grouped data for analysis (derived from list data)
        observed_orders_cl = {}
        O_a_i_dict_cl = {}
        
        for k in cluster_ids:
            observed_orders_cl[k] = []
            O_a_i_dict_cl[k] = []
        
        # Group the list data by cluster
        for list_idx in range(n_lists):
            cluster_id = c_vec[list_idx]
            observed_orders_cl[cluster_id].append(observed_orders[list_idx])
            O_a_i_dict_cl[cluster_id].append(O_a_i_dict[list_idx])
        
        # 8. Validation
        assert len(O_a_i_dict) == n_lists, f"Expected {n_lists} choice sets, got {len(O_a_i_dict)}"
        assert len(observed_orders) == n_lists, f"Expected {n_lists} orders, got {len(observed_orders)}"
        assert len(c_vec) == n_lists, f"Expected {n_lists} cluster assignments, got {len(c_vec)}"
        
        # Verify cluster grouping
        total_grouped = sum(len(observed_orders_cl[k]) for k in cluster_ids)
        assert total_grouped == n_lists, f"Cluster grouping error: {total_grouped} != {n_lists}"
        
        return {
            # True parameters
            'true_params': {
                'K': K_true,
                'rho': rho_true,
                'tau': tau_true,
                'prob_noise': prob_noise_true,
                'D_discount': Discount_d,
                'Concentration_theta':Concentration_theta 
            },
            # Data structures
            'M0': M0,
            'c_vec': c_vec,
            'cluster_ids': cluster_ids,
            'na_dict': na_dict,
            'M_a_dict': M_a_dict,
            'U0_true': U0_true,
            'U_a_dict_true': U_a_dict_true,
            'h_U_true': h_U_true,
            # MCMC input data (correctly structured)
            'O_a_i_dict': O_a_i_dict,
            'observed_orders': observed_orders,
            'observed_orders_cl': observed_orders_cl,
            'O_a_i_dict_cl': O_a_i_dict_cl
        }

    @staticmethod
    def generate_assessor_cluster_data(
        n_items: int = 10,
        n_lists: int = 20,              # used here as the number of ASSESSORS (kept name for backward-compat)
        K_true: int = 3,
        rho_true: float = 0.8,
        tau_true: float = 0.7,
        Discount_d: float = 0.0,        # Pitman–Yor discount d ∈ [0,1)
        Concentration_theta: float = 1.0,  # Pitman–Yor concentration θ > -d
        prob_noise_true: float = 0.1,
        min_tasks_per_list: int = 3,    # reinterpreted as MIN lists per assessor
        max_tasks_per_list: int = 5,    # reinterpreted as MAX lists per assessor
        min_choice_set_size: int = 2,
        rng=None
    ) -> Dict[str, Any]:
        """
        Generate synthetic data for an assessor-cluster HPO model using a Pitman–Yor prior.
        - We treat `n_lists` as the NUMBER OF ASSESSORS for compatibility with existing code.
        - Each assessor produces a random number of lists in [min_tasks_per_list, max_tasks_per_list].
        - Cluster assignments are sampled at the assessor level (not list level).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_assessors = n_lists  # keep param name but use assessor semantics
        # -------------------------
        # 1) Items
        # -------------------------
        M0 = list(range(n_items))

        # -------------------------
        # 2) Assessor-level clusters via Pitman–Yor(d, θ)
        # -------------------------
        c_vec_assessors, na_dict = StatisticalUtils.generate_cluster_assignments_pitman_yor(
            n_lists=n_assessors,
            discount_d=Discount_d,
            concentration_theta=Concentration_theta,
            rng=rng,
            start_label=1,
        )
        cluster_ids = sorted(na_dict.keys())
        assessors = list(range(n_assessors))
        assessor_to_cluster = {a: c_vec_assessors[a] for a in assessors}
        cluster_to_assessors: Dict[int, list] = {k: [] for k in cluster_ids}
        for a in assessors:
            cluster_to_assessors[assessor_to_cluster[a]].append(a)

        print(f"   Generated {len(cluster_ids)} clusters: {na_dict}")

        # -------------------------
        # 3) Global latent utilities U0
        # -------------------------
        Sigma_rho_true = BasicUtils.build_Sigma_rho(K_true, rho_true)
        U0_true = rng.multivariate_normal(mean=np.zeros(K_true), cov=Sigma_rho_true, size=n_items)

        # -------------------------
        # 4) Cluster-specific item sets and Ua
        # -------------------------
        U_a_dict_true: Dict[int, np.ndarray] = {}
        M_a_dict: Dict[int, List[int]] = {}

        for k in cluster_ids:
            # choose item subset for cluster k
            n_items_in_cluster = int(rng.integers(max(2, n_items // 2), n_items + 1))
            cluster_items = sorted(rng.choice(M0, size=n_items_in_cluster, replace=False))
            M_a_dict[k] = cluster_items

            # sample Ua rows ~ N(τ U0[j], (1-τ^2) Σ_ρ)
            n_a = len(cluster_items)
            Ua = np.zeros((n_a, K_true), dtype=float)
            cov_k = (1.0 - tau_true**2) * Sigma_rho_true + 1e-8 * np.eye(K_true)
            for i_loc, j_global in enumerate(cluster_items):
                mean_vec = tau_true * U0_true[j_global, :]
                Ua[i_loc, :] = rng.multivariate_normal(mean=mean_vec, cov=cov_k)
            U_a_dict_true[k] = Ua

        # -------------------------
        # 5) Build hierarchical partial orders h_U
        # -------------------------
        h_U_true = StatisticalUtils.build_hierarchical_partial_orders(
            M0=M0,
            assessors=cluster_ids,   # here "assessors" means cluster IDs in that utility
            M_a_dict=M_a_dict,
            U0=U0_true,
            U_a_dict=U_a_dict_true
        )

        # -------------------------
        # 6) Generate per-assessor lists (choice sets & orders)
        # -------------------------
        assessor_data: Dict[int, List[List[int]]] = {}         # assessor_id -> list of observed orders
        assessor_choice_sets: Dict[int, List[List[int]]] = {}  # assessor_id -> list of choice sets

        total_lists = 0
        for a in assessors:
            k = assessor_to_cluster[a]
            cluster_items = M_a_dict[k]
            h_cluster = h_U_true[k]

            # how many lists this assessor provides
            n_lists_a = int(rng.integers(min_tasks_per_list, max_tasks_per_list + 1))
            orders_a: List[List[int]] = []
            choices_a: List[List[int]] = []

            for _ in range(n_lists_a):
                max_size_allowed = min(len(cluster_items), 6)  # keep choice set small-ish
                if max_size_allowed < max(min_choice_set_size, 2):
                    # if the cluster is tiny, fall back to all its items
                    choice_set = sorted(cluster_items)
                else:
                    size = int(rng.integers(min_choice_set_size, max_size_allowed + 1))
                    choice_set = sorted(rng.choice(cluster_items, size=size, replace=False))

                # draw a noisy order on the choice set
                order = StatisticalUtils.generate_total_order_for_choice_set_with_queue_jump(
                    subset=choice_set,
                    M_a=cluster_items,
                    h_local=h_cluster,
                    prob_noise=prob_noise_true
                )
                if not order:  # safety
                    order = choice_set

                orders_a.append(order)
                choices_a.append(choice_set)

            assessor_data[a] = orders_a
            assessor_choice_sets[a] = choices_a
            total_lists += n_lists_a

        # -------------------------
        # 7) Build cluster-grouped containers from assessor data
        # -------------------------
        observed_orders_cl: Dict[int, List[List[int]]] = {k: [] for k in cluster_ids}
        O_a_i_dict_cl: Dict[int, List[List[int]]] = {k: [] for k in cluster_ids}
        for a in assessors:
            k = assessor_to_cluster[a]
            observed_orders_cl[k].extend(assessor_data[a])
            O_a_i_dict_cl[k].extend(assessor_choice_sets[a])

        # also provide flattened (list-level) sequences if needed elsewhere
        O_a_i_dict_flat: List[List[int]] = []
        observed_orders_flat: List[List[int]] = []
        for a in assessors:
            O_a_i_dict_flat.extend(assessor_choice_sets[a])
            observed_orders_flat.extend(assessor_data[a])

        # -------------------------
        # 8) Validations
        # -------------------------
        assert len(assessors) == n_assessors
        assert sum(len(v) for v in assessor_data.values()) == total_lists
        assert sum(len(v) for v in assessor_choice_sets.values()) == total_lists
        for k in cluster_ids:
            assert len(observed_orders_cl[k]) == len(O_a_i_dict_cl[k])


        return {
            # true params
            'true_params': {
                'K': K_true,
                'rho': rho_true,
                'tau': tau_true,
                'prob_noise': prob_noise_true,
                'discount_d': Discount_d,
                'concentration_theta': Concentration_theta,
            },
            # universe
            'M0': M0,
            # assessor clustering truth
            'assessors': assessors,
            'assessor_to_cluster': assessor_to_cluster,
            'cluster_to_assessors': cluster_to_assessors,
            'cluster_ids': cluster_ids,
            'na_dict': na_dict,  # sizes per cluster
            # latent truths
            'U0_true': U0_true,
            'U_a_dict_true': U_a_dict_true,
            'M_a_dict': M_a_dict,
            'h_U_true': h_U_true,
            # assessor-indexed observed data (for assessor-clustering MCMC)
            'assessor_data': assessor_data,
            'assessor_choice_sets': assessor_choice_sets,
            # cluster-indexed (useful for diagnostics)
            'observed_orders_cl': observed_orders_cl,
            'O_a_i_dict_cl': O_a_i_dict_cl,
            # flattened list-level (optional)
            'O_a_i_dict': O_a_i_dict_flat,
            'observed_orders': observed_orders_flat,
        }


class BasicUtils:
    """
    Utility class for basic operations on partial orders.
    """    
    @staticmethod
    def apply_transitive_reduction_hpo(h_U: dict) -> None:
        """
        For each key in h_U, if the value is a NumPy array, replace it with its transitive closure.
        If the value is a dictionary (e.g. assessor-level partial orders by task), then apply the
        operation to each matrix in that dictionary.
        
        This function modifies h_U in place.
        OPTIMIZED: Uses fast transitive reduction algorithm.
        """
        for key, value in h_U.items():
            if isinstance(value, dict):
                # If value is a dictionary, iterate over its keys
                for subkey, subval in value.items():
                    if isinstance(subval, np.ndarray):
                        value[subkey] = BasicUtils.transitive_reduction_optimized(subval)
            elif isinstance(value, np.ndarray):
                h_U[key] = BasicUtils.transitive_reduction_optimized(value)

    @staticmethod
    def build_Sigma_rho(K, rho_val: float) -> np.ndarray:
        mat = np.full((K, K), rho_val, dtype=float)
        np.fill_diagonal(mat, 1.0)
        return mat
    @staticmethod
    def generate_partial_order(Z):
        """
        Vectorised partial‑order generator.
        h[i,j] = 1  ⇔  Z[i] is strictly greater than Z[j] in **all** K dimensions.
        """
        # Z has shape (n, K)
        # Expand to (n, 1, K) and (1, n, K) for broadcasting
        greater = (Z[:, None, :] > Z[None, :, :])        # shape (n, n, K)
        h = np.all(greater, axis=2).astype(np.int8)       # collapse K‑axis
        np.fill_diagonal(h, 0)                            # ensure h[i,i] = 0
        return h
    
    @staticmethod
    def is_total_order(adj_matrix: np.ndarray) -> bool:

        n = adj_matrix.shape[0]
        # Compute transitive closure
        closure = BasicUtils.transitive_closure(adj_matrix)

        # For every pair (i,j), i != j, check comparability
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Must have either i->j or j->i
                    if not (closure[i, j] or closure[j, i]):
                        return False
    
        return True

    @staticmethod
    def restrict_partial_order(h: np.ndarray, subset: List[int]) -> np.ndarray:
        """
        Restrict the partial order matrix `h` to the given `subset`.

        Parameters:
        - h: The original partial order adjacency matrix.
        - subset: List of node indices to restrict to.

        Returns:
        - restricted_matrix: The adjacency matrix restricted to the subset.
        """
        subset_indices = subset  # Elements are already 0-based indices
        restricted_matrix = h[np.ix_(subset_indices, subset_indices)]
        return restricted_matrix

    @staticmethod
    def transitive_reduction(adj_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the transitive reduction of a partial order represented by an transitive closure matrix.
        transitive reduction need to be computed based on the transitive closure matrix.
        Parameters:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the partial order.

        Returns:
        - tr: An n x n numpy array representing the adjacency matrix of the transitive reduction.
        """
        n = adj_matrix.shape[0]
        tr = adj_matrix.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if tr[i, k] and tr[k, j]:
                        tr[i, j] = 0
        return tr


    # ------------------------------------------------------------------ #
    @staticmethod
    def transitive_reduction_optimized(C):
        C = C.astype(bool, copy=False)
        n  = C.shape[0]
        if n <= 1:
            return C.astype(np.int8)

        # remove diagonal so a vertex cannot be its own intermediate
        B = C.copy()
        np.fill_diagonal(B, False)

        # Boolean matrix product — path of length ≥2
        P = (B.astype(np.int8) @ B.astype(np.int8)) > 0

        # keep only indispensable edges
        red = C & ~P
        np.fill_diagonal(red, False)
        return red.astype(np.int8)



    @staticmethod
    def transitive_closure(adj_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the transitive closure of a relation represented by an adjacency matrix.
        
        OPTIMIZED: Uses vectorized NumPy operations instead of O(n³) Python loops.
        Achieves 10-100x speedup over the original implementation.

        Parameters:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the relation.

        Returns:
        - closure: An n x n numpy array representing the adjacency matrix of the transitive closure.
        """
        n = adj_matrix.shape[0]
        if n == 0:
            return adj_matrix.copy()
        
        # Convert to boolean for faster operations
        closure = adj_matrix.astype(bool)
        
        # Vectorized Floyd-Warshall: closure |= (closure[:, k:k+1] & closure[k:k+1, :])
        for k in range(n):
            # Outer product: which pairs (i,j) can reach through k?
            closure |= (closure[:, k:k+1] & closure[k:k+1, :])
        
        return closure.astype(adj_matrix.dtype)

    @staticmethod
    def nle(tr: np.ndarray) -> int:
        if tr.size == 0 or len(tr) == 1:
            return 1

        n = tr.shape[0]
        cs = np.sum(tr, axis=0)
        csi = (cs == 0)
        bs = np.sum(tr, axis=1)
        bsi = (bs == 0)
        free = np.where(bsi & csi)[0]
        k = len(free)

        if k == n:
            return math.factorial(n)

        if k > 0:
            # Delete free rows and columns
            tr = np.delete(np.delete(tr, free, axis=0), free, axis=1)
            fac = math.factorial(n) // math.factorial(n - k)
        else:
            fac = 1

        # Recompute cs and csi based on the updated tr
        cs = np.sum(tr, axis=0)
        csi = (cs == 0)
        bs = np.sum(tr, axis=1)
        bsi = (bs == 0)
        tops = np.where(csi)[0]
        bots = np.where(bsi)[0]

        # Special case: if n - k == 2, return fac
        if (n - k) == 2:
            return fac

        # Check for a unique top and bottom
        if len(tops) == 1 and len(bots) == 1:
            i = tops[0]
            j = bots[0]
            if i < tr.shape[0] and j < tr.shape[1]:
                trr = np.delete(np.delete(tr, [i, j], axis=0), [i, j], axis=1)
                return fac * BasicUtils.nle(trr)
            else:
                return 0  # Or handle appropriately

        # Iterate over all top elements
        count = 0
        for i in tops:
            if i >= tr.shape[0]:
                continue
            trr = np.delete(np.delete(tr, i, axis=0), i, axis=1)
            count += BasicUtils.nle(trr)

        return fac * count
    
    @staticmethod
    def compute_missing_relationships(
        h_true: np.ndarray,
        h_final: np.ndarray,
        index_to_item: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Compute the missing relationships present in the true partial order but absent in the inferred one.

        Parameters
        ----------
        h_true : np.ndarray
            Adjacency matrix representing the true partial order.
        h_final : np.ndarray
            Adjacency matrix representing the inferred partial order.
        index_to_item : Dict[int, int]
            Mapping from matrix indices to items.

        Returns
        -------
        List[Tuple[int, int]]
            List of tuples (i, j) indicating missing relationships.
        """

        missing = []
        n = h_true.shape[0]
        h_true_reduced = BasicUtils.transitive_reduction_optimized(h_true)
        h_final_reduced = BasicUtils.transitive_reduction_optimized(h_final)
        for i in range(n):
            for j in range(n):
                if h_true_reduced[i, j] == 1 and h_final_reduced[i, j] == 0:
                    missing.append((index_to_item[i], index_to_item[j]))
        return missing



    @staticmethod
    def compute_redundant_relationships(
        h_true: np.ndarray,
        h_final: np.ndarray,
        index_to_item: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Compute the redundant relationships present in the inferred partial order but absent in the true one.

        Parameters
        ----------
        h_true : np.ndarray
            Adjacency matrix representing the true partial order.
        h_final : np.ndarray
            Adjacency matrix representing the inferred partial order.
        index_to_item : Dict[int, int]
            Mapping from matrix indices to items.

        Returns
        -------
        List[Tuple[int, int]]
            List of tuples (i, j) indicating redundant relationships.
        """
        redundant = []
        n = h_true.shape[0]
        for i in range(n):
            for j in range(n):
                if h_true[i, j] == 0 and h_final[i, j] == 1:
                    redundant.append((index_to_item[i], index_to_item[j]))
        return redundant



    @staticmethod
    def find_tops(tr):
        """
        Identify all top elements (nodes with no incoming edges) in the partial order.

        Parameters:
        - tr: Adjacency matrix of the partial order (numpy.ndarray).

        Returns:
        - List of indices representing top elements.
        """
        incoming = np.sum(tr, axis=0)
        tops = [i for i, count in enumerate(incoming) if count == 0]
        return tops
    
    @staticmethod
    def is_transitive_closure(matrix: np.ndarray) -> bool:
        """
        Check if a matrix is already a transitive closure.
        Returns True if the matrix doesn't need transitive closure computation.
        
        This is much faster than blindly applying transitive_closure() everywhere.
        """
        if matrix.size == 0:
            return True
            
        n = matrix.shape[0]
        if n <= 1:
            return True
            
        # Quick check: if any (i,j)=1 and (j,k)=1 but (i,k)=0, it's not transitive
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 1:
                    for k in range(n):
                        if matrix[j, k] == 1 and matrix[i, k] == 0:
                            return False
        return True
    
    @staticmethod
    def ensure_transitive_closure(matrix: np.ndarray) -> np.ndarray:
        """
        Ensure matrix is a transitive closure, but only compute if needed.
        Much more efficient than always calling transitive_closure().
        """
        if BasicUtils.is_transitive_closure(matrix):
            return matrix  # Already a closure, no computation needed!
        raise ValueError("Input matrix is not a transitive closure; aborting.")

    @staticmethod
    def num_extensions_with_first(tr, first_item_idx):
        """
        Compute how many linear extensions of the partial order `tr` start with the item `first_item_idx`.

        Parameters:
        - tr: Adjacency matrix of the partial order (numpy.ndarray).
        - first_item_idx: The index of the item we want to be the first in the linear extension.

        Returns:
        - int: The number of linear extensions of `tr` that start with `first_item_idx`.
        """
        # Identify top elements of the current poset
        tops = BasicUtils.find_tops(tr)
        
        # If first_item_idx is not a top element, no linear extension can start with it.
        if first_item_idx not in tops:
            return 0

        # If it is top, remove it from tr and count the nle of the reduced poset.
        tr_reduced = np.delete(np.delete(tr, first_item_idx, axis=0), first_item_idx, axis=1)
        return BasicUtils.nle(tr_reduced)

    @staticmethod
    def is_consistent(h: np.ndarray, observed_orders: List[List[int]]) -> bool:
        """
        Check if all observed orders are consistent with the partial order h.

        Parameters:
        - h: The partial order matrix (NumPy array).
        - observed_orders: List of observed total orders (each is a list of item indices).

        Returns:
        - True if all observed orders are consistent with h, False otherwise.
        """
        # Create a directed graph from the partial order matrix h
        G_PO = nx.DiGraph(h)
        # Compute the transitive closure to capture all implied precedence relations
        tc_PO = BasicUtils.transitive_closure(h)

        # Iterate over each observed order
        for idx, order in enumerate(observed_orders):
            # Create a mapping from item to its position in the observed order
            position = {item: pos for pos, item in enumerate(order)}

            # Check all edges in the transitive closure
            for u, v in zip(*np.where(tc_PO == 1)):
                # Check if both u and v are in the observed order
                if u in position and v in position:
                    # If u comes after v in the observed order, it's a conflict
                    if position[u] > position[v]:
                        return False  # Inconsistency found

        return True



    @staticmethod
    def generate_all_linear_extensions(h: np.ndarray, items: Optional[List[Any]] = None) -> List[List[Any]]:
        """
        Generate all linear extensions (i.e. valid total orders) of a partial order
        represented by the adjacency matrix h. Here, h is an n x n matrix where h[i, j] == 1
        means that index i must precede index j. The items are by default the indices [0,1,...,n-1],
        but if a list 'items' is provided, it will be used to map indices to actual items.

        Parameters:
            h: n x n numpy array representing the partial order.
            items: Optional list of items corresponding to the indices of h.
                If None, items are assumed to be [0, 1, ..., n-1].

        Returns:
            A list of linear extensions, each represented as a list of items (or indices if items is None).
        """
        n = h.shape[0]
        if items is None:
            items = list(range(n))        
        def _recursive_extensions(h_sub: np.ndarray, remaining: List[int]) -> List[List[int]]:
            # Base case: if no elements remain, return an empty extension.
            if not remaining:
                return [[]]
            
            m = len(remaining)
            # Compute in-degrees for the current submatrix.
            in_degree = [0] * m
            for i in range(m):
                for j in range(m):
                    if h_sub[i, j]:
                        in_degree[j] += 1
            
            # Minimal elements are those with in-degree zero.
            minimal_indices = [i for i, d in enumerate(in_degree) if d == 0]
            
            extensions = []
            for idx in minimal_indices:
                # 'current' is the actual index from the original set.
                current = remaining[idx]
                # Remove the minimal element from the remaining list.
                new_remaining = remaining[:idx] + remaining[idx+1:]
                # Remove the corresponding row and column from the matrix.
                new_h = np.delete(np.delete(h_sub, idx, axis=0), idx, axis=1)
                # Recursively generate extensions for the reduced poset.
                for ext in _recursive_extensions(new_h, new_remaining):
                    extensions.append([current] + ext)
            return extensions

        # Start the recursion with all indices [0, 1, ..., n-1].
        index_extensions = _recursive_extensions(h, list(range(n)))
        # Map the indices to actual items if provided.
        extensions = [[items[i] for i in extension] for extension in index_extensions]
        return extensions

    @staticmethod
    def intersection_of_extensions(items, order_list):
        """
        Given a list of linear orders (each order is a permutation of [0, 1, ..., n-1]),
        return the intersection adjacency representation.
        
        For any distinct pair x and y, add an edge x -> y if and only if x precedes y in every order.
        
            order_list: A list of linear orders (each order is a tuple or list of integers).
        
        Returns:
            A dictionary where each key is an item and the value is a set of items that follow it
            in every linear extension.
        """
        # Define the items and a mapping (though items are already indices here)
 
        # Create a position map for each order:
        # For each order, pos_map[x] gives the position of x in that order.
        pos_maps = [{x: pos for pos, x in enumerate(order)} for order in order_list]
        
        # Initialize the intersection adjacency dictionary.
        intersection_adj = {x: set() for x in items}
        
        # For every distinct pair (x, y), check if x precedes y in every order.
        for x in items:
            for y in items:
                if x != y and all(pos_map[x] < pos_map[y] for pos_map in pos_maps):
                    intersection_adj[x].add(y)
        
        return intersection_adj


class StatisticalUtils:
    """
    Utility class for statistical computations related to partial orders.
    """

    @staticmethod
    def count_unique_partial_orders(h_trace):
        """
        Count the frequency of each unique partial order in h_trace.
        
        Parameters:
        - h_trace: List of NumPy arrays representing partial orders.
        
        Returns:
        - Dictionary with partial order representations as keys and their counts as values.
        """
        unique_orders = defaultdict(int)
        
        for h_Z in h_trace:
            # Convert the matrix to a tuple of tuples for immutability
            h_tuple = tuple(map(tuple, h_Z))
            unique_orders[h_tuple] += 1
    

        sorted_unique_orders = sorted(unique_orders.items(), key=lambda x: x[1], reverse=True)
        
        # Convert the sorted tuples back to NumPy arrays for readability
        sorted_unique_orders = [(np.array(order), count) for order, count in sorted_unique_orders]
        return sorted_unique_orders

    @staticmethod
    def generate_cluster_assignments_pitman_yor(
        n_lists: int,
        discount_d: float,
        concentration_theta: float,
        rng: Optional[np.random.Generator] = None,
        start_label: int = 1,
    ) -> Tuple[List[int], Dict[int, int]]:

        if rng is None:
            rng = np.random.default_rng(42)

        d = float(discount_d)
        theta = float(concentration_theta)

        if not (0.0 <= d < 1.0):
            raise ValueError(f"discount_d must be in [0,1); got {d}")
        if theta <= -d:
            raise ValueError(f"concentration_theta must satisfy theta > -d; got theta={theta}, d={d}")

        c_vec: List[int] = []
        na_dict: Dict[int, int] = {}
        active_labels: List[int] = []  # keeps current labels (A_{i-1})

        next_label = start_label

        for i in range(1, n_lists + 1):
            denom = theta + i - 1.0
            if denom <= 0.0:
                # This can only happen if theta is negative and i is small; rule disallows it.
                raise ValueError(f"Invalid denominator at i={i}: theta + i - 1 = {denom} <= 0")

            # weights for existing clusters
            weights = []
            labels = []
            for a in active_labels:
                w = na_dict[a] - d
                # Numerical safety: should be positive since na>=1 and d<1
                weights.append(max(w, 0.0))
                labels.append(a)

            # weight for a new cluster
            w_new = theta + d * len(active_labels)
            weights.append(max(w_new, 0.0))
            labels.append(None)  # None sentinel indicates "new cluster"

            # normalize
            w_arr = np.array(weights, dtype=float)
            s = w_arr.sum()
            if s <= 0.0 or not np.isfinite(s):
                # Extremely defensive fallback: uniform over options
                probs = np.ones_like(w_arr) / len(w_arr)
            else:
                probs = w_arr / s

            # categorical draw
            u = rng.random()
            cdf = np.cumsum(probs)
            idx = int(np.searchsorted(cdf, u, side="right"))
            chosen = labels[idx]

            if chosen is None:
                # start a new cluster
                chosen = next_label
                next_label += 1
                active_labels.append(chosen)
                na_dict[chosen] = 1
            else:
                # join existing
                na_dict[chosen] += 1

            c_vec.append(chosen)

        return c_vec, na_dict

    @staticmethod
    def log_U_prior(Z: np.ndarray, rho: float, K: int, debug: bool = False) -> float:
        """
        Compute the log prior probability of Z.

        Parameters:
        - Z: Current latent variable matrix (numpy.ndarray).
        - rho: Step size for proposal (used here to scale covariance).
        - K: Number of dimensions.
        - debug: If True, prints the covariance matrix.

        Returns:
        - log_prior: Scalar log prior probability.
        """
        # Covariance matrix is scaled identity matrix
        Sigma =BasicUtils.build_Sigma_rho(K,rho)

        if debug:
            print(f"Covariance matrix Sigma:\n{Sigma}")

        # Compute log prior for each row in Z assuming independent MVN
        try:
            mvn = multivariate_normal(mean=np.zeros(K), cov=Sigma)
            log_prob = mvn.logpdf(Z)
            log_prior = np.sum(log_prob)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError in log_prior: {e}")
            print(f"Covariance matrix Sigma:\n{Sigma}")
            raise e

        return log_prior
    @staticmethod
    def log_U_prior_optimized(Z: np.ndarray, rho: float, K: int, debug: bool = False) -> float:
        """
        OPTIMIZED: Vectorized log prior computation for U0.
        
        Up to 20x faster than original by eliminating scipy object creation
        and using pure numpy operations.
        """
        # Build covariance matrix
     
        
        # Use pure numpy operations instead of scipy
        s1 = np.einsum('ij,ij->i', Z, Z)         # ||x||^2 row-wise
        s2 = np.sum(Z, axis=1)                   # 1^T x row-wise
        inv_scale = 1.0 / (1.0 - rho)
        c = rho / (1.0 + (K-1.0)*rho)
        quad = inv_scale * (s1 - c * s2 * s2)
        logdet = (K-1.0)*np.log1p(-rho) + np.log1p(-rho + rho*K)
        const = K*np.log(2.0*np.pi)
        return float(-0.5*(Z.shape[0]*const + Z.shape[0]*logdet + np.sum(quad)))
            
                
         
    @staticmethod
    def transform_U_to_eta(U: np.ndarray) -> np.ndarray:
        """
        Transform latent positions U to eta using Gumbel link function.
        
        Parameters:
        -----------
        U : np.ndarray
            Matrix of latent positions (n_global × K)
        Returns:
        --------
        np.ndarray
            Matrix of transformed positions (n_global × K)
        """
        n_global, K = U.shape
        
        # Initialize output matrix
        eta = np.zeros((n_global, K))
        
        # Gumbel inverse link function with boundary protection
        def gumbel_inv(p):
            p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.log(-np.log(p_clipped))
        
        # Transform each row
        for j in range(n_global):
            # Step 1: Convert to probabilities using normal CDF
            p_vec = norm.cdf(U[j, :])
            
            # Step 2: Apply Gumbel inverse link function
            gumbel_vec = np.array([gumbel_inv(px) for px in p_vec])
            
            eta[j, :] = gumbel_vec
        
        return eta
    @staticmethod
    def description_partial_order(h: np.ndarray) -> Dict[str, Any]:
        """
        Provides a detailed description of the partial order represented by the adjacency matrix h.

        Parameters:
        - h: An n x n numpy array representing the adjacency matrix of the partial order.

        Returns:
        - description: A dictionary containing descriptive statistics of the partial order.
        """
        G = nx.DiGraph(h)
        n = h.shape[0]
        node_num= G.number_of_nodes()

        # Number of relationships (edges)
        num_relationships = G.number_of_edges()

        # Number of alone nodes (no incoming or outgoing edges)
        alone_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
        num_alone_nodes = len(alone_nodes)

        # Maximum number of relationships a node can have with other nodes
        # Considering both in-degree and out-degree
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 0
        max_out_degree = max(out_degrees.values()) if out_degrees else 0
        max_relationships = max(max_in_degree, max_out_degree)

        # Number of linear extensions
        tc=BasicUtils.transitive_closure(h)
        tr = BasicUtils.transitive_reduction_optimized(tc)
        num_linear_extensions = BasicUtils._r(tr)

        # Depth of the partial order (length of the longest chain)
        try:
            depth = nx.dag_longest_path_length(G)
        except nx.NetworkXUnfeasible:
            depth = None  # If the graph is not a DAG

        description = {
            "Number of Nodes": node_num,
            "Number of Relationships": num_relationships,
            "Number of Alone Nodes": num_alone_nodes,
            "Alone Nodes": alone_nodes,
            "Maximum In-Degree": max_in_degree,
            "Maximum Out-Degree": max_out_degree,
            "Maximum Relationships per Node": max_relationships,
            "Number of Linear Extensions": num_linear_extensions,
            "Depth of Partial Order": depth
        }

        # Print the description
        print("\n--- Partial Order Description ---")
        for key, value in description.items():
            print(f"{key}: {value}")
        print("---------------------------------")



    @staticmethod
    def sample_conditional_z(Z, rZ, cZ, rho):
        K = Z.shape[1]

        # Build correlation matrix
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1.0)

        dependent_ind = cZ
        given_inds = [i for i in range(K) if i != cZ]

        Sigma_dd = Sigma[dependent_ind, dependent_ind]  # scalar
        Sigma_dg = Sigma[dependent_ind, given_inds]     # shape (K-1,)  <-- FIXED HERE
        Sigma_gg = Sigma[np.ix_(given_inds, given_inds)]

        # X_g is also shape (K-1,)
        X_g = Z[rZ, given_inds]

        # Means are 0
        mu_d = 0.0
        mu_g = 0.0

        # Invert Sigma_gg
        try:
            Sigma_gg_inv = np.linalg.inv(Sigma_gg)
        except np.linalg.LinAlgError:
            Sigma_gg += np.eye(Sigma_gg.shape[0]) * 1e-8
            Sigma_gg_inv = np.linalg.inv(Sigma_gg)

        # Conditional mean
        mu_cond = mu_d + Sigma_dg @ Sigma_gg_inv @ (X_g - mu_g)
        # Conditional variance
        var_cond = Sigma_dd - Sigma_dg @ Sigma_gg_inv @ Sigma_dg

        var_cond = max(var_cond, 1e-8)
        Z_new = np.random.normal(loc=mu_cond, scale=np.sqrt(var_cond))

        return Z_new

#############################################Hyperparameter prior for HPO#############################################

#   ### rho 
    @staticmethod
    def rRprior(fac=1/6, tol=5e-3, rng=None):
        """
        Draw a sample for ρ from a Beta(1, fac) distribution, but reject any sample
        for which 1 - ρ < tol, to avoid numerical instability when ρ is extremely close to 1.
        
        Parameters:
        fac: Second parameter of the Beta distribution (default 1/6).
        tol: Tolerance such that we require 1 - ρ >= tol (default 1e-4).
        rng: numpy random generator (if None, uses scipy.stats)
        
        Returns:
        A single float value for ρ.
        """
        while True:
            if rng is not None:
                rho = rng.beta(1, fac)
            else:
                rho = beta.rvs(1, fac)
            if 1 - rho >= tol:
                return rho
    @staticmethod
    def dRprior(rho: float, fac=1/6, tol=5e-3) -> float:
        """
        Compute the log prior for ρ from a Beta(1, fac) distribution, with truncation
        at 1 - tol. If ρ > 1 - tol, return -Inf. Otherwise, adjust the log density
        by subtracting the log cumulative probability at 1-tol.
        
        Parameters:
        rho: the value of ρ.
        fac: the Beta distribution second parameter (default 1/6).
        tol: tolerance for the upper bound (default 1e-4).
        
        Returns:
        The log density (a float).
        """
        if rho > 1 - tol:
            return -np.inf
        # Compute the log PDF at rho.
        log_pdf = beta.logpdf(rho, 1, fac)
        # Subtract the log of the cumulative probability up to 1-tol, effectively renormalizing.
        log_cdf_trunc = beta.logcdf(1 - tol, 1, fac)
        return log_pdf - log_cdf_trunc
####Prob 

    @staticmethod
    def rPprior(noise_beta_prior, rng=None):
        if rng is not None:
            return rng.beta(1, noise_beta_prior)
        else:
            return beta.rvs(1, noise_beta_prior)
    
    @staticmethod
    def dPprior(p, beta_param):
        """
        Log-prior for p ~ Beta(1, beta_param).

        Returns -inf if p is out of (0,1).
        Otherwise, logpdf of Beta(1, beta_param).
        """
        if p <= 0.0 or p >= 1.0:
            return -math.inf
        
        return beta.logpdf(p, 1.0, beta_param)

### Tau 
    @staticmethod
    def rTauprior(tol: float = 5e-3, rng=None):
        """Sample tau ~ Uniform(0, 1 - tol] to avoid singular covariance when tau is too close to 1."""
        if rng is not None:
            return rng.uniform(0.0, 1.0 - tol)
        else:
            return random.uniform(0.0, 1.0 - tol)


    @staticmethod
    def dTauprior(tau: float, tol: float = 5e-3):
        """Log-density of the (truncated) Uniform(0, 1-tol] prior for tau."""
        if tau <= 0.0 or tau >= 1.0 - tol:
            return -math.inf
        # Density of Uniform(0, 1-tol] is 1/(1-tol)
        return -math.log(1.0 - tol)
####K  
    @staticmethod
    def dKprior(k: int, lam: float) -> float:
        """Log PMF of Poisson(λ) truncated at k ≥ 1."""
        if k < 1:
            return -np.inf
        # log(k!) using gammaln(k+1)
        log_k_fact = math.lgamma(k+1)
        # normalizing constant for truncation
        norm_const = -np.log(1 - np.exp(-lam))
        val = -lam + k * np.log(lam) - log_k_fact + norm_const
        return val
    
    @staticmethod
    def rKprior(lam: float = 3.0) -> int:
        """
        Sample a new K from a Poisson(λ) distribution truncated to k ≥ 1.
        
        This function repeatedly draws from a Poisson(λ) until a value ≥ 1 is obtained.
        """
        candidate = np.random.poisson(lam)
        while candidate < 1:
            candidate = np.random.poisson(lam)
        return candidate


    
    @staticmethod
    def log_U_hierarchical_prior(
        U0: np.ndarray,                  # shape (|M0|, K)
        U_a_list: list,                  # length A, each shape (|M_a|, K)
        M_a_dict: list,                  # length A, each is a list of global object indices
        tau: float,
        Sigma_rho: np.ndarray       # shape (K,K)
                    # function log_mvnorm(x, mean, cov) -> float
    ) -> float:

        logp = 0.0

        # 1) log for each U^(0)[j,:] ~ N(0, Sigma_rho)
        n_global = U0.shape[0]
        for j in range(n_global):
            x_j = U0[j,:]                # a 1D vector of length K
            zero_vec = np.zeros_like(x_j)
            logp += np.log(multivariate_normal(x_j, zero_vec, Sigma_rho))

        # 2) for each assessor a, for each j in M_a
        A = len(U_a_list)
        for a_idx in range(A):
            Ua = U_a_list[a_idx]        # shape (|M_a|, K)
            Ma = M_a_dict.get(a_idx,[])            # list of global indices
            for row_loc, j_global in enumerate(Ma):
                # row in U^(a) => U_a_list[a_idx][row_loc,:]
                x_aj = Ua[row_loc,:]
                # mean is tau * U0[j_global,:]
                mean_aj = tau * U0[j_global,:]
                # cov is (1 - tau^2)*Sigma_rho                
                cov_aj = (1.0 - tau**2) * Sigma_rho


                logp += np.log(multivariate_normal(x_aj, mean_aj, cov_aj))

        return logp
    @staticmethod
    def sample_conditional_column(Z, rho, rng=None):
        """
        Z is shape (n, K). For each row i, we want the bridging col
        of shape (n,) that respects the correlation among columns.
        
        We assume an equicorrelation or some covariance Sigma_full 
        of shape (K+1, K+1).
        """
        n, K = Z.shape
        Kplus1 = K + 1

        # Build the (K+1)x(K+1) covariance:
        Sigma_full = BasicUtils.build_Sigma_rho(Kplus1, rho)
        # Partition Sigma_full:
        # Sigma_gg = Sigma_full[0:K,0:K]
        # Sigma_dg = Sigma_full[K,   0:K]
        # Sigma_dd = Sigma_full[K,   K]
        
        Sigma_gg = Sigma_full[:K, :K]
        Sigma_dg = Sigma_full[K, :K]       # shape (K,)
        Sigma_dd = Sigma_full[K, K]        # scalar

        # Invert Sigma_gg once for all
        Sigma_gg_inv = np.linalg.inv(Sigma_gg)

        bridging_col = np.zeros(n)
        for i in range(n):
            x_i = Z[i,:]  # existing coords
            # conditional mean
            mu_cond = Sigma_dg @ Sigma_gg_inv @ x_i
            # conditional var
            var_cond = Sigma_dd - Sigma_dg @ Sigma_gg_inv @ Sigma_dg
            # sample
            if rng is not None:
                bridging_col[i] = rng.normal(mu_cond, np.sqrt(var_cond))
            else:
                bridging_col[i] = np.random.normal(mu_cond, np.sqrt(var_cond))

        return bridging_col
        
    @staticmethod
    def sample_conditional_column_child(Ua, U0_subset, tau, rho, rng):
        """
        Ua        : (n_a , K)   – current assessor matrix (all K old columns)
        U0_subset : (n_a , K)   – corresponding rows (same items) from U0
        returns   : vector length n_a  – new column c
        """
        n_a, K = Ua.shape
        Sigma_num = (1 - tau**2)*(1 - rho)*(1 + (K-1)*rho)
        Sigma_den = 1 + (K-2)*rho
        var_c     = Sigma_num / Sigma_den       # scalar σ²_{j,c}

        # pre-compute the normalising factor
        w = rho / (1 + (K-2)*rho)

        new_col = np.empty(n_a)
        for j in range(n_a):
            diff = Ua[j] - tau * U0_subset[j]   # vector (K,)
            mu   = tau * U0_subset[j, 0] + w * diff.sum()  # formula above
            new_col[j] = rng.normal(mu, np.sqrt(var_c))
        return new_col

    def U0_conditional_update(
        j_global,        # index of the row in U0 we want to update
        U0,              # current U0, shape (n_global, K)
        U_a_dict,        # dictionary of child latents {a: U^a}, each shape (len(M_a), K)
        M_a_dict,        # {a: list_of_indices_in_M_a}, tells which global indices belong to assessor a
        tau,             # correlation parameter
        Sigma_rho,       # K x K covariance for the base distribution
        rng              # random number generator, e.g., np.random.default_rng()
    ):
        """
        Perform a direct Gibbs draw for row j_global of U0 given all child rows U^(a).
        """
        # 1) Gather all child-latent vectors that correspond to the same "global" item j_global
        #    For each a in U_a_dict, find the local index i_loc where j_global appears in M_a_dict[a].
        #    If j_global is not in M_a_dict[a], skip it. Otherwise get U_a[i_loc].
        child_vectors = []
        for a, U_a in U_a_dict.items():
            if j_global in M_a_dict[a]:
                i_loc = M_a_dict[a].index(j_global)
                child_vectors.append(U_a[i_loc, :])
        
        A_j = len(child_vectors)  # how many assessors actually have j_global in their list

        # 2) If no child has j_global, posterior = prior => Normal(0, Sigma_rho)
        if A_j == 0:
            post_mean = np.zeros_like(U0[j_global, :])
            post_cov = Sigma_rho
        else:
            # 3) Compute the posterior mean & covariance for that row
            sum_child = np.sum(child_vectors, axis=0)  # sum_{a=1..A_j} u_j^(a)

            denom = (1 - tau**2) + A_j * (tau**2)
            # Posterior mean
            post_mean = (tau / denom) * sum_child

            # Posterior covariance
            shrink_factor = (1 - tau**2) / denom
            post_cov = shrink_factor * Sigma_rho
        
        # 4) Draw a new sample from N(post_mean, post_cov)
        new_row = rng.multivariate_normal(post_mean, post_cov)
    

        return new_row 

####################Below are separate coding for hpo####
    @staticmethod
    def gumbel_inv_cdf(p: float, eps: float = 1e-15) -> float:
        # Clip probability to avoid numerical issues at boundaries
        p_clipped = np.clip(p, eps, 1 - eps)
        return -np.log(-np.log(p_clipped))
    
    @staticmethod
    def log_U_a_prior(U_a_dict: Dict[int, np.ndarray], tau: float, rho: float, K: int, M_a_dict: Dict[int, List[int]], U0: np.ndarray) -> float:
        """
        Compute the log prior probability for assessor-level latent variables.
        
        Each assessor a has latent variables U_a ~ N(tau * U0[j], (1 - tau^2)*Sigma_rho)
        for each global item j in M_a_dict[a].

        Parameters:
        U_a_dict: Dictionary with keys as assessor IDs and values as latent matrices (shape: (|M_a|, K)).
        tau: The branching parameter.
        rho: The correlation parameter.
        K: Dimensionality of the latent space.
        M_a_dict: Dictionary with keys as assessor IDs and values as lists of global item indices for that assessor.
        U0: Global latent matrix (shape: (|M0|, K)).

        Returns:
        log_prior_total: The sum of log prior probabilities over all assessor-level latent variables.
        """
        Sigma_rho =BasicUtils.build_Sigma_rho(K,rho)
        log_prior_total = 0.0

        for a, U_a in U_a_dict.items():
            # Get the list of global items for assessor a.
            Ma = M_a_dict.get(a, [])
            log_prior = 0.0
            for i, j_global in enumerate(Ma):

                mean_vec = tau * U0[j_global, :]

                cov_mat= (1.0 - tau**2) * Sigma_rho
                log_prob = multivariate_normal.logpdf(
                            U_a[i, :],
                            mean=mean_vec,
                            cov=cov_mat,
                            allow_singular=True
                        )

                log_prior += log_prob
            log_prior_total += log_prior

        return log_prior_total


    @staticmethod


    def log_U_a_prior_fast(
        U_a_dict: Dict[int, np.ndarray],
        tau: float,
        rho: float,
        K: int,
        M_a_dict: Dict[int, List[int]],
        U0: np.ndarray,
        *,                      # force keywords after here
        regularise: float = 1e-8
    ) -> float:
        """
        Ultra‑fast vectorised prior with *cached* Σ⁻¹ and log|Σ|.
        NOTE: This is the HIERARCHICAL version (used when tau and U0 are present).
        For independent assessor model, use log_U_a_prior_independent instead.
        """
        total=0.0
        for a, Ua in U_a_dict.items():
            Ma = M_a_dict[a]
            idx = np.array([i for i in Ma], dtype=int)     # global IDs must already be 0..|M0|-1; if not, map them
            Z = Ua - tau * U0[idx, :]
            # scale covariance is (1 - tau^2) * Σρ -> just multiply the density by scale factor in the quad/logdet
            # Equivalent: divide Z by sqrt(1 - tau^2) and use Σρ
            denom = max(1e-8, (1.0 - tau*tau))
            Z_scaled = Z / np.sqrt(denom)
            total += StatisticalUtils.log_U_prior_optimized(Z_scaled, rho, K) - 0.5*Ua.shape[0]*K*np.log(denom)
        return float(total)

    @staticmethod
    def log_U_a_prior_independent(
        U_a_dict: Dict[int, np.ndarray],
        rho: float,
        K: int,
    ) -> float:
        """
        Log prior for assessor-level latent variables in the NON-HIERARCHICAL model.
        
        Each assessor's latent utilities are INDEPENDENT:
            U_a ~ N(0, Sigma_rho)
        
        This is used when there is NO global partial order (no tau, no U0).
        Each assessor has their own partial order sampled independently.
        
        Parameters:
        -----------
        U_a_dict : Dict[int, np.ndarray]
            Dictionary mapping assessor IDs to their latent matrices (shape: n_a x K)
        rho : float
            Correlation parameter for the covariance matrix
        K : int
            Dimensionality of the latent space
            
        Returns:
        --------
        float
            Sum of log prior probabilities for all assessor latent variables
        """
        total = 0.0
        for a, Ua in U_a_dict.items():
            # Each row of Ua is independently sampled from N(0, Sigma_rho)
            # Use the optimized log prior function
            total += StatisticalUtils.log_U_prior_optimized(Ua, rho, K)
        return float(total)

    @staticmethod
    ### The objective of this function is buidling a hierarchical partial order of H(U) given M0, Ma, Oa_list and U_alist 
    def build_hierarchical_partial_orders(
        M0,
        assessors,
        M_a_dict,
        U0,           # shape: (|M0|, K) or None for non-hierarchical model
        U_a_dict,
        link_inv=None
    ):
        """
        Build partial orders for assessors.
        
        Supports both hierarchical (with U0) and non-hierarchical (U0=None) models.
        In hierarchical model: builds global h0 from U0 and assessor h_a from U_a
        In non-hierarchical model: only builds assessor h_a from U_a (no global h0)
        """
        if link_inv is None:
            # Default to Gumbel quantile
            link_inv = StatisticalUtils.gumbel_inv_cdf

        h_U = {}
        
        # Build global partial order ONLY if U0 is provided (hierarchical model)
        if U0 is not None:
            n_global, K = U0.shape
            eta0 = np.zeros_like(U0)
            for j_global in range(n_global):
                p_vec = norm.cdf(U0[j_global, :])  # coordinate-wise
                gumbel_vec = np.array([link_inv(px) for px in p_vec])
                eta0[j_global, :] = gumbel_vec

            h0 = BasicUtils.generate_partial_order(eta0)
            h_U[0] = h0
        # NOTE: For non-hierarchical model (U0=None), we skip global partial order

        # Loop over assessors to build assessor-specific partial orders
        for idx_a, a in enumerate(assessors):
            # 1) Build the *full partial order* on M_a
            Ma = M_a_dict.get(a,[])               # e.g. [0,2,4]
            Ua = U_a_dict.get(a,[])               # shape (|M_a|, K)
            
            # Skip empty assessors
            if len(Ma) == 0 or (isinstance(Ua, np.ndarray) and Ua.size == 0):
                continue
            
            # (a) Compute eta^(a) for each item j in M_a
            #     eqn (21): eta_j^{(a)} = G^-1( Phi(U_j^{(a)}) ) + alpha_j
            # We do it row by row
            eta_a = np.zeros_like(Ua)
            for i_loc, j_global in enumerate(Ma):
                p_vec = norm.cdf(Ua[i_loc, :])
                gumbel_vec = np.array([link_inv(px) for px in p_vec])
                eta_a[i_loc, :] = gumbel_vec

            # adjacency for M_a
            h_a = BasicUtils.generate_partial_order(eta_a)
            # store in dictionary
            h_U[a] = h_a

        return h_U
    
    @staticmethod
    def dict_array_equal(d1, d2):
        """Recursively compare two dictionaries where values may be NumPy arrays."""
        if d1.keys() != d2.keys():
            return False
        for key in d1:
            v1, v2 = d1[key], d2[key]
            if isinstance(v1, dict) and isinstance(v2, dict):
                if not StatisticalUtils.dict_array_equal(v1, v2):
                    return False
            elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True
    @staticmethod
    def generate_total_order_for_choice_set_with_queue_jump(
        subset: List[int],
        M_a: List[int],
        h_local: np.ndarray,
        prob_noise: float
    ) -> List[int]:
        """
        Given:
        - 'subset': A list of *global* item IDs we want to order.
        - 'M_a': The assessor's entire set of global item IDs (size = |M_a|).
        - 'h_local': A local partial-order matrix of shape (|M_a|, |M_a|),
                    where h_local[i,j]=1 => "item M_a[i] < item M_a[j]" in the assessor's order.
        - 'prob_noise': Probability of a 'jump' (i.e., random pick) in the queue-jump.

        Returns:
        A total order of items in 'subset', as a list of global item IDs.
        """
        
        # CRITICAL: Ensure h_local is a transitive closure for correct submatrix extraction
        h_local = BasicUtils.ensure_transitive_closure(h_local)

        # 1) Build a map from *global* item => local index in M_a
        #    so we can slice h_local properly.
        global2local = { g: i for i, g in enumerate(M_a) }

        # 2) Identify which global items in 'subset' are also in M_a,
        #    and convert them to local indices
        local_subset_idx = []
        local_subset_global = []  # store the same items, but parallel to local indices
        for g in subset:
            if g in global2local:          # only items that exist in M_a
                local_idx = global2local[g]
                local_subset_idx.append(local_idx)
                local_subset_global.append(g)

        # If no overlap, return empty
        if not local_subset_idx:
            return []

        # 3) Extract the local submatrix for these items
        #    shape = (len(local_subset_idx), len(local_subset_idx))
        h_matrix_subset = h_local[np.ix_(local_subset_idx, local_subset_idx)]

        # 4) We'll do the queue-jump logic in local SUBSET indices = [0..(n_sub-1)]
        n_sub = len(local_subset_idx)
        # So we make a direct mapping from "subset index" => "global item ID"
        # e.g. subset_idx2global[i] = local_subset_global[i]
        # And we'll keep 'remaining' as [0..n_sub-1].
        subset_idx2global = { i: local_subset_global[i] for i in range(n_sub) }

        remaining = list(range(n_sub))  # local indices in [0..n_sub-1]
        total_order_local = []


        while remaining:
            m = len(remaining)
            if m == 1:
                total_order_local.append(remaining[0])
                break

            # Build sub-submatrix for 'remaining'
            # shape => (m, m)
            h_rem = h_matrix_subset[np.ix_(remaining, remaining)]

            # Transitive reduction of that sub-submatrix
            tr_rem = BasicUtils.transitive_reduction_optimized(h_rem)

            # Count total # of linear extensions
            N_total = BasicUtils.nle(tr_rem)

            # Compute candidate probabilities for each local_idx in [0..m-1]
            candidate_probs = []
            for local_idx in range(m):
                # Number of linear extensions that start with 'local_idx'
                # This uses BasicUtils.num_extensions_with_first
                # but that function expects the partial order submatrix + top elements, etc.
                # So local_idx is an index in [0..m-1].
                # We pass 'tr_rem' and local_idx to BasicUtils.num_extensions_with_first
                N_first = BasicUtils.num_extensions_with_first(tr_rem, local_idx)
                p_no_jump = (1 - prob_noise) * (N_first / N_total)
                candidate_probs.append(p_no_jump)

            # Probability of 'jump' => prob_noise, distributed uniformly among m candidates
            p_jump = prob_noise * (1.0 / m)
            candidate_probs = [p + p_jump for p in candidate_probs]

            # normalize
            total_p = sum(candidate_probs)
            candidate_probs = [p / total_p for p in candidate_probs]

            # Sample an index from 'remaining' with these weights
            chosen_subindex = random.choices(range(m), weights=candidate_probs, k=1)[0]
            chosen_local = remaining[chosen_subindex]

            total_order_local.append(chosen_local)
            remaining.remove(chosen_local)

        # 5) Convert 'total_order_local' (which are indices in [0..n_sub-1])
        #    back to *global* item IDs
        global_order = [subset_idx2global[i] for i in total_order_local]

        return global_order

    @staticmethod
    def truncated_poisson_pdf(x, mu):
        """
        Calculate the PDF of a truncated Poisson distribution.
        
        Parameters:
        - x: Value to evaluate the PDF at
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - PDF value at x
        """
        # Use dKprior to calculate the log probability
        log_prob = StatisticalUtils.dKprior(x, mu)
        # Convert from log probability to probability
        return np.exp(log_prob)
    
    @staticmethod
    def truncated_poisson_cdf(x, mu):

        if x < 1:
            return 0
        
        # Calculate the CDF by summing the PDF values from 1 to x
        cdf = 0
        for k in range(1, int(x) + 1):
            cdf += StatisticalUtils.truncated_poisson_pdf(k, mu)
        
        return cdf
    
    @staticmethod
    def truncated_poisson_mean(mu):
        """
        Calculate the mean of a truncated Poisson distribution.
        
        Parameters:
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - Mean of the truncated distribution
        """
        from scipy.stats import poisson
        # Mean of truncated Poisson
        norm_const = 1 - poisson.pmf(0, mu)
        return mu / norm_const
    
    @staticmethod
    def truncated_poisson_var(mu):
        """
        Calculate the variance of a truncated Poisson distribution.
        
        Parameters:
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - Variance of the truncated distribution
        """
        from scipy.stats import poisson
        # Variance of truncated Poisson
        norm_const = 1 - poisson.pmf(0, mu)
        return mu * (1 - mu * poisson.pmf(0, mu) / norm_const) / norm_const
    
    @staticmethod
    def TruncatedPoisson(mu):
        """
        Create a truncated Poisson distribution object.
        
        Parameters:
        - mu: Mean parameter of the Poisson distribution
        
        Returns:
        - A distribution object with pdf, cdf, mean, and var methods
        """
        class TruncatedPoissonDist:
            def __init__(self, mu):
                self.mu = mu
                self.name = "TruncatedPoisson"
                
            def pdf(self, x):
                return StatisticalUtils.truncated_poisson_pdf(x, self.mu)
                
            def cdf(self, x):
                return StatisticalUtils.truncated_poisson_cdf(x, self.mu)
                
            def mean(self):
                return StatisticalUtils.truncated_poisson_mean(self.mu)
                
            def var(self):
                return StatisticalUtils.truncated_poisson_var(self.mu)
        
        return TruncatedPoissonDist(mu)


    @staticmethod
    def dBetaprior(beta: np.ndarray, sigma_beta: Union[float, np.ndarray]) -> float:
        """
        Log-pdf of a multivariate Normal(0, Sigma) at point 'beta', where Sigma is a diagonal matrix.
        
        Parameters:
        -----------
        beta: shape (p,)
            Vector of coefficients
        sigma_beta: float or np.ndarray of shape (p,)
            The prior standard deviation(s) for each coefficient.
            Can be either a scalar (same std dev for all coefficients) or an array (different std dev per coefficient)
        
        Returns:
        --------
        float
            The log-density value
            
        Notes:
        ------
        When sigma_beta is a scalar, formula is:
          - (p/2) * log(2*pi) 
          - p*log(sigma_beta)
          - (1 / (2*sigma_beta^2)) * sum(beta^2)
          
        When sigma_beta is an array, formula is:
          - (p/2) * log(2*pi) 
          - sum(log(sigma_beta))  # sum of logs instead of p times log of one value
          - sum(beta^2 / (2*sigma_beta^2))  # element-wise division by the variances
        """
        p = len(beta)
        
        if np.isscalar(sigma_beta):
            # Original implementation for scalar sigma_beta
            log_det_part = -0.5 * p * math.log(2.0 * math.pi) - p * math.log(sigma_beta)
            quad_part = -0.5 * np.sum(beta**2) / (sigma_beta**2)
        else:
            # Handle array case
            if len(sigma_beta) != p:
                raise ValueError(f"sigma_beta must be a scalar or have length {p} to match beta")
            
            log_det_part = -0.5 * p * math.log(2.0 * math.pi) - np.sum(np.log(sigma_beta))
            quad_part = -0.5 * np.sum(beta**2 / (sigma_beta**2))
            
        return log_det_part + quad_part

    @staticmethod
    def rBetaPrior(sigma_beta: Union[float, np.ndarray], p: int) -> np.ndarray:
        """
        Sample a new beta from a Normal(0, Sigma) distribution, where Sigma is a diagonal matrix.
        
        Parameters:
        -----------
        sigma_beta: float or np.ndarray
            If scalar: the same prior std dev for each coefficient (diagonal elements will be sigma_beta^2)
            If array: different prior std dev for each coefficient (must have length p)
        p: integer
            Dimension of beta.
        
        Returns:
        --------
        np.ndarray
            Sampled beta vector of shape (p,)
        """
        if np.isscalar(sigma_beta):
            return np.random.normal(loc=0.0, scale=sigma_beta, size=(p,))
        else:
            if len(sigma_beta) != p:
                raise ValueError(f"If sigma_beta is an array, it must have length {p}")
            return np.random.normal(loc=0.0, scale=sigma_beta, size=(p,))
        
    @staticmethod
    def generate_cluster_assignments_crp(n_lists: int, alpha: float, rng) -> Tuple[List[int], Dict[int, int]]:
        """
        Generate cluster assignments using Chinese Restaurant Process.
        
        Args:
            n_lists: Number of assessor lists
            alpha: CRP concentration parameter
            rng: Random number generator
        
        Returns:
            c_vec: List of cluster assignments for each list
            na_dict: Dictionary mapping cluster_id -> number of lists in cluster
        """
        c_vec = []
        cluster_counts = defaultdict(int)
        
        for i in range(n_lists):
            # Calculate probabilities for existing clusters
            existing_clusters = list(cluster_counts.keys())
            
            if existing_clusters:
                # Probability of joining existing cluster k
                probs_existing = [cluster_counts[k] / (i + alpha) for k in existing_clusters]
                # Probability of creating new cluster
                prob_new = alpha / (i + alpha)
                
                all_probs = probs_existing + [prob_new]
                all_options = existing_clusters + [max(existing_clusters) + 1]
                
                # Sample cluster assignment
                chosen_cluster = rng.choice(all_options, p=all_probs)
            else:
                # First list always goes to cluster 1
                chosen_cluster = 1
            
            c_vec.append(chosen_cluster)
            cluster_counts[chosen_cluster] += 1
        
        return c_vec, dict(cluster_counts)




    @staticmethod
    def build_final_partial_orders(
        h_trace: List[Dict[Any, np.ndarray]],
        *,
        method: str = "mode",              # "mode" or "threshold"
        threshold: float = 0.5,            # used only when method="threshold"
        top_n: int = 4,
        item_labels: Optional[List[Any]] = None,
        plot_top: bool = True,
    ) -> Dict[Any, np.ndarray]:
        """
        Aggregate per-assessor partial orders into final H matrices.
        
        Parameters
        ----------
        h_trace : list of dict
            Each element is a dict {assessor_id -> adjacency matrix} for that iteration.
        method : {"mode", "threshold"}
            - "mode": choose the most frequent relationship per edge (i→j, j→i, incomparable).
            - "threshold": average probability and threshold at `threshold`.
        threshold : float
            Mean threshold when method="threshold".
        top_n : int
            Number of most frequent partial orders to show in diagnostics.
        item_labels : list, optional
            Labels to display in plots (falls back to 0..n-1).
        plot_top : bool
            Show top-N partial orders via `PO_plot` if available.
        
        Returns
        -------
        final_H : dict
            {assessor_id -> final adjacency matrix (after transitive reduction)}.
        """
        if not h_trace:
            raise ValueError("Empty H_trace provided.")

        assessor_ids = list(h_trace[0].keys())
        final_H = {}

        for assessor_id in assessor_ids:
            mats = [iter_dict[assessor_id] for iter_dict in h_trace if assessor_id in iter_dict]
            if not mats:
                continue

            T = len(mats)
            mats_arr = np.stack(mats, axis=0)  # (T, n, n)
            _, n, _ = mats_arr.shape

            if method == "threshold":
                mean_mat = np.mean(mats_arr, axis=0)
                agg_mat = (mean_mat >= float(threshold)).astype(np.int8)
            elif method == "mode":
                counts_ij = (mats_arr > 0).sum(axis=0)
                counts_ji = counts_ij.T
                counts_inc = T - counts_ij - counts_ji

                agg_mat = np.zeros((n, n), dtype=np.int8)
                iu = np.triu_indices(n, 1)
                for i, j in zip(*iu):
                    votes = {
                        (i, j): counts_ij[i, j],
                        (j, i): counts_ji[i, j],
                        None:  counts_inc[i, j],
                    }
                    best = max(votes.items(), key=lambda kv: kv[1])[0]
                    if best == (i, j):
                        agg_mat[i, j] = 1
                    elif best == (j, i):
                        agg_mat[j, i] = 1
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'mode' or 'threshold'.")

            # Direction-consistency cleanup (optional but keeps DAG tidy)
            if method == "mode":
                score = (counts_ij - counts_ji).sum(axis=1) / max(T, 1)
                order = np.argsort(-score)
                rank = np.empty(n, dtype=int); rank[order] = np.arange(n)
                for i in range(n):
                    for j in range(n):
                        if agg_mat[i, j] and rank[i] >= rank[j]:
                            agg_mat[i, j] = 0

            np.fill_diagonal(agg_mat, 0)

            # Remove transitive edges / enforce DAG
            H_final = BasicUtils.transitive_reduction(agg_mat.astype(int))
            final_H[assessor_id] = H_final

            if plot_top:
                sorted_hist = StatisticalUtils.count_unique_partial_orders(mats)
                total = sum(cnt for _, cnt in sorted_hist)
                top = sorted_hist[:top_n]
                print(f"Assessor {assessor_id}: {total} post-burn samples")
                for idx, (mat, cnt) in enumerate(top, 1):
                    pct = 100.0 * cnt / max(total, 1)
                    print(f"  {idx}. count={cnt} ({pct:.1f}%)")
                # Lazy import to avoid circular dependency
                try:
                    from src.utils.po_fun_plot import PO_plot
                    labels = item_labels if item_labels and len(item_labels) == n else list(range(n))
                    top_perc = [(mat, cnt, 100.0 * cnt / max(total, 1)) for mat, cnt in top]
                    PO_plot.plot_top_partial_orders(top_perc, top_n=len(top), item_labels=labels)
                except (ImportError, Exception):
                    pass

        return final_H
# =============================================================================
# OPTIMIZED HIERARCHICAL PARTIAL ORDER BUILDER
# =============================================================================

class OptimizedHierarchicalBuilder:
    """Optimized hierarchical partial order builder with caching and vectorization"""
    
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self):
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def vectorized_gumbel_transform(self, U: np.ndarray) -> np.ndarray:
        """
        Vectorized gumbel transformation: eta = G^-1(Phi(U))
        
        Args:
            U: shape (n, K) - latent variables
            
        Returns:
            eta: shape (n, K) - transformed values
        """
        # Vectorized CDF computation
        p_vec = norm.cdf(U)  # shape (n, K)
        
        # Vectorized gumbel inverse CDF
        # G^-1(p) = -log(-log(p))
        # Handle edge cases to avoid log(0) or log(negative)
        p_safe = np.clip(p_vec, 1e-10, 1 - 1e-10)
        gumbel_vec = -np.log(-np.log(p_safe))
        
        return gumbel_vec
    
    def vectorized_partial_order(self, eta: np.ndarray) -> np.ndarray:
        """
        Vectorized partial order generation
        
        Args:
            eta: shape (n, K) - transformed values
            
        Returns:
            h: shape (n, n) - adjacency matrix
        """
        # Vectorized comparison: eta[i] > eta[j] for all dimensions
        # Use broadcasting for efficiency
        greater = (eta[:, None, :] > eta[None, :, :])  # shape (n, n, K)
        h = np.all(greater, axis=2).astype(np.int8)    # collapse K-axis
        np.fill_diagonal(h, 0)                         # ensure h[i,i] = 0
        return h
    
    def build_hierarchical_partial_orders_optimized(
        self,
        M0: List[int],
        assessors: List[int],
        M_a_dict: Dict[int, List[int]],
        U0: np.ndarray,           # shape: (|M0|, K) or None for non-hierarchical model
        U_a_dict: Dict[int, np.ndarray],
        link_inv=None,
        use_cache: bool = False  # Disable caching since structural components are constant
    ) -> Dict[int, np.ndarray]:
        """
        Optimized version of build_hierarchical_partial_orders
        
        Key optimizations:
        1. Vectorized gumbel transformations
        2. Vectorized partial order generation
        3. Reduced function call overhead
        Note: Caching disabled since structural components are constant across iterations
        
        Supports both hierarchical (with U0) and non-hierarchical (U0=None) models.
        """
        if link_inv is not None:
            # Fall back to original implementation if custom link_inv is provided
            return self._build_hierarchical_partial_orders_original(
                M0, assessors, M_a_dict, U0, U_a_dict, link_inv
            )
        
        h_U = {}
        
        # 1. Build global partial order ONLY if U0 is provided (hierarchical model)
        if U0 is not None:
            n_global, K = U0.shape
            eta0 = self.vectorized_gumbel_transform(U0)
            h0 = self.vectorized_partial_order(eta0)
            h_U[0] = h0
        # NOTE: For non-hierarchical model (U0=None), we skip global partial order
        
        # 2. Build assessor-specific partial orders (vectorized)
        for a in assessors:
            Ma = M_a_dict.get(a, [])
            Ua = U_a_dict.get(a, np.array([]))
            
            # Handle empty cases - skip them like the original
            if len(Ma) == 0 or Ua.size == 0:
                continue
            
            # Vectorized transformation for this assessor
            eta_a = self.vectorized_gumbel_transform(Ua)
            h_a = self.vectorized_partial_order(eta_a)
            h_U[a] = h_a
        
        return h_U
    
    def _create_cache_key(self, M0, assessors, M_a_dict, U0, U_a_dict):
        """Create a cache key for the computation"""
        # Only cache structural components that remain constant
        # Don't cache U0 or U_a_dict as they are continuous variables
        key_components = (
            tuple(M0),
            tuple(assessors),
            tuple((k, tuple(v)) for k, v in sorted(M_a_dict.items()))
        )
        return hash(key_components)
    
    def _build_hierarchical_partial_orders_original(
        self,
        M0,
        assessors,
        M_a_dict,
        U0,
        U_a_dict,
        link_inv=None
    ):
        """Original implementation for fallback"""
        return StatisticalUtils.build_hierarchical_partial_orders(
            M0, assessors, M_a_dict, U0, U_a_dict, link_inv
        )

# Global instance for caching across calls
_optimized_builder = OptimizedHierarchicalBuilder()

def build_hierarchical_partial_orders_optimized(
    M0: List[int],
    assessors: List[int],
    M_a_dict: Dict[int, List[int]],
    U0: np.ndarray,
    U_a_dict: Dict[int, np.ndarray],
    link_inv=None,
    use_cache: bool = False  # Disable caching since structural components are constant
) -> Dict[int, np.ndarray]:
    """
    Optimized wrapper function for build_hierarchical_partial_orders
    
    Returns:
        h_U: Dictionary mapping assessor IDs to partial order matrices
    """
    return _optimized_builder.build_hierarchical_partial_orders_optimized(
        M0, assessors, M_a_dict, U0, U_a_dict, link_inv, use_cache
    )

def get_cache_stats():
    """Get cache statistics"""
    return _optimized_builder.get_cache_stats()

def clear_cache():
    """Clear the cache"""
    _optimized_builder.clear_cache()
