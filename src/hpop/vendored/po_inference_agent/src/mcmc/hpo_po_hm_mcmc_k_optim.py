import time
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any

from src.utils.po_fun import BasicUtils, StatisticalUtils
from src.utils.po_accelerator_nle_optimized import HPO_LogLikelihoodCache_Optimized

SOFTMAX_NOISE_OPTIONS = ("log_successors_queue_jump", "frontier_softmax_uni_utility")


def calculate_dimension_proportional_weights(
    n_items: int,
    cycle_length: int = 1000,
    has_K_updates: bool = True,
) -> tuple[dict, list]:
    """
    Calculate dimension-proportional weights and create sequential update schedule.
    
    Parameters:
    -----------
    n_items : int
        Number of items in the partial order
    cycle_length : int
        Length of update cycle (default 1000)
    has_K_updates : bool
        Whether to include K dimension updates
        
    Returns:
    --------
    tuple[dict, list]
        Dictionary of update frequencies and sequential update schedule list
    """
    # Calculate weighted dimensions for each parameter block
    # Single partial order model - just U, rho, noise, and optionally K
    weighted_dims = {
        'U': 1.0 * n_items,              # Update latent utilities (one update per item)
        'rho': 2.0 * 1,                  # Correlation parameter
        'noise': 2.0 * 1,                # Noise parameter
    }
    
    # Add optional parameter blocks
    # K updates are expensive (reversible jump), so use sqrt(n) scaling to avoid over-proposing
    if has_K_updates:
        weighted_dims['K'] = max(3.0, n_items ** 0.5)
    
    # Calculate proportional weights
    total_weighted_dim = sum(weighted_dims.values())
    mcmc_weights = {param: weight / total_weighted_dim 
                   for param, weight in weighted_dims.items()}
    
    # Convert to update frequencies per cycle
    update_frequencies = {param: int(weight * cycle_length) 
                         for param, weight in mcmc_weights.items()}
    
    # Ensure we have at least 1 update for each parameter type
    for param in update_frequencies:
        if update_frequencies[param] == 0:
            update_frequencies[param] = 1
    
    # Adjust total to match cycle_length exactly
    current_total = sum(update_frequencies.values())
    if current_total != cycle_length:
        largest_param = max(update_frequencies.keys(), key=lambda x: update_frequencies[x])
        update_frequencies[largest_param] += (cycle_length - current_total)
    
    # Create sequential update schedule
    update_schedule = []
    for param, freq in update_frequencies.items():
        update_schedule.extend([param] * freq)
    
    # Shuffle to avoid systematic patterns within each cycle
    random.shuffle(update_schedule)
    
    return update_frequencies, update_schedule


def mcmc_simulation_po(
    num_iterations: int,
    # Data
    items: List[int],
    choice_sets: List[List[int]],
    observed_orders: List[List[int]],
    # Model parameters
    dr: float,          # multiplicative step size for rho
    noise_option: str,
    # Priors
    rho_prior, 
    noise_beta_prior: float,
    K_prior: int,
    # Optional parameters
    fixed_K: Optional[int] = None,
    random_seed: int = 42,
    cycle_length: int = 1000,
    # Trembling-hand epsilon (fixed noise probability)
    epsilon: float = 0.01,
    # Softmax-specific parameters (beta is updated, epsilon is fixed)
    softmax_beta_prior: tuple = (2.0, 2.0),
    softmax_beta_stepsize: float = 0.3,
    init_state: Optional[Dict[str, Any]] = None, 
    checkpoint_interval: Optional[int] = None,
    checkpoint_callback = None
) -> Dict[str, Any]:
    """
    MCMC simulation for SINGLE partial order inference.
    
    This is a simplified model that infers ONE partial order from observed data.
    
    Parameters:
    -----------
    items : List[int]
        List of item indices (e.g., [0, 1, 2, 3, 4])
    choice_sets : List[List[int]]
        List of choice sets, each is a subset of items
    observed_orders : List[List[int]]
        List of observed total orders corresponding to choice sets
    dr : float
        Multiplicative step size for rho proposals
    noise_option : str
        Noise model: "queue_jump", "weighted_queue_jump", "log_successors_queue_jump", "frontier_softmax_uni_utility"
    rho_prior : float
        Prior parameter for rho (Beta distribution)
    noise_beta_prior : float
        Prior parameter for noise probability (only used if epsilon not fixed)
    K_prior : int
        Prior mean for latent dimension K
    epsilon : float
        Fixed trembling-hand noise probability ε ∈ (0,1). 
        For all likelihoods: p(y_t) = (1-ε)·π_β(y_t) + ε/|R_t|
    
    Returns:
    --------
    Dict with traces, final state, and diagnostics
    """
    
    n = len(items)
    item_to_index = {item: idx for idx, item in enumerate(items)}
    
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)
    update_K = fixed_K is None
    is_softmax_noise_model = noise_option in SOFTMAX_NOISE_OPTIONS
    softmax_beta = None
    epsilon_val = epsilon  # Fixed trembling-hand epsilon
    softmax_beta_trace = []
    epsilon_trace = []
    proposed_softmax_beta_vals = []

    # ───────────────────────────────────────────────────────────────
    # RESUME or FRESH START
    # ───────────────────────────────────────────────────────────────
    if init_state is not None:
        # Load traces
        U_trace = init_state.get("U_trace", [])
        H_trace = init_state.get("H_trace", [])
        rho_trace = init_state.get("rho_trace", [])
        K_trace = init_state.get("K_trace", [])
        prob_noise_trace = init_state.get("prob_noise_trace", [])
        softmax_beta_trace = init_state.get("softmax_beta_trace", [])
        epsilon_trace = init_state.get("epsilon_trace", [])
        acceptance_decisions = init_state.get("acceptance_decisions", [])
        acceptance_rates = init_state.get("acceptance_rates", [])
        log_likelihood_currents = init_state.get("log_likelihood_currents", [])
        log_likelihood_primes = init_state.get("log_likelihood_primes", [])
        
        if isinstance(log_likelihood_currents, list) and log_likelihood_currents:
            log_llk_current = log_likelihood_currents[-1]
        else:
            log_llk_current = -float("inf")
        
        num_acceptances = init_state.get("num_acceptances", 0)
        accepted_before_resume = num_acceptances
    
        # Current values
        iteration_start = init_state["iteration"]
        target_iterations = iteration_start + num_iterations
        rho = init_state["rho_final"]
        
        # Restore noise parameters
        if is_softmax_noise_model:
            # For softmax: epsilon_val is fixed, softmax_beta is restored/initialized
            softmax_beta = init_state.get("softmax_beta_final")
            if softmax_beta is None:
                softmax_beta = rng.gamma(softmax_beta_prior[0], 1.0 / softmax_beta_prior[1])
            prob_noise = None
        else:
            # For queue_jump: prob_noise is restored
            prob_noise = init_state.get("prob_noise_final")
            if prob_noise is None:
                prob_noise = StatisticalUtils.rPprior(noise_beta_prior, rng=rng)
        
        K = init_state["K_final"]
        if fixed_K is not None and K != fixed_K:
            raise ValueError(f"fixed_K={fixed_K} does not match checkpoint K={K}")
        if fixed_K is not None:
            K = fixed_K
            
        U = init_state["U_final"]
        h = init_state.get("H_final", None)

        iteration_list = init_state.get("iteration_list", [])
        update_category_list = init_state.get("update_category_list", [])
        prior_timing_list = init_state.get("prior_timing_list", [])
        likelihood_timing_list = init_state.get("likelihood_timing_list", [])
        update_timing_list = init_state.get("update_timing_list", [])
        update_records = init_state.get("update_records", [])
        proposed_rho_vals = init_state.get("proposed_rho_vals", [])
        proposed_prob_noise_vals = init_state.get("proposed_prob_noise_vals", [])
        proposed_softmax_beta_vals = init_state.get("proposed_softmax_beta_vals", [])

        print(f"⏪  Resuming from iteration {iteration_start:,d}")
        
    else:
        iteration_start = 0
        log_llk_current = -float("inf")
        target_iterations = num_iterations
        accepted_before_resume = 0
        num_acceptances = 0
    
        # Sample initial parameters
        rho = StatisticalUtils.rRprior(rho_prior, rng=rng)
        K = fixed_K if fixed_K is not None else K_prior
        
        # Noise model specific parameters
        if is_softmax_noise_model:
            # For softmax: epsilon_val is fixed, softmax_beta is updated (or ignored for uni_utility)
            softmax_beta = rng.gamma(softmax_beta_prior[0], 1.0 / softmax_beta_prior[1])
            prob_noise = None  # Not used for softmax
        else:
            # For queue_jump: prob_noise is sampled and updated
            prob_noise = StatisticalUtils.rPprior(noise_beta_prior, rng=rng)
    
        Sigma_rho = BasicUtils.build_Sigma_rho(K, rho)

        # Initialize U ~ N(0, Sigma_rho) for each item
        U = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma_rho, size=n)
        
        # Build partial order from U
        eta = StatisticalUtils.transform_U_to_eta(U)
        h = BasicUtils.generate_partial_order(eta)
        
        # Setup traces
        U_trace = []
        H_trace = []
        rho_trace = []
        prob_noise_trace = []
        softmax_beta_trace = []
        epsilon_trace = []
        K_trace = []
        acceptance_decisions = []
        acceptance_rates = []
        log_likelihood_currents = []
        log_likelihood_primes = []
        update_records = []

        proposed_rho_vals = []
        proposed_prob_noise_vals = []
        proposed_softmax_beta_vals = []
    
        iteration_list = []
        update_category_list = []
        prior_timing_list = []
        likelihood_timing_list = []
        update_timing_list = []

    # Correlation matrix
    Sigma_rho = BasicUtils.build_Sigma_rho(K, rho)
    
    # Calculate update schedule
    update_frequencies, update_schedule = calculate_dimension_proportional_weights(
        n_items=n,
        cycle_length=cycle_length,
        has_K_updates=update_K,
    )
    
    print(f"📊 Update Schedule Frequencies: {update_frequencies}")
    print(f"📋 Sequential update cycle length: {len(update_schedule)}")

    # Start MCMC
    log_llk_proposed = -float("inf")
    
    # Precompute progress intervals
    progress_intervals = set([0]) | set(int(target_iterations * frac) for frac in np.arange(0.05, 1.05, 0.05))

    def print_progress_bar(iteration, total):
        fraction = iteration / total
        bar_length = 40
        filled = int(bar_length * fraction)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\rProgress: [{bar}] {fraction*100:.1f}%  (Iteration {iteration}/{total})", end="")
        if iteration == total:
            print()
    
    def prepare_softmax_params(noise_opt, softmax_beta_val, epsilon_val):
        if noise_opt == "log_successors_queue_jump":
            return {"beta": softmax_beta_val, "epsilon": epsilon_val}
        elif noise_opt == "frontier_softmax_uni_utility":
            # For uniform utility, beta has no effect (Q=0), but we pass it for API consistency
            return {"beta": 1.0, "epsilon": epsilon_val}
        return None

    # Determine which noise parameter to use
    # For softmax: use fixed epsilon_val
    # For queue_jump: use sampled prob_noise
    def get_noise_param():
        if is_softmax_noise_model:
            return epsilon_val
        else:
            return prob_noise

    def compute_log_likelihood(
        U_input: np.ndarray,
        h_input: np.ndarray,
        noise_param_override: Optional[float] = None,
        softmax_beta_override: Optional[float] = None,
    ) -> float:
        effective_noise_param = (
            get_noise_param() if noise_param_override is None else noise_param_override
        )
        effective_softmax_beta = (
            softmax_beta if softmax_beta_override is None else softmax_beta_override
        )
        return HPO_LogLikelihoodCache_Optimized.calculate_log_likelihood_po_optimized(
            U=U_input,
            h=h_input,
            observed_orders=observed_orders,
            choice_sets=choice_sets,
            items=items,
            item_to_index=item_to_index,
            prob_noise=effective_noise_param,
            softmax_params=prepare_softmax_params(noise_option, effective_softmax_beta, epsilon_val),
            noise_option=noise_option,
        )
    
    # Calculate initial likelihood
    log_llk_current = compute_log_likelihood(U, h)

    for iteration in range(iteration_start + 1, target_iterations + 1):
        iteration_list.append(iteration)
        accepted_this_iter = False
        update_category = None
        total_prior_time = 0.0
        total_likelihood_time = 0.0
        update_type_timing = 0.0 
        
        # Get the current noise parameter
        noise_param = get_noise_param()
        
        cycle_position = (iteration - 1) % len(update_schedule)
        update_category = update_schedule[cycle_position]

        # ------------------------------------------------
        # Update: Rho
        # ------------------------------------------------
        if update_category == "rho":
            upd_start = time.time()
            delta = rng.uniform(dr, 1.0 / dr)
            rho_prime = 1.0 - (1.0 - rho) * delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho

            prior_start = time.time()
            Sigma_rho_prime = BasicUtils.build_Sigma_rho(K, rho_prime) 
            log_prior_current = StatisticalUtils.dRprior(rho, rho_prior) + StatisticalUtils.log_U_prior_optimized(U, rho, K)
            log_prior_proposed = StatisticalUtils.dRprior(rho_prime, rho_prior) + StatisticalUtils.log_U_prior_optimized(U, rho_prime, K)
            total_prior_time = time.time() - prior_start

            # Rho doesn't change likelihood
            log_llk_proposed = log_llk_current
            total_likelihood_time = 0.0

            log_accept = (log_prior_proposed) - (log_prior_current) - math.log(delta)
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                rho = rho_prime
                Sigma_rho = Sigma_rho_prime
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_rho_vals.append(rho_prime)
            update_type_timing = time.time() - upd_start

        # ------------------------------------------------
        # Update: Noise parameter
        # ------------------------------------------------
        elif update_category == "noise":
            upd_start = time.time()
            if noise_option in ("queue_jump", "weighted_queue_jump"):
                prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior, rng=rng)
                prior_start = time.time()
                lp_current = StatisticalUtils.dPprior(prob_noise, noise_beta_prior)
                lp_proposed = StatisticalUtils.dPprior(prob_noise_prime, noise_beta_prior)
                total_prior_time = time.time() - prior_start

                llk_start = time.time() 
                log_llk_proposed = HPO_LogLikelihoodCache_Optimized.calculate_log_likelihood_po_optimized(
                    U=U,
                    h=h,
                    observed_orders=observed_orders,
                    choice_sets=choice_sets,
                    items=items,
                    item_to_index=item_to_index,
                    prob_noise=prob_noise_prime,
                    softmax_params=None,
                    noise_option=noise_option,
                )
                total_likelihood_time = time.time() - llk_start

                # Likelihood-only MH ratio: proposal draws prob_noise from its prior,
                # so prior and proposal terms cancel in the acceptance ratio.
                log_accept = (log_llk_proposed - log_llk_current)
                accept_prob = min(1.0, math.exp(min(log_accept, 700)))
                if rng.random() < accept_prob:
                    prob_noise = prob_noise_prime
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                proposed_prob_noise_vals.append(prob_noise_prime)
                
            elif noise_option == "log_successors_queue_jump":
                prior_start = time.time()
                log_beta_current = math.log(max(softmax_beta, 0.01))
                rw_step = rng.normal(0, softmax_beta_stepsize)
                log_beta_proposed = log_beta_current + rw_step
                beta_proposed = math.exp(log_beta_proposed)
                
                shape, rate = softmax_beta_prior
                lp_current = (shape - 1) * log_beta_current - rate * softmax_beta
                lp_proposed = (shape - 1) * log_beta_proposed - rate * beta_proposed
                total_prior_time = time.time() - prior_start
                
                llk_start = time.time()
                softmax_params = {'beta': beta_proposed, 'epsilon': epsilon_val}
                log_llk_proposed = HPO_LogLikelihoodCache_Optimized.calculate_log_likelihood_po_optimized(
                    U=U,
                    h=h,
                    observed_orders=observed_orders,
                    choice_sets=choice_sets,
                    items=items,
                    item_to_index=item_to_index,
                    prob_noise=epsilon_val,
                    softmax_params=softmax_params,
                    noise_option=noise_option,
                )
                total_likelihood_time = time.time() - llk_start
                
                log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current) + log_beta_proposed - log_beta_current
                accept_prob = min(1.0, math.exp(min(log_accept, 700)))
                if rng.random() < accept_prob:
                    softmax_beta = beta_proposed
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                proposed_softmax_beta_vals.append(beta_proposed)
    
            update_type_timing = time.time() - upd_start

        # ------------------------------------------------
        # Update: U (latent utilities)
        # ------------------------------------------------
        elif update_category == "U":
            upd_start = time.time()
            
            # Pick a random row to update
            i = rng.integers(0, n)
            current_row = U[i, :].copy()
            
            # Random walk proposal: N(current_row, Sigma_rho)
            Sigma = BasicUtils.build_Sigma_rho(K, rho)
            proposed_row = rng.multivariate_normal(mean=current_row, cov=Sigma)

            U_prime = U.copy()
            U_prime[i, :] = proposed_row

            prior_start = time.time()
            lp_current = StatisticalUtils.log_U_prior_optimized(U, rho, K)
            lp_proposed = StatisticalUtils.log_U_prior_optimized(U_prime, rho, K)
            total_prior_time = time.time() - prior_start
            
            # Build new partial order
            llk_start = time.time()
            eta_prime = StatisticalUtils.transform_U_to_eta(U_prime)
            h_prime = BasicUtils.generate_partial_order(eta_prime)
            
            log_llk_proposed = HPO_LogLikelihoodCache_Optimized.calculate_log_likelihood_po_optimized(
                U=U_prime,
                h=h_prime,
                observed_orders=observed_orders,
                choice_sets=choice_sets,
                items=items,
                item_to_index=item_to_index,
                prob_noise=noise_param,
                softmax_params=prepare_softmax_params(noise_option, softmax_beta, epsilon_val),
                noise_option=noise_option,
            )
            total_likelihood_time = time.time() - llk_start

            log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current)
            accept_prob = min(1.0, math.exp(min(log_accept, 700)))
            if rng.random() < accept_prob:
                U = U_prime
                h = h_prime
                log_llk_current = log_llk_proposed
                accepted_this_iter = True
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)

            update_type_timing = time.time() - upd_start
            
        # ------------------------------------------------
        # Update: K (dimension)
        # ------------------------------------------------
        elif update_category == "K":
            update_category = "K_dim"
            upd_start = time.time()

            if K == 1:
                move = "up"
            else:
                move = "up" if rng.random() < 0.5 else "down"

            if move == "up":
                K_prime = K + 1
                col_ins = rng.integers(0, K_prime)
                Sigma_rho_prime = BasicUtils.build_Sigma_rho(K_prime, rho)

                # Sample new column from conditional distribution
                new_col = StatisticalUtils.sample_conditional_column(U, rho, rng=rng)
                U_prime = np.insert(U, col_ins, new_col, axis=1)

                prior_start = time.time()
                logK_current = StatisticalUtils.dKprior(K, K_prior)
                logK_proposed = StatisticalUtils.dKprior(K_prime, K_prior)
                lp_current = logK_current 
                lp_proposed = logK_proposed 
                total_prior_time = time.time() - prior_start

                llk_start = time.time()
                eta_prime = StatisticalUtils.transform_U_to_eta(U_prime)
                h_prime = BasicUtils.generate_partial_order(eta_prime)
                log_llk_proposed = HPO_LogLikelihoodCache_Optimized.calculate_log_likelihood_po_optimized(
                    U=U_prime,
                    h=h_prime,
                    observed_orders=observed_orders,
                    choice_sets=choice_sets,
                    items=items,
                    item_to_index=item_to_index,
                    prob_noise=noise_param,
                    softmax_params=prepare_softmax_params(noise_option, softmax_beta, epsilon_val),
                    noise_option=noise_option,
                )
                total_likelihood_time = time.time() - llk_start
                
                rho_fk = 1.0 if K == 1 else 0.5
                rho_bk = 0.5
                log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current) + math.log(rho_bk) - math.log(rho_fk)
                accept_prob = min(1.0, math.exp(min(log_accept, 700)))
                if rng.random() < accept_prob:
                    K = K_prime
                    U = U_prime
                    h = h_prime
                    Sigma_rho = Sigma_rho_prime
                    log_llk_current = log_llk_proposed
                    accepted_this_iter = True
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                update_type_timing = time.time() - upd_start
            else:
                # Move down
                K_prime = K - 1
                if K_prime < 1:
                    acceptance_decisions.append(0)
                else:
                    col_del = rng.integers(0, K)
                    Sigma_rho_prime = BasicUtils.build_Sigma_rho(K_prime, rho)
                    U_prime = np.delete(U, col_del, axis=1)

                    prior_start = time.time()
                    logK_current = StatisticalUtils.dKprior(K, K_prior)
                    logK_proposed = StatisticalUtils.dKprior(K_prime, K_prior)
                    lp_current = logK_current 
                    lp_proposed = logK_proposed
                    total_prior_time = time.time() - prior_start
                    
                    llk_start = time.time()
                    eta_prime = StatisticalUtils.transform_U_to_eta(U_prime)
                    h_prime = BasicUtils.generate_partial_order(eta_prime)
                    log_llk_proposed = HPO_LogLikelihoodCache_Optimized.calculate_log_likelihood_po_optimized(
                        U=U_prime,
                        h=h_prime,
                        observed_orders=observed_orders,
                        choice_sets=choice_sets,
                        items=items,
                        item_to_index=item_to_index,
                        prob_noise=noise_param,
                        softmax_params=prepare_softmax_params(noise_option, softmax_beta, epsilon_val),
                        noise_option=noise_option,
                    )
                    total_likelihood_time = time.time() - llk_start

                    # Proposal probability correction for reversible jump (reverse of up move)
                    # Forward (K→K-1): prob = 0.5 (can go up or down from K)
                    # Backward (K-1→K): prob = 1.0 if K-1=1, else 0.5
                    rho_fk = 0.5  
                    rho_bk = 1.0 if K_prime == 1 else 0.5
                    log_accept = (lp_proposed + log_llk_proposed) - (lp_current + log_llk_current) + math.log(rho_bk) - math.log(rho_fk)
                    accept_prob = min(1.0, math.exp(min(log_accept, 700)))
           
                    if rng.random() < accept_prob:
                        K = K_prime
                        U = U_prime
                        h = h_prime
                        Sigma_rho = Sigma_rho_prime
                        log_llk_current = log_llk_proposed
                        accepted_this_iter = True
                        num_acceptances += 1
                        acceptance_decisions.append(1)
                    else:
                        acceptance_decisions.append(0)

                update_type_timing = time.time() - upd_start 

        else:
            print(f"⚠️  Warning: Unknown update category '{update_category}' at iteration {iteration}")
            acceptance_decisions.append(0)
            update_type_timing = 0.0
            
        # Record iteration info
        accepted_run = num_acceptances - accepted_before_resume
        elapsed_run = iteration - iteration_start
        current_accept_rate = accepted_run / max(elapsed_run, 1)
        acceptance_rates.append(current_accept_rate)
        update_category_list.append(update_category)
        prior_timing_list.append(total_prior_time)
        likelihood_timing_list.append(total_likelihood_time)
        update_timing_list.append(update_type_timing)

        # Store traces every 100 iterations
        if iteration % 100 == 0:
            rho_trace.append(rho)
            if noise_option == "log_successors_queue_jump":
                softmax_beta_trace.append(softmax_beta)
                epsilon_trace.append(epsilon_val)
            else:
                prob_noise_trace.append(prob_noise)
                epsilon_trace.append(prob_noise)  # For queue_jump, epsilon = prob_noise
            K_trace.append(K)
            U_trace.append(U.copy())
            H_trace.append(h.copy())
            update_records.append((iteration, update_category, accepted_this_iter))

        if checkpoint_interval and checkpoint_callback:
            if iteration % checkpoint_interval == 0:
                state_dict = {
                    "iteration": iteration,
                    "rho_final": rho,
                    "K_final": K,
                    "U_final": U,
                    "H_final": h,
                    "log_likelihood_currents": log_likelihood_currents,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds")
                }
                if noise_option == "log_successors_queue_jump":
                    state_dict["softmax_beta_final"] = softmax_beta
                    state_dict["epsilon_final"] = epsilon_val
                else:
                    state_dict["prob_noise_final"] = prob_noise
                    state_dict["epsilon_final"] = prob_noise
                checkpoint_callback(iteration, state_dict)
                
        log_likelihood_currents.append(log_llk_current)
        log_likelihood_primes.append(log_llk_proposed)

        if iteration in progress_intervals:
            done = iteration - iteration_start
            print(f"Iteration {done}/{num_iterations} - Accept Rate: {current_accept_rate:.2%}")

    # Done
    overall_acceptance_rate = num_acceptances / num_iterations
    print_progress_bar(target_iterations, target_iterations)
    update_df = pd.DataFrame(update_records, columns=["iteration", "category", "accepted"])

    result_dict = {
        "rho_trace": rho_trace,
        "prob_noise_trace": prob_noise_trace,
        "softmax_beta_trace": softmax_beta_trace,
        "epsilon_trace": epsilon_trace,
        "K_trace": K_trace,

        "U_trace": U_trace,
        "H_trace": H_trace,

        "proposed_rho_vals": proposed_rho_vals,
        "proposed_prob_noise_vals": proposed_prob_noise_vals,
        "proposed_softmax_beta_vals": proposed_softmax_beta_vals,

        "acceptance_decisions": acceptance_decisions,
        "acceptance_rates": acceptance_rates,
        "overall_acceptance_rate": overall_acceptance_rate, 
        "log_likelihood_currents": log_likelihood_currents,
        "log_likelihood_primes": log_likelihood_primes,
        "num_acceptances": num_acceptances,

        # Final state
        "rho_final": rho,
        "prob_noise_final": prob_noise,
        "softmax_beta_final": softmax_beta,
        "epsilon_final": epsilon_val if noise_option == "log_successors_queue_jump" else prob_noise,
        "K_final": K,
        "U_final": U,
        "H_final": h,

        "update_df": update_df,

        # Timing
        "total_prior_timing": total_prior_time,
        "total_likelihood_timing": total_likelihood_time,
        "update_type_timing": update_type_timing,

        # Per-iteration lists
        "iteration_list": iteration_list,
        "update_category_list": update_category_list,
        "prior_timing_list": prior_timing_list,
        "likelihood_timing_list": likelihood_timing_list,
        "update_timing_list": update_timing_list,
        "iteration": target_iterations,
    }
    return result_dict


def mcmc_simulation_hpo_k_optim(
    num_iterations: int,
    M0: Optional[List[int]] = None,
    assessors: Optional[List[Any]] = None,
    M_a_dict: Optional[Dict[Any, List[int]]] = None,
    O_a_i_dict: Optional[Dict[Any, List[List[int]]]] = None,
    observed_orders: Optional[Dict[Any, List[List[int]]]] = None,
    dr: float = 0.95,
    drrt: Optional[float] = None,
    noise_option: str = "queue_jump",
    rho_prior: float = 1.0,
    noise_beta_prior: float = 1.0,
    K_prior: int = 2,
    fixed_K: Optional[int] = None,
    random_seed: int = 42,
    cycle_length: int = 1000,
    epsilon: float = 0.01,
    softmax_lambda: Optional[float] = None,
    softmax_beta_prior: tuple = (2.0, 2.0),
    softmax_beta_stepsize: float = 0.3,
    init_state: Optional[Dict[str, Any]] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_callback=None,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for the pre-refactor hierarchical API.

    The current codebase uses `mcmc_simulation_po`, which infers a single partial order.
    Old notebooks still import `mcmc_simulation_hpo_k_optim` and pass a one-assessor
    hierarchical payload. This wrapper accepts that older call shape, unwraps the single
    assessor, runs the single-PO sampler, then repacks `U_trace`/`H_trace` and the final
    matrices into the old dict-of-assessor format expected by the notebooks.

    Notes:
    - Only the single-assessor case is supported.
    - `drrt` and `softmax_lambda` are accepted for compatibility but ignored.
    """
    del drrt, softmax_lambda

    if M_a_dict is None or O_a_i_dict is None or observed_orders is None:
        raise ValueError(
            "Compatibility wrapper requires M_a_dict, O_a_i_dict, and observed_orders."
        )

    if assessors is None:
        assessors = list(M_a_dict.keys())

    if len(assessors) != 1:
        raise NotImplementedError(
            "mcmc_simulation_hpo_k_optim compatibility wrapper only supports a single assessor. "
            "Use mcmc_simulation_po directly for the current single-PO workflow."
        )

    assessor_id = assessors[0]
    if assessor_id not in M_a_dict or assessor_id not in O_a_i_dict or assessor_id not in observed_orders:
        raise KeyError(f"Assessor {assessor_id!r} missing from compatibility-wrapper inputs.")

    po_init_state = None
    if init_state is not None:
        po_init_state = dict(init_state)
        if "U_trace" in po_init_state:
            po_init_state["U_trace"] = [
                entry[assessor_id] if isinstance(entry, dict) else entry
                for entry in po_init_state["U_trace"]
            ]
        if "H_trace" in po_init_state:
            po_init_state["H_trace"] = [
                entry[assessor_id] if isinstance(entry, dict) else entry
                for entry in po_init_state["H_trace"]
            ]
        if isinstance(po_init_state.get("U_final"), dict):
            po_init_state["U_final"] = po_init_state["U_final"][assessor_id]
        if isinstance(po_init_state.get("H_final"), dict):
            po_init_state["H_final"] = po_init_state["H_final"][assessor_id]

    result = mcmc_simulation_po(
        num_iterations=num_iterations,
        items=list(M_a_dict[assessor_id]),
        choice_sets=O_a_i_dict[assessor_id],
        observed_orders=observed_orders[assessor_id],
        dr=dr,
        noise_option=noise_option,
        rho_prior=rho_prior,
        noise_beta_prior=noise_beta_prior,
        K_prior=K_prior,
        fixed_K=fixed_K,
        random_seed=random_seed,
        cycle_length=cycle_length,
        epsilon=epsilon,
        softmax_beta_prior=softmax_beta_prior,
        softmax_beta_stepsize=softmax_beta_stepsize,
        init_state=po_init_state,
        checkpoint_interval=checkpoint_interval,
        checkpoint_callback=checkpoint_callback,
    )

    compat = dict(result)
    compat["M0"] = M0
    compat["assessors"] = list(assessors)
    compat["M_a_dict"] = M_a_dict
    compat["O_a_i_dict"] = O_a_i_dict
    compat["observed_orders"] = observed_orders
    compat["U_trace"] = [{assessor_id: U} for U in result.get("U_trace", [])]
    compat["H_trace"] = [{assessor_id: H} for H in result.get("H_trace", [])]
    compat["U_final"] = {assessor_id: result.get("U_final")}
    compat["H_final"] = {assessor_id: result.get("H_final")}
    return compat
