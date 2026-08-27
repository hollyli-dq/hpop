"""Stage 6A — held-out diagnostic scoring.

These are **diagnostics, not identifiability claims**. Only one is a hard criterion:
the true model must beat a wrong *antichain* ``U`` on held-out NLL. If that failed, the
likelihood would be insensitive to the very structure it is supposed to encode.

For every other perturbation the paired per-block NLL difference is reported with its
mean, standard error, median and the fraction of blocks favouring the truth — a small
mean with a wide spread is reported as such, never rounded up into a claim. Formal
scalar-parameter recovery belongs to Stage 6B.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_rfs import RecurrentRFSParameters, logit
from hpop.mcmc_original.recurrent_rfs import recurrent_rfs_log_likelihood as _ll
from hpop.mcmc_original.recurrent_synthetic import U_ANTICHAIN, U_TRUE

__all__ = ["per_step_nll", "held_out_diagnostics", "generator_likelihood_parity"]


def per_step_nll(blocks, u, params: RecurrentRFSParameters) -> np.ndarray:
    """Per-block mean negative log-likelihood per step."""
    out = []
    for block in blocks:
        value = _ll(block.roles, u, params.beta, params.epsilon, params.shared_omega,
                    params.lambda_rep, params.lambda_back)
        out.append(-value / len(block.roles))
    return np.array(out)


def held_out_diagnostics(blocks, u_true=None, params=None) -> dict:
    u_true = U_TRUE if u_true is None else u_true
    if params is None:
        from hpop.mcmc_original.recurrent_synthetic import TRUE_PARAMETERS
        params = TRUE_PARAMETERS

    def swap(**kw):
        return RecurrentRFSParameters(
            beta=kw.get("beta", params.beta), epsilon=kw.get("epsilon", params.epsilon),
            shared_omega=kw.get("shared_omega", params.shared_omega),
            lambda_rep=kw.get("lambda_rep", params.lambda_rep),
            lambda_back=kw.get("lambda_back", params.lambda_back))

    baseline = per_step_nll(blocks, u_true, params)
    variants = {
        "wrong_antichain_U": (U_ANTICHAIN, params),
        "beta_0": (u_true, swap(beta=0.0)),
        "kappa_0.05": (u_true, swap(shared_omega=logit(0.05))),
        "lambda_rep_0": (u_true, swap(lambda_rep=0.0)),
        "lambda_back_0": (u_true, swap(lambda_back=0.0)),
        "lambda_rep_and_back_0": (u_true, swap(lambda_rep=0.0, lambda_back=0.0)),
    }
    rows = {}
    for name, (u_alt, p_alt) in variants.items():
        alt = per_step_nll(blocks, u_alt, p_alt)
        diff = alt - baseline                         # positive => truth is better
        rows[name] = {
            "nll_true": float(baseline.mean()), "nll_variant": float(alt.mean()),
            "paired_mean_difference": float(diff.mean()),
            "standard_error": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
            "median_difference": float(np.median(diff)),
            "fraction_favouring_true": float((diff > 0).mean()),
            "true_model_better_on_average": bool(diff.mean() > 0),
        }
    return {"n_blocks": len(blocks), "nll_true_parameters": float(baseline.mean()),
            "variants": rows,
            "hard_criterion_true_beats_antichain":
                bool(rows["wrong_antichain_U"]["nll_variant"] > float(baseline.mean()))}


def generator_likelihood_parity(blocks, u, params) -> dict:
    """Replay each block independently and compare against what the generator used."""
    worst_ll, worst_q, worst_p = 0.0, 0.0, 0.0
    for block in blocks:
        value, trace = _ll(block.roles, u, params.beta, params.epsilon, params.shared_omega,
                           params.lambda_rep, params.lambda_back, return_trace=True)
        worst_ll = max(worst_ll, abs(value - block.generator_log_likelihood))
        for step, record in enumerate(trace):
            worst_q = max(worst_q,
                          float(np.abs(record["q_before"] - np.array(block.q_before[step])).max()),
                          float(np.abs(record["q_after"] - np.array(block.q_after[step])).max()))
            worst_p = max(worst_p, abs(np.log(record["observed_probability"])
                                       - block.step_log_probabilities[step]))
    return {"n_blocks": len(blocks), "max_abs_log_likelihood_difference": worst_ll,
            "max_abs_q_difference": worst_q, "max_abs_step_logp_difference": worst_p}


def identifiability_curvature(blocks, u, params, delta: float = 0.10) -> dict:
    """Block-level Fisher curvature per parameter — what predicts Stage-6B recoverability.

    Why block-level and not per-step: ``omega`` acts mainly on how the validity state
    ``q`` evolves, and that channel only appears once a whole sequence is re-scored.
    Evaluating the same observed sequence under a perturbed ``omega`` produces a
    different internal ``q`` recursion, so the block log-likelihood picks it up; a
    per-step KL at fixed ``q`` does not, and understates ``omega`` by ~25x.

    Uses a symmetric second difference, ``KL ~ 0.5 * I * delta^2``, giving the expected
    posterior sd as ``1 / sqrt(I * n_blocks)``.

    This is the check the exposure audit cannot make: the audit counts *opportunities*
    for a parameter to act, this measures how sharply the likelihood responds.
    """
    roles = [b.roles for b in blocks]

    def swap(**kw):
        return RecurrentRFSParameters(
            kw.get("beta", params.beta), params.epsilon,
            kw.get("shared_omega", params.shared_omega),
            kw.get("lambda_rep", params.lambda_rep),
            kw.get("lambda_back", params.lambda_back))

    def block_ll(p):
        return np.array([_ll(y, u, p.beta, p.epsilon, p.shared_omega,
                             p.lambda_rep, p.lambda_back) for y in roles])

    base = block_ll(params)
    out = {}
    for name, key, value in (("beta", "beta", params.beta),
                             ("lambda_rep", "lambda_rep", params.lambda_rep),
                             ("lambda_back", "lambda_back", params.lambda_back),
                             ("omega", "shared_omega", params.shared_omega)):
        up = block_ll(swap(**{key: value + delta}))
        down = block_ll(swap(**{key: value - delta}))
        kl = float(np.mean(base - 0.5 * (up + down)))
        fisher = max(2.0 * kl / delta ** 2, 1e-12)
        sd = float(1.0 / np.sqrt(fisher * len(roles)))
        out[name] = {"true_value": float(value), "delta": delta, "kl_per_block": kl,
                     "fisher_per_block": fisher, "expected_posterior_sd": sd,
                     "true_over_sd": float(abs(value) / sd)}
    weakest = min(out, key=lambda k: out[k]["true_over_sd"])
    return {"n_blocks": len(roles), "by_parameter": out, "weakest_parameter": weakest,
            "weakest_true_over_sd": out[weakest]["true_over_sd"]}


def profile_log_likelihood(blocks, u, params, parameter: str, grid) -> dict:
    """Exact 1-D log-likelihood profile: vary one parameter, hold the rest at truth.

    This is the object Stage-6B MCMC must be checked against. It is computed by brute
    force over a grid, so it has no sampler in it and cannot inherit a sampler's bug.

    Deliberately a *likelihood* profile, not a posterior one: priors for ``omega``,
    ``lambda_rep`` and ``lambda_back`` are not registered anywhere yet, and inventing
    them here would silently fix a modelling choice that belongs to Stage 6B.

    Every candidate value re-scores each block from ``q_0 = 0``, so a parameter that
    acts through the validity recursion (``omega``) is handled correctly.
    """
    key = {"beta": "beta", "omega": "shared_omega", "lambda_rep": "lambda_rep",
           "lambda_back": "lambda_back"}[parameter]
    roles = [b.roles for b in blocks]

    def at(value):
        kw = {"beta": params.beta, "epsilon": params.epsilon,
              "shared_omega": params.shared_omega, "lambda_rep": params.lambda_rep,
              "lambda_back": params.lambda_back}
        kw[key] = value
        p = RecurrentRFSParameters(**kw)
        return float(sum(_ll(y, u, p.beta, p.epsilon, p.shared_omega,
                             p.lambda_rep, p.lambda_back) for y in roles))

    grid = np.asarray(grid, dtype=float)
    values = np.array([at(g) for g in grid])
    best = int(np.argmax(values))
    true_value = {"beta": params.beta, "omega": params.shared_omega,
                  "lambda_rep": params.lambda_rep, "lambda_back": params.lambda_back}[parameter]

    # quadratic fit near the peak -> curvature and an implied standard error
    lo, hi = max(best - 3, 0), min(best + 4, len(grid))
    sd = float("nan")
    if hi - lo >= 3:
        c = np.polyfit(grid[lo:hi], values[lo:hi], 2)
        if c[0] < 0:
            sd = float(1.0 / np.sqrt(-2.0 * c[0]))
    return {"parameter": parameter, "grid": grid.tolist(),
            "log_likelihood": values.tolist(), "true_value": float(true_value),
            "argmax": float(grid[best]), "max_log_likelihood": float(values[best]),
            "offset_from_truth": float(grid[best] - true_value),
            "implied_sd_from_curvature": sd,
            "log_likelihood_at_truth": at(true_value),
            "unimodal": bool(np.all(np.diff(values[:best + 1]) >= -1e-9)
                             and np.all(np.diff(values[best:]) <= 1e-9))}
