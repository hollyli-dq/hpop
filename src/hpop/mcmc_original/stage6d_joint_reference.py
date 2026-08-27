"""Stage 6D1 — an independent continuous-latent reference for the small joint model.

The reference shares no code path with the Stage 6D transition kernel. It evaluates the
same direct target, but reaches it by scrambled-Sobol importance sampling in **prior
coordinates**, not by sampling.

## Why non-centred prior coordinates

Draw the QMC point in the space the prior is easy to invert:

    u_rho              -> rho      by the exact prior inverse CDF
    z_U in R^{m x d}   -> Z        by the standard-normal inverse CDF
    u_beta, ...        -> scalars  by their exact prior inverse CDFs

then build the latent matrix **conditionally**,

    U = Z L(rho)^T,    L(rho) L(rho)^T = Sigma_rho.

Because the proposal is then *exactly* the joint prior, the unnormalised importance
weight collapses to the likelihood alone,

    w = p_RFS(x | h(U), beta, omega, lambda_rep, lambda_back, epsilon),

with no prior densities left to cancel by hand — which is the point: it removes the
Gaussian determinant from the weight entirely, so a determinant error in the *sampler*
cannot hide behind the same error in the *reference*. The centred density is nevertheless
checked against this construction separately (`centred_matches_non_centred`), so the two
routes are known to agree rather than assumed to.

`U` is continuous throughout. `H = h(U)` is computed from each draw with the production
mapping and used only for reporting and for the discrete comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.stats import qmc

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import (
    PRIORS, cached_batch_log_likelihood,
)
from hpop.mcmc_original.sampler_u import log_u_prior, sigma_rho_matrix
from hpop.mcmc_original.stage6d_frozen import RHO_UPPER, log_det_sigma_rho

__all__ = [
    "SmallModel", "small_model", "prior_inverse_cdf", "build_state", "sobol_points",
    "qmc_replicate", "replicate_summary", "combine_replicates",
    "centred_matches_non_centred", "induced_h_labels", "QMC_DIMENSION",
]

# rho, then m*d latent normals, then the four scalars. Fixed before any comparison.
def qmc_dimension(m: int, d: int) -> int:
    return 1 + m * d + 4


QMC_DIMENSION = qmc_dimension


@dataclass(frozen=True)
class SmallModel:
    """The registered Stage 6D1 problem. Dimensions frozen before any MCMC comparison."""
    m: int
    d: int
    n_blocks: int
    T: int
    u_true: np.ndarray
    roles: np.ndarray
    epsilon: float
    truth: dict

    @property
    def n_skills(self) -> int:
        return 1

    @property
    def qmc_dimension(self) -> int:
        return qmc_dimension(self.m, self.d)


def small_model(seed: int = 20260812) -> SmallModel:
    """The registered Stage 6D1 problem: m = 3, d = 2, one U, 3 oracle blocks of T = 5.

    `U_TRUE` is chosen so the induced order is nondegenerate: role 0 dominates both
    others, which remain incomparable. That is neither an antichain nor a total order, so
    the reference has genuine structural uncertainty to represent.

    **Why the corpus is deliberately small.** These dimensions were chosen for *reference
    quality*, before any Stage 6D MCMC comparison existed, and are frozen here. Prior
    importance sampling degrades as the likelihood sharpens relative to the priors: at 30
    blocks of T = 8 the relative ESS was 0.005 with a single point holding 10% of the
    weight, which §10.2 forbids accepting. At 3 blocks of T = 5 the relative ESS is 0.081,
    stable across point counts, and the maximum normalised weight falls as 1/n.

    The small corpus is a feature rather than a concession: it leaves the induced-order
    posterior genuinely uncertain — six of the nineteen labelled posets on three elements
    carry more than 1% mass — so this problem exercises structural *mixing*, which the
    full Stage 6C corpus could not (there the poset posterior was a point mass).
    """
    m, d, n_blocks, T = 3, 2, 3, 5
    u_true = np.array([[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    rng = np.random.default_rng(seed)
    roles = rng.integers(0, m, size=(n_blocks, T)).astype(int)
    truth = {"beta": 1.5, "omega": 1.7346, "lambda_rep": 0.8, "lambda_back": 0.25}
    return SmallModel(m=m, d=d, n_blocks=n_blocks, T=T, u_true=u_true, roles=roles,
                      epsilon=0.02, truth=truth)


# ------------------------------------------------------------------ prior inverse CDFs
def prior_inverse_cdf(name: str, u):
    """Exact inverse CDF of a registered prior, evaluated on uniforms in (0, 1)."""
    u = np.asarray(u, dtype=float)
    if name == "rho":
        # Beta(1, 1) truncated to (0, RHO_UPPER) and renormalised == Uniform(0, RHO_UPPER)
        return u * RHO_UPPER
    spec = PRIORS[name]
    if spec["family"] == "gamma":
        return stats.gamma.ppf(u, a=spec["shape"], scale=1.0 / spec["rate"])
    return stats.norm.ppf(u, loc=spec["mean"], scale=spec["sd"])


def build_state(points: np.ndarray, model: SmallModel) -> dict:
    """Map QMC uniforms to `(rho, U, scalars)` through the non-centred construction."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n, m, d = points.shape[0], model.m, model.d

    rho = prior_inverse_cdf("rho", points[:, 0])
    z = stats.norm.ppf(points[:, 1:1 + m * d]).reshape(n, m, d)

    # U = Z L(rho)^T, one Cholesky per distinct rho. Vectorised by sorting into a small
    # number of groups would help; at these sizes the direct loop is clear and fast enough.
    u = np.empty((n, m, d))
    for i in range(n):
        chol = np.linalg.cholesky(sigma_rho_matrix(d, float(rho[i])))
        u[i] = z[i] @ chol.T

    offset = 1 + m * d
    scalars = {name: prior_inverse_cdf(name, points[:, offset + k])
               for k, name in enumerate(("beta", "omega", "lambda_rep", "lambda_back"))}
    return {"rho": rho, "z": z, "u": u, **scalars}


def centred_matches_non_centred(u: np.ndarray, z: np.ndarray, rho: float) -> float:
    """|log p_U(U) - [log p_Z(Z) - (m/2) log|Sigma_rho|]|, which must be ~0.

    The change of variables for `U = Z L^T` is `log p_U(U) = log p_Z(Z) - m log|det L|`
    and `2 log|det L| = log|Sigma_rho|`. Checking it pins the non-centred construction
    against the centred density the sampler actually uses.
    """
    u = np.asarray(u, dtype=float)
    z = np.asarray(z, dtype=float)
    m, d = u.shape
    centred = log_u_prior(u, rho)
    non_centred = (float(stats.norm.logpdf(z).sum())
                   - 0.5 * m * log_det_sigma_rho(d, rho))
    return abs(centred - non_centred)


# ------------------------------------------------------------------------- likelihood
def log_likelihood_batch(state: dict, model: SmallModel, chunk_report=None) -> np.ndarray:
    """Complete recurrent log likelihood at every QMC point, by full replay."""
    n = state["rho"].shape[0]
    out = np.empty(n)
    for i in range(n):
        features = vectorized_state_features(model.roles, state["u"][i],
                                             float(state["omega"][i]))
        out[i] = cached_batch_log_likelihood(
            features, float(state["beta"][i]), model.epsilon,
            float(state["lambda_rep"][i]), float(state["lambda_back"][i]))
        if chunk_report is not None and (i + 1) % chunk_report == 0:
            print(f"    {i + 1:,}/{n:,}", flush=True)
    return out


def induced_h_labels(u_draws: np.ndarray) -> tuple[np.ndarray, list[bytes], np.ndarray]:
    """Canonical `H` label per draw, plus the closure stack of the distinct labels."""
    closures = np.array([precedence_from_u(u) for u in u_draws])
    keys = [c.tobytes() for c in closures]
    index: dict[bytes, int] = {}
    labels = np.empty(len(keys), dtype=np.int32)
    order: list[bytes] = []
    for i, key in enumerate(keys):
        if key not in index:
            index[key] = len(order)
            order.append(key)
        labels[i] = index[key]
    unique = np.array([closures[keys.index(k)] for k in order])
    return labels, order, unique


# --------------------------------------------------------------------------- replicate
def sobol_points(n_points: int, dimension: int, seed: int) -> np.ndarray:
    """One scrambled Sobol replicate, on the open unit cube."""
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    points = engine.random(n_points)
    # Sobol can emit exact 0 or 1; inverse CDFs need the open cube.
    return np.clip(points, 1e-12, 1.0 - 1e-12)


def qmc_replicate(model: SmallModel, n_points: int, seed: int,
                  progress: bool = False) -> dict:
    """One independent scrambled replicate: draws, weights, and its own quality metrics."""
    points = sobol_points(n_points, model.qmc_dimension, seed)
    state = build_state(points, model)
    log_w = log_likelihood_batch(state, model,
                                 chunk_report=n_points // 8 if progress else None)

    shift = float(np.max(log_w))
    w = np.exp(log_w - shift)
    total = float(w.sum())
    normalised = w / total
    ess = 1.0 / float((normalised ** 2).sum())
    return {
        "seed": seed, "n_points": int(n_points),
        "log_evidence": shift + math.log(total / n_points),
        "ess": ess, "relative_ess": ess / n_points,
        "max_normalised_weight": float(normalised.max()),
        "weights": normalised, "log_weights": log_w,
        "rho": state["rho"], "u": state["u"],
        **{k: state[k] for k in ("beta", "omega", "lambda_rep", "lambda_back")},
    }


def _weighted_summary(values, weights) -> dict:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = float((weights * values).sum())
    var = float((weights * (values - mean) ** 2).sum())
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    quantile = lambda q: float(np.interp(q, cdf, values[order]))   # noqa: E731
    return {"mean": mean, "sd": math.sqrt(max(var, 0.0)), "median": quantile(0.5),
            "q025": quantile(0.025), "q975": quantile(0.975)}


def replicate_summary(replicate: dict, model: SmallModel) -> dict:
    """Everything §10.1 asks a replicate to report."""
    w = replicate["weights"]
    labels, keys, closures = induced_h_labels(replicate["u"])
    n_states = len(keys)
    h_probability = np.zeros(n_states)
    np.add.at(h_probability, labels, w)

    flat = closures.reshape(n_states, -1).astype(float)
    relation_marginal = h_probability @ flat
    relation_counts = flat.sum(axis=1)
    max_pairs = model.m * (model.m - 1) // 2
    count_distribution = np.zeros(max_pairs + 1)
    np.add.at(count_distribution, relation_counts.astype(int), h_probability)

    scalars = {name: _weighted_summary(replicate[name], w)
               for name in ("rho", "beta", "omega", "lambda_rep", "lambda_back")}

    coordinates = np.column_stack([replicate[n] for n in
                                   ("rho", "beta", "omega", "lambda_rep", "lambda_back")]
                                  + [relation_counts[labels]])
    means = (w[:, None] * coordinates).sum(axis=0)
    centred = coordinates - means
    cov = (w[:, None, None] * centred[:, :, None] * centred[:, None, :]).sum(axis=0)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(sd, sd)

    return {
        "log_evidence": replicate["log_evidence"], "ess": replicate["ess"],
        "relative_ess": replicate["relative_ess"],
        "max_normalised_weight": replicate["max_normalised_weight"],
        "n_points": replicate["n_points"], "seed": replicate["seed"],
        "n_induced_h_states": n_states,
        "h_probability": h_probability, "h_keys": keys,
        "relation_marginal": relation_marginal,
        "relation_count_distribution": count_distribution,
        "scalars": scalars,
        "correlation": corr,
        "correlation_names": ["rho", "beta", "omega", "lambda_rep", "lambda_back",
                              "relation_count"],
    }


def rqmc_standard_error(replicate_estimates: np.ndarray) -> dict:
    """RQMC standard error of the **averaged** reference, per coordinate.

        rqmc_se = sd(replicate_estimates, ddof=1) / sqrt(R)

    This is the statistic that governs how precisely the frozen reference is known. It is
    NOT the maximum spread across replicates: that maximum estimates the dispersion of a
    single replicate, so it does not shrink as `R` grows and is not an uncertainty for the
    quantity actually used downstream (the replicate mean). Independent scrambles make the
    per-replicate estimates iid, which is exactly what licenses this formula.
    """
    estimates = np.atleast_2d(np.asarray(replicate_estimates, dtype=float))
    n_replicates = estimates.shape[0]
    if n_replicates < 2:
        raise ValueError("rqmc_se needs at least two independent scrambles")
    se = estimates.std(axis=0, ddof=1) / math.sqrt(n_replicates)
    half_width = float(stats.t.ppf(0.975, n_replicates - 1)) * se
    return {"standard_error": se, "half_width_95": half_width,
            "max_standard_error": float(np.max(se)),
            "max_half_width_95": float(np.max(half_width)),
            "n_replicates": int(n_replicates),
            "t_multiplier": float(stats.t.ppf(0.975, n_replicates - 1))}


def combine_replicates(summaries: list[dict]) -> dict:
    """Across-replicate stability — the evidence that the reference may be frozen."""
    log_z = np.array([s["log_evidence"] for s in summaries])
    ess = np.array([s["ess"] for s in summaries])
    rel = np.array([s["relative_ess"] for s in summaries])
    mx = np.array([s["max_normalised_weight"] for s in summaries])

    names = ("rho", "beta", "omega", "lambda_rep", "lambda_back")
    scalar_spread = {}
    for n in names:
        means = np.array([s["scalars"][n]["mean"] for s in summaries])
        sds = np.array([s["scalars"][n]["sd"] for s in summaries])
        scalar_spread[n] = {"mean_of_means": float(means.mean()),
                            "sd_across_replicates": float(means.std(ddof=1)),
                            "range": float(means.max() - means.min()),
                            "mean_of_sds": float(sds.mean())}

    # H probabilities are indexed per replicate; align on the union of keys.
    union: list[bytes] = []
    for s in summaries:
        for k in s["h_keys"]:
            if k not in union:
                union.append(k)
    aligned = np.zeros((len(summaries), len(union)))
    for i, s in enumerate(summaries):
        for k, p in zip(s["h_keys"], s["h_probability"]):
            aligned[i, union.index(k)] = p
    max_h_spread = float(np.abs(aligned - aligned.mean(axis=0)).max())
    pooled_h = aligned.mean(axis=0)

    relation = np.array([s["relation_marginal"] for s in summaries])

    # ---- primary precision: the standard error of the AVERAGED reference -------------
    h_precision = rqmc_standard_error(aligned)
    relation_precision = rqmc_standard_error(relation)
    scalar_means = np.array([[s["scalars"][n]["mean"] for n in names]
                             for s in summaries])
    scalar_precision = rqmc_standard_error(scalar_means)

    # ---- secondary dispersion: per-replicate departure from the replicate mean -------
    mean_h = aligned.mean(axis=0)
    replicate_tv = [0.5 * float(np.abs(row / row.sum() - mean_h / mean_h.sum()).sum())
                    for row in aligned]
    mean_relation = relation.mean(axis=0)
    replicate_relation_departure = float(np.abs(relation - mean_relation).max())

    return {
        "n_replicates": len(summaries),
        "precision": {
            "h_probability": {k: v for k, v in h_precision.items()
                              if k.startswith("max") or k in ("n_replicates",
                                                              "t_multiplier")},
            "relation_marginal": {k: v for k, v in relation_precision.items()
                                  if k.startswith("max")},
            "scalar_means": {k: v for k, v in scalar_precision.items()
                             if k.startswith("max")},
            "h_probability_se": h_precision["standard_error"],
            "h_probability_half_width_95": h_precision["half_width_95"],
            "relation_marginal_se": relation_precision["standard_error"],
            "max_structural_half_width_95": max(
                h_precision["max_half_width_95"],
                relation_precision["max_half_width_95"]),
            "definition": "rqmc_se = sd(replicate_estimates, ddof=1)/sqrt(R); the "
                          "half-width is t(0.975, R-1) * rqmc_se",
        },
        "replicate_dispersion": {
            "max_h_total_variation_from_mean": float(np.max(replicate_tv)),
            "per_replicate_h_total_variation": replicate_tv,
            "max_relation_departure_from_mean": replicate_relation_departure,
        },
        "per_replicate_h_probability": aligned,
        "per_replicate_relation_marginal": relation,
        "log_evidence": {"mean": float(log_z.mean()), "sd": float(log_z.std(ddof=1)),
                         "range": float(log_z.max() - log_z.min()),
                         "values": log_z.tolist()},
        "ess": {"min": float(ess.min()), "mean": float(ess.mean()),
                "values": ess.tolist()},
        "relative_ess": {"min": float(rel.min()), "mean": float(rel.mean())},
        "max_normalised_weight": {"max": float(mx.max()), "mean": float(mx.mean())},
        "scalar_spread": scalar_spread,
        "max_h_probability_spread": max_h_spread,
        "max_relation_marginal_spread": float(
            np.abs(relation - relation.mean(axis=0)).max()),
        "pooled_h_probability": pooled_h, "h_keys": union,
        "pooled_relation_marginal": relation.mean(axis=0),
    }
