"""Stage 6E1B — the mixed unknown-boundary reference: scrambled QMC over the continuous
coordinates, exact enumeration over the discrete ones.

This is the Stage 6D1 strategy with one addition. Stage 6D1 drew `(rho, U, scalars)` from
the joint prior by inverse-CDF in **prior coordinates**, so the importance weight collapsed
to the likelihood alone. Stage 6E1B does the same, but the "likelihood" of a draw is now
the *marginal* segmentation likelihood

    Z_n(theta) = sum over every legal (S_n, z_n) of w(S_n, z_n | theta)

which is available in closed form because the legal set is small and enumerable. The
unnormalised weight of a QMC point is therefore

    w(theta) = prod_n Z_n(theta),

and the reference targets the complete mixed posterior
`p(S, z, U, rho, beta, omega, lambda_rep, lambda_back | x)`.

`pi` and `P` are held **fixed** for this reference, which §9 permits. They are not
tractable to add: their conjugate update is defined given the *sampled* labels, so putting
them in the QMC construction would need a Dirichlet inverse-CDF in the proposal and would
add `K` more nearly-unidentified dimensions for no gain in what the reference is meant to
test. They are inferred in Stage 6E2, where the Stage 3 update is the object under test.

## Two estimators of the same reference, both reported

Given a QMC point, the conditional `p(S_n, z_n | theta, x)` is known *exactly* — it is the
enumerated weight vector normalised by `Z_n`. So the reference offers two estimators of any
segmentation functional `f`:

* **conditional** (Rao-Blackwellised): `sum_i w_i * E[f | theta_i]`, using the exact
  conditional;
* **sampled**: `sum_i w_i * f(S_i, z_i)` with one `(S_i, z_i)` drawn from that conditional,
  which is §9's iid-equivalent construction.

They estimate the same quantity; the conditional one has strictly lower variance, because
the sampled one adds a multinomial draw on top of it. The conditional estimator is
registered as **primary** — a reference must be more precise than the 0.01 budget it feeds,
and the sampled estimator's extra variance is pure noise added to the reference side. The
sampled estimator is computed and reported alongside, against the same gates, so §9's
construction is carried out and can be seen to agree rather than being replaced.

## The label-exchangeability question, settled by construction

If `pi` were uniform and `P` symmetric, permuting the skill labels together with the `U_k`
would leave this posterior invariant, and per-skill summaries would be meaningless. The
registered `pi` and `P` for this reference are **deliberately asymmetric** with distinct
rows, so no label permutation is a symmetry of the target and per-skill comparison is
well posed. `label_permutation_audit` measures how far from invariant the target actually
is rather than asserting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from scipy.stats import qmc

from hpop.mcmc_original.latent_poset import precedence_from_u
from hpop.mcmc_original.recurrent_joint_scalar_mcmc import vectorized_state_features
from hpop.mcmc_original.recurrent_scalar_posterior import PRIORS
from hpop.mcmc_original.sampler_u import sigma_rho_matrix
from hpop.mcmc_original.stage6d_frozen import RHO_UPPER
from hpop.mcmc_original.stage6d_joint_reference import (
    prior_inverse_cdf, rqmc_standard_error, sobol_points,
)
from hpop.mcmc_original.stage6e_exact import enumerate_states
from hpop.mcmc_original.stage6e_frozen import (
    DELTA_B, MAX_BLOCK_WIDTH, MIN_BLOCK_WIDTH,
)

__all__ = [
    "MixedModel", "per_block_log_likelihood", "build_mixed_state", "mixed_replicate",
    "mixed_replicate_summary", "combine_mixed_replicates", "label_permutation_audit",
    "h_label_of", "SCALAR_NAMES",
]

SCALAR_NAMES = ("beta", "omega", "lambda_rep", "lambda_back")

# Bit weights for packing an m x m closure into one integer label. m*m must stay under 63.
_BIT_WEIGHTS = (1 << np.arange(62, dtype=np.int64))


def _decode_closure(code: int, m: int) -> np.ndarray:
    """Inverse of the integer packing: recover the `m x m` boolean closure."""
    bits = ((int(code) >> np.arange(m * m)) & 1).astype(bool)
    return bits.reshape(m, m)


def per_block_log_likelihood(features, beta, epsilon, lambda_rep, lambda_back):
    """Per-row log likelihood; `.sum()` is `cached_batch_log_likelihood` exactly.

    The frozen function returns only the total. Stage 6E needs the terms separately,
    because a block's score belongs to whichever candidate segment contains it. The
    arithmetic below is the frozen function's, with the final `.sum()` deferred; a test
    pins `per_block_log_likelihood(...).sum() == cached_batch_log_likelihood(...)`.
    """
    exponent = (beta * features["Q"] - lambda_rep * features["q"]
                - lambda_back * features["C_back"])
    exponent = exponent - exponent.max(axis=-1, keepdims=True)
    weights = features["F"] * np.exp(exponent)
    structural = weights / weights.sum(axis=-1, keepdims=True)
    mixed = (1.0 - epsilon) * structural + epsilon / features["m"]
    n, T = features["obs"].shape
    chosen = mixed[np.arange(n)[:, None], np.arange(T)[None, :], features["obs"]]
    return np.log(chosen).sum(axis=1)


# ------------------------------------------------------------------------ the problem
@dataclass
class MixedModel:
    """The registered Stage 6E1B problem. Frozen before any MCMC comparison exists."""

    traces: tuple
    n_skills: int
    m: int
    d: int
    epsilon: float
    pi: np.ndarray
    transition: np.ndarray
    delta_b: float = DELTA_B
    min_width: int = MIN_BLOCK_WIDTH
    max_width: int = MAX_BLOCK_WIDTH
    states: list = field(default_factory=list, repr=False)
    _width_groups: dict = field(default_factory=dict, repr=False)
    _span_row: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.traces = tuple(tuple(int(v) for v in t) for t in self.traces)
        self.pi = np.asarray(self.pi, dtype=float)
        self.transition = np.asarray(self.transition, dtype=float)
        lengths = {len(t) for t in self.traces}
        if len(lengths) != 1:
            raise ValueError("the Stage 6E1B reference uses equal-length traces")
        self.states = enumerate_states(len(self.traces[0]), self.n_skills,
                                       self.min_width, self.max_width)
        # every distinct (trace, start, end) that any enumerated state can ask about
        spans = sorted({(start, end)
                        for key in self.states
                        for start, end in _span_bounds(key)})
        rows: dict = {}
        for n in range(len(self.traces)):
            for start, end in spans:
                rows.setdefault(end - start, []).append((n, start, end))
        self._width_groups = {}
        self._span_row = {}
        for width, items in rows.items():
            self._width_groups[width] = np.array(
                [self.traces[n][a:b] for n, a, b in items], dtype=int)
            for index, item in enumerate(items):
                self._span_row[item] = (width, index)

    @property
    def J(self) -> int:
        return len(self.traces[0])

    @property
    def n_traces(self) -> int:
        return len(self.traces)

    @property
    def qmc_dimension(self) -> int:
        return 1 + self.n_skills * self.m * self.d + 4

    def coordinate_names(self) -> list:
        names = ["rho"]
        for k in range(self.n_skills):
            for i in range(self.m):
                for r in range(self.d):
                    names.append(f"z_U[{k},{i},{r}]")
        return names + list(SCALAR_NAMES)

    # -- per-draw evaluation (reference implementation) --------------------------------
    def span_log_likelihoods(self, u_by_skill, beta, omega, lambda_rep,
                             lambda_back) -> dict:
        """`{(trace, start, end, skill): log p_RFS}` for every span, from `q_0 = 0`.

        The straightforward implementation, built on the frozen feature bundle. It is the
        reference that `batched_span_log_likelihoods` is checked against; production uses
        the batched form because the reference needs millions of draws.
        """
        out = {}
        for skill in range(self.n_skills):
            u_k = np.asarray(u_by_skill[skill], dtype=float)
            for width, roles in self._width_groups.items():
                features = vectorized_state_features(roles, u_k, float(omega))
                values = per_block_log_likelihood(features, float(beta), self.epsilon,
                                                  float(lambda_rep), float(lambda_back))
                for (n, a, b), (w, index) in self._span_row.items():
                    if w == width:
                        out[(n, a, b, skill)] = float(values[index])
        return out

    # -- batched evaluation (production) ------------------------------------------------
    @property
    def span_list(self) -> list:
        """`[(trace, start, end), ...]` in a fixed order — the columns of the batch."""
        if not hasattr(self, "_span_list"):
            self._span_list = sorted(self._span_row)
        return self._span_list

    def batched_span_log_likelihoods(self, u, beta, omega, lambda_rep,
                                     lambda_back) -> np.ndarray:
        """`(B, n_spans, K)` log likelihoods, replayed from `q_0 = 0` for every block.

        Identical arithmetic to `span_log_likelihoods`, with the QMC draw promoted to a
        leading array axis: `precedence`, `kappa`, the feasibility product, the successor
        utilities and the back costs all carry a batch index. Nothing is truncated and no
        state crosses a block, a draw or a skill. The `q` trajectory is consumed step by
        step rather than stored, because only the chosen probability is ever needed.
        """
        u = np.asarray(u, dtype=float)                 # (B, K, m, d)
        B, K, m, _ = u.shape
        beta = np.asarray(beta, dtype=float).reshape(B, 1)
        lambda_rep = np.asarray(lambda_rep, dtype=float).reshape(B, 1)
        lambda_back = np.asarray(lambda_back, dtype=float).reshape(B, 1)
        kappa = (1.0 / (1.0 + np.exp(-np.asarray(omega, dtype=float)))).reshape(B, 1, 1)

        spans = self.span_list
        column = {span: i for i, span in enumerate(spans)}
        out = np.empty((B, len(spans), K))

        for k in range(K):
            u_k = u[:, k]                              # (B, m, d)
            # h(U): i > j iff U[i, r] > U[j, r] for every column r, transitively closed by
            # construction of the product order — the same map precedence_from_u applies.
            precedence = np.all(u_k[:, :, None, :] > u_k[:, None, :, :], axis=3)
            succ = precedence.astype(float)            # succ[b, x, z] = x precedes z
            succ_off = succ.copy()
            succ_off[:, np.arange(m), np.arange(m)] = 0.0
            pred = np.transpose(precedence, (0, 2, 1))  # pred[b, x, z] = z precedes x

            for width, roles in self._width_groups.items():
                R = roles.shape[0]
                q = np.zeros((B, R, m))
                total = np.zeros((B, R))
                for t in range(width):
                    feasibility = np.prod(
                        np.where(pred[:, None, :, :], q[:, :, None, :], 1.0), axis=3)
                    utilities = np.log1p(np.einsum("brz,bxz->brx", 1.0 - q, succ))
                    back = kappa * np.einsum("brz,bxz->brx", q, succ_off)
                    exponent = (beta[:, :, None] * utilities
                                - lambda_rep[:, :, None] * q
                                - lambda_back[:, :, None] * back)
                    exponent -= exponent.max(axis=2, keepdims=True)
                    weights = feasibility * np.exp(exponent)
                    structural = weights / weights.sum(axis=2, keepdims=True)
                    mixed = (1.0 - self.epsilon) * structural + self.epsilon / m
                    observed = roles[:, t]                          # (R,)
                    total += np.log(mixed[:, np.arange(R), observed])
                    gate = np.where(precedence[:, observed, :], kappa, 0.0)  # (B, R, m)
                    q = q * (1.0 - gate)
                    q[:, np.arange(R), observed] = 1.0
                for span in spans:
                    group_width, row = self._span_row[span]
                    if group_width == width:
                        out[:, column[span], k] = total[:, row]
        return out

    def state_matrix(self, log_pi, log_transition) -> tuple:
        """`(A, const)` such that `log w = span_ll_flat @ A.T + const`.

        `A[j, span * K + k]` is 1 when state `j` uses span `span` carrying skill `k`, so
        the emission sum becomes one matrix product; `const[j]` collects the boundary
        prior, the initial term and the transitions, none of which depend on the draw.
        """
        spans = self.span_list
        column = {span: i for i, span in enumerate(spans)}
        K = self.n_skills
        A = np.zeros((self.n_traces, len(self.states), len(spans) * K))
        const = np.zeros((self.n_traces, len(self.states)))
        log_db = math.log(self.delta_b)
        log_1mdb = math.log1p(-self.delta_b)
        for t in range(self.n_traces):
            for j, key in enumerate(self.states):
                labels = [k for _, k in key]
                for (a, b), skill in zip(_span_bounds(key), labels):
                    A[t, j, column[(t, a, b)] * K + skill] = 1.0
                value = float(log_pi[labels[0]])
                value += (len(labels) - 1) * log_db + (self.J - len(labels)) * log_1mdb
                for x, y in zip(labels[:-1], labels[1:]):
                    value += float(log_transition[x, y])
                const[t, j] = value
        return A, const

    def state_log_weights(self, trace_index: int, span_ll: dict, log_pi,
                          log_transition) -> np.ndarray:
        log_db = math.log(self.delta_b)
        log_1mdb = math.log1p(-self.delta_b)
        J = self.J
        out = np.empty(len(self.states))
        for i, key in enumerate(self.states):
            spans = _span_bounds(key)
            labels = [k for _, k in key]
            total = float(log_pi[labels[0]])
            for (start, end), skill in zip(spans, labels):
                total += span_ll[(trace_index, start, end, skill)]
            total += (len(labels) - 1) * log_db + (J - len(labels)) * log_1mdb
            for a, b in zip(labels[:-1], labels[1:]):
                total += float(log_transition[a, b])
            out[i] = total
        return out


def _span_bounds(key):
    out, start = [], 0
    for end, _ in key:
        out.append((start, int(end)))
        start = int(end)
    return out


# --------------------------------------------------------------------- QMC state build
def build_mixed_state(points: np.ndarray, model: MixedModel) -> dict:
    """QMC uniforms -> `(rho, U_{1:K}, scalars)` through the non-centred construction.

    Identical in form to the Stage 6D1 build, extended over `K` skills: each skill's `U_k`
    has its own `m x d` block of standard normals, and all of them share the single `rho`
    that the registered prior gives them.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n, K, m, d = points.shape[0], model.n_skills, model.m, model.d
    rho = prior_inverse_cdf("rho", points[:, 0])
    z = stats.norm.ppf(points[:, 1:1 + K * m * d]).reshape(n, K, m, d)
    u = np.empty((n, K, m, d))
    for i in range(n):
        chol = np.linalg.cholesky(sigma_rho_matrix(d, float(rho[i])))
        for k in range(K):
            u[i, k] = z[i, k] @ chol.T
    offset = 1 + K * m * d
    scalars = {name: prior_inverse_cdf(name, points[:, offset + j])
               for j, name in enumerate(SCALAR_NAMES)}
    return {"rho": rho, "z": z, "u": u, **scalars}


def h_label_of(u_k) -> bytes:
    return precedence_from_u(np.asarray(u_k, dtype=float)).tobytes()


# ------------------------------------------------------------------------- a replicate
def _systematic_resample(log_weights: np.ndarray, n: int, rng) -> np.ndarray:
    """Indices of `n` iid-equivalent draws from a weighted sample, by systematic resampling.

    Systematic (rather than multinomial) resampling is used because it has strictly lower
    variance for the same weight vector, and the mixed multivariate statistic downstream
    is sensitive to resampling noise on the reference side.
    """
    weights = np.exp(log_weights - log_weights.max())
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cumulative, positions).clip(0, len(weights) - 1)


def mixed_replicate(model: MixedModel, n_points: int, seed: int, chunk: int = 8192,
                    n_retained: int = 0, progress: int = 0) -> dict:
    """One independent scrambled replicate, accumulated in chunks.

    Memory is flat in `n_points`: the per-draw conditional distributions and `U` matrices
    are consumed as they are produced, and only the weighted accumulators, the log weights
    and the five scalar coordinates are kept. That is what makes a reference of several
    million points affordable at this precision.

    The running shift is the standard online log-sum-exp rescaling: when a later chunk's
    maximum log weight exceeds the current shift, every accumulator is rescaled by
    `exp(old - new)` before the chunk is added, so no accumulation ever overflows and the
    result does not depend on chunk order.
    """
    log_pi = np.log(model.pi)
    with np.errstate(divide="ignore"):
        log_transition = np.log(model.transition)
    A, const = model.state_matrix(log_pi, log_transition)
    n_states = len(model.states)
    K, m = model.n_skills, model.m
    rng = np.random.default_rng(seed)

    shift = -np.inf
    weight_total = 0.0
    weight_square = 0.0
    weight_max = 0.0
    conditional_sum = np.zeros((model.n_traces, n_states))
    sampled_sum = np.zeros((model.n_traces, n_states))
    h_weight = [dict() for _ in range(K)]
    relation_sum = np.zeros((K, m * m))
    log_w = np.empty(n_points)
    scalar_values = {name: np.empty(n_points) for name in ("rho", *SCALAR_NAMES)}
    # Retained for the mixed multivariate statistic: a weighted systematic resample needs
    # the joint coordinates of each draw, so the closure indicators and the sampled (S, z)
    # are kept alongside the scalars. U itself is not kept -- only h(U), which is what the
    # statistic uses and is 1/12 the size.
    keep_closures = np.empty((n_points, K, m * m), dtype=bool) if n_retained else None
    keep_sampled = np.empty((n_points, model.n_traces), dtype=np.int16) if n_retained \
        else None

    # One scrambled engine, drawn from in chunks: the sequence is identical to taking
    # `sobol_points(n_points, ...)` in one call, but memory stays flat in n_points.
    engine = qmc.Sobol(d=model.qmc_dimension, scramble=True, seed=seed)
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        points = np.clip(engine.random(stop - start), 1e-12, 1.0 - 1e-12)
        state = build_mixed_state(points, model)
        span_ll = model.batched_span_log_likelihoods(
            state["u"], state["beta"], state["omega"], state["lambda_rep"],
            state["lambda_back"])                                   # (B, n_spans, K)
        B = stop - start
        flat = span_ll.reshape(B, -1)
        # log w(S, z) for every state, every trace: one matrix product plus a constant
        weights = np.einsum("bs,tjs->btj", flat, A) + const[None, :, :]
        log_z = logsumexp(weights, axis=2)                          # (B, n_traces)
        conditional = np.exp(weights - log_z[:, :, None])
        chunk_log_w = log_z.sum(axis=1)
        log_w[start:stop] = chunk_log_w
        for name in ("rho", *SCALAR_NAMES):
            scalar_values[name][start:stop] = state[name]
        if keep_closures is not None:
            for k in range(K):
                u_k = state["u"][:, k]
                keep_closures[start:stop, k] = np.all(
                    u_k[:, :, None, :] > u_k[:, None, :, :], axis=3).reshape(B, -1)

        # one (S, z) per trace from its exact conditional -- section 9's iid-equivalent
        # construction, drawn by inverse-CDF so it costs no Python loop over draws
        cumulative = np.cumsum(conditional, axis=2)
        uniforms = rng.random((B, model.n_traces, 1))
        sampled = (uniforms > cumulative).sum(axis=2).clip(0, n_states - 1)
        if keep_sampled is not None:
            keep_sampled[start:stop] = sampled

        chunk_shift = float(chunk_log_w.max())
        if chunk_shift > shift:
            rescale = math.exp(shift - chunk_shift) if np.isfinite(shift) else 0.0
            weight_total *= rescale
            weight_square *= rescale ** 2
            weight_max *= rescale
            conditional_sum *= rescale
            sampled_sum *= rescale
            relation_sum *= rescale
            for table in h_weight:
                for key in table:
                    table[key] *= rescale
            shift = chunk_shift
        w = np.exp(chunk_log_w - shift)
        weight_total += float(w.sum())
        weight_square += float((w ** 2).sum())
        weight_max = max(weight_max, float(w.max()))
        conditional_sum += np.einsum("b,btj->tj", w, conditional)
        for t in range(model.n_traces):
            np.add.at(sampled_sum[t], sampled[:, t], w)

        for k in range(K):
            u_k = state["u"][:, k]
            closure = np.all(u_k[:, :, None, :] > u_k[:, None, :, :], axis=3)
            flat_closure = closure.reshape(B, -1)
            relation_sum[k] += w @ flat_closure.astype(float)
            # Label each induced order by packing its m x m closure into one integer, so
            # the per-draw grouping is a bincount rather than a Python loop over draws.
            code = flat_closure.astype(np.int64) @ _BIT_WEIGHTS[:m * m]
            unique, inverse = np.unique(code, return_inverse=True)
            sums = np.bincount(inverse, weights=w, minlength=len(unique))
            for value, mass in zip(unique.tolist(), sums.tolist()):
                h_weight[k][value] = h_weight[k].get(value, 0.0) + mass
        if progress and (stop % progress) < chunk:
            print(f"    {stop:,}/{n_points:,}", flush=True)

    normaliser = weight_total
    ess = (normaliser ** 2) / weight_square
    scalar_weights = np.exp(log_w - shift) / normaliser
    retained = None
    if n_retained:
        index = _systematic_resample(log_w, n_retained, np.random.default_rng(seed + 1))
        retained = {
            "closures": keep_closures[index], "sampled": keep_sampled[index],
            **{name: scalar_values[name][index] for name in ("rho", *SCALAR_NAMES)},
            "note": "iid-equivalent draws from this replicate, by systematic resampling "
                    "on the exact importance weights; used only by the mixed multivariate "
                    "statistic",
        }
    return {
        "retained": retained,
        "seed": seed, "n_points": int(n_points),
        "log_evidence": shift + math.log(normaliser / n_points),
        "ess": ess, "relative_ess": ess / n_points,
        "max_normalised_weight": weight_max / normaliser,
        "conditional_probability": conditional_sum / normaliser,
        "sampled_probability": sampled_sum / normaliser,
        "h_probability": [
            {key: value / normaliser for key, value in table.items()}
            for table in h_weight],
        "relation_marginal": relation_sum / normaliser,
        "weights": scalar_weights,
        **{name: scalar_values[name] for name in ("rho", *SCALAR_NAMES)},
    }


def _weighted_summary(values, weights) -> dict:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = float((weights * values).sum())
    var = float((weights * (values - mean) ** 2).sum())
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    def quantile(q):
        return float(np.interp(q, cdf, values[order]))
    return {"mean": mean, "sd": math.sqrt(max(var, 0.0)), "median": quantile(0.5),
            "q025": quantile(0.025), "q975": quantile(0.975)}


def segmentation_marginal_maps(model: MixedModel) -> tuple:
    """`(boundary_map, label_map, count_map)`: linear maps from a state distribution.

    Every segmentation marginal is a linear functional of the distribution over states, so
    they are derived from the accumulated distribution rather than accumulated separately.
    """
    J, K = model.J, model.n_skills
    n_states = len(model.states)
    boundary = np.zeros((n_states, J - 1))
    labels = np.zeros((n_states, J, K))
    counts = np.zeros((n_states, J + 1))
    for j, key in enumerate(model.states):
        for end, _ in key[:-1]:
            boundary[j, end - 1] = 1.0
        for (a, b), k in zip(_span_bounds(key), [s for _, s in key]):
            labels[j, a:b, k] = 1.0
        counts[j, len(key)] = 1.0
    return boundary, labels, counts


def mixed_replicate_summary(replicate: dict, model: MixedModel) -> dict:
    """Everything the reference must report, by BOTH estimators."""
    m, K = model.m, model.n_skills
    boundary_map, label_map, count_map = segmentation_marginal_maps(model)
    conditional_p = replicate["conditional_probability"]
    sampled_p = replicate["sampled_probability"]

    def marginals(distribution):
        return (distribution @ boundary_map,
                np.einsum("tj,jik->tik", distribution, label_map),
                distribution @ count_map)

    boundary_c, labels_c, counts_c = marginals(conditional_p)
    boundary_s, labels_s, counts_s = marginals(sampled_p)

    h_probability, h_keys, relation_counts = [], [], []
    for k in range(K):
        table = replicate["h_probability"][k]
        order = sorted(table)
        probability = np.array([table[key] for key in order])
        closures = np.array([_decode_closure(key, m).reshape(-1) for key in order],
                            dtype=float)
        h_probability.append(probability)
        h_keys.append([_decode_closure(key, m).tobytes() for key in order])
        relation_counts.append(float((probability * closures.sum(axis=1)).sum()))

    scalars = {name: _weighted_summary(replicate[name], replicate["weights"])
               for name in ("rho", *SCALAR_NAMES)}

    return {
        "log_evidence": replicate["log_evidence"], "ess": replicate["ess"],
        "relative_ess": replicate["relative_ess"],
        "max_normalised_weight": replicate["max_normalised_weight"],
        "n_points": replicate["n_points"], "seed": replicate["seed"],
        "segmentation_conditional": conditional_p, "segmentation_sampled": sampled_p,
        "boundary_conditional": boundary_c, "boundary_sampled": boundary_s,
        "labels_conditional": labels_c, "labels_sampled": labels_s,
        "segment_counts_conditional": counts_c, "segment_counts_sampled": counts_s,
        "h_probability": h_probability, "h_keys": h_keys,
        "relation_marginal": replicate["relation_marginal"],
        "expected_relation_count": np.array(relation_counts),
        "scalars": scalars,
    }


def combine_mixed_replicates(summaries: list, model: MixedModel) -> dict:
    """Pool the replicates and measure the reference's own precision.

    The primary precision statistic is `rqmc_se = sd(estimates, ddof=1)/sqrt(R)` — the
    uncertainty of the replicate *mean*, which is what the downstream comparison consumes.
    The maximum departure across replicates is retained only as a descriptive diagnostic:
    it estimates the dispersion of a single replicate, does not shrink with `R`, and was
    superseded in Stage 6D1 for exactly that reason.
    """
    R = len(summaries)
    segmentation = np.array([s["segmentation_conditional"] for s in summaries])
    segmentation_sampled = np.array([s["segmentation_sampled"] for s in summaries])
    boundary = np.array([s["boundary_conditional"] for s in summaries])
    labels = np.array([s["labels_conditional"] for s in summaries])
    counts = np.array([s["segment_counts_conditional"] for s in summaries])
    relation = np.array([s["relation_marginal"] for s in summaries])

    log_z = np.array([s["log_evidence"] for s in summaries])
    rel_ess = np.array([s["relative_ess"] for s in summaries])
    max_w = np.array([s["max_normalised_weight"] for s in summaries])

    # H probabilities: align on the union of keys, per skill
    pooled_h, h_union, h_aligned = [], [], []
    for k in range(model.n_skills):
        union: list = []
        for s in summaries:
            for key in s["h_keys"][k]:
                if key not in union:
                    union.append(key)
        aligned = np.zeros((R, len(union)))
        for i, s in enumerate(summaries):
            for key, p in zip(s["h_keys"][k], s["h_probability"][k]):
                aligned[i, union.index(key)] = p
        h_union.append(union)
        h_aligned.append(aligned)
        pooled_h.append(aligned.mean(axis=0))

    names = ("rho", *SCALAR_NAMES)
    scalar_means = np.array([[s["scalars"][n]["mean"] for n in names] for s in summaries])

    precision = {
        "segmentation": rqmc_standard_error(segmentation.reshape(R, -1)),
        "boundary": rqmc_standard_error(boundary.reshape(R, -1)),
        "labels": rqmc_standard_error(labels.reshape(R, -1)),
        "relation": rqmc_standard_error(relation.reshape(R, -1)),
        "scalar_means": rqmc_standard_error(scalar_means),
        "h": [rqmc_standard_error(a) for a in h_aligned],
    }
    max_rqmc_se = max(
        float(precision["segmentation"]["max_standard_error"]),
        float(precision["boundary"]["max_standard_error"]),
        float(precision["labels"]["max_standard_error"]),
        float(precision["relation"]["max_standard_error"]),
        max(float(p["max_standard_error"]) for p in precision["h"]))
    max_half_width = max(
        float(precision["segmentation"]["max_half_width_95"]),
        float(precision["boundary"]["max_half_width_95"]),
        float(precision["labels"]["max_half_width_95"]),
        float(precision["relation"]["max_half_width_95"]),
        max(float(p["max_half_width_95"]) for p in precision["h"]))

    # ---- superseded descriptive diagnostics -------------------------------------------
    mean_h = [a.mean(axis=0) for a in h_aligned]
    replicate_h_tv = [max(0.5 * float(np.abs(row / row.sum() - mh / mh.sum()).sum())
                          for row in a)
                      for a, mh in zip(h_aligned, mean_h)]
    replicate_relation_departure = float(
        np.abs(relation - relation.mean(axis=0)).max())

    return {
        "n_replicates": R,
        "pooled_segmentation": segmentation.mean(axis=0),
        "pooled_segmentation_sampled": segmentation_sampled.mean(axis=0),
        "pooled_boundary": boundary.mean(axis=0),
        "pooled_labels": labels.mean(axis=0),
        "pooled_segment_counts": counts.mean(axis=0),
        "pooled_relation": relation.mean(axis=0),
        "pooled_h_probability": pooled_h, "h_keys": h_union,
        "per_replicate_h": h_aligned,
        "scalars": {n: {"mean_of_means": float(scalar_means[:, j].mean()),
                        "sd_across_replicates": float(scalar_means[:, j].std(ddof=1)),
                        "mean_of_sds": float(np.mean(
                            [s["scalars"][n]["sd"] for s in summaries])),
                        "pooled_summary": {
                            key: float(np.mean([s["scalars"][n][key] for s in summaries]))
                            for key in ("mean", "sd", "median", "q025", "q975")}}
                    for j, n in enumerate(names)},
        "precision": {
            "max_rqmc_standard_error": max_rqmc_se,
            "max_half_width_95": max_half_width,
            "t_multiplier": float(precision["segmentation"]["t_multiplier"]),
            "definition": "rqmc_se = sd(replicate_estimates, ddof=1)/sqrt(R); the "
                          "half-width is t(0.975, R-1) * rqmc_se",
        },
        "superseded_descriptive": {
            "max_replicate_h_total_variation": float(max(replicate_h_tv)),
            "per_skill_replicate_h_total_variation": replicate_h_tv,
            "max_replicate_relation_departure": replicate_relation_departure,
            "why_superseded": "a maximum over replicates estimates the dispersion of a "
                              "SINGLE replicate; it does not shrink as R grows and is not "
                              "an uncertainty for the replicate mean, which is the "
                              "quantity the comparison consumes. Superseded in Stage 6D1 "
                              "and retained here as a descriptive diagnostic only.",
        },
        "log_evidence": {"mean": float(log_z.mean()), "sd": float(log_z.std(ddof=1)),
                         "range": float(log_z.max() - log_z.min()),
                         "values": log_z.tolist()},
        "relative_ess": {"min": float(rel_ess.min()), "mean": float(rel_ess.mean())},
        "max_normalised_weight": {"max": float(max_w.max()), "mean": float(max_w.mean())},
    }


# ------------------------------------------------------------- exchangeability audit
def label_permutation_audit(model: MixedModel) -> dict:
    """Is any skill relabelling a symmetry of this target? Measured, not asserted.

    A permutation `sigma` is a symmetry only if `pi[sigma(k)] = pi[k]` and
    `P[sigma(h), sigma(k)] = P[h, k]` for every pair, since `U` carries an exchangeable
    prior and the likelihood is otherwise label-blind. Any departure is reported.
    """
    from itertools import permutations
    K = model.n_skills
    rows = []
    for sigma in permutations(range(K)):
        if sigma == tuple(range(K)):
            continue
        pi_gap = float(np.abs(model.pi[list(sigma)] - model.pi).max())
        permuted = model.transition[np.ix_(list(sigma), list(sigma))]
        p_gap = float(np.abs(permuted - model.transition).max())
        rows.append({"permutation": list(sigma), "max_pi_departure": pi_gap,
                     "max_P_departure": p_gap,
                     "is_symmetry": bool(pi_gap < 1e-12 and p_gap < 1e-12)})
    return {
        "permutations_checked": len(rows), "rows": rows,
        "any_nontrivial_symmetry": any(r["is_symmetry"] for r in rows),
        "min_departure_over_permutations": min(
            max(r["max_pi_departure"], r["max_P_departure"]) for r in rows),
        "conclusion": ("labels are exchangeable; per-skill summaries are not identified"
                       if any(r["is_symmetry"] for r in rows)
                       else "no nontrivial relabelling is a symmetry of this target, so "
                            "per-skill summaries are well posed and raw per-skill R-hat "
                            "is meaningful"),
    }
