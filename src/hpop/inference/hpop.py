"""HPOP: merge-only semi-Markov segmentation over a library of revisitable local partial orders.

Variational EM over the seed lattice:

* **E-step** — exact. Both the model and the variational family are semi-Markov over the same
  merge-only seed lattice, so setting the inference potential equal to the expected generative
  segment score `l_n(a,b,k)` (Eq. expected-generative-segment-score) makes `q_phi` the exact
  conditional posterior. Forward-backward gives the segment marginals `mu_n(a,b,k)` and the exact
  log partition. No amortized inference network is needed.
* **M-step** — per skill: (i) composition logits `theta_k` by weighted gradient ascent (the gradient
  is the standard softmax form because `theta` enters `log W` additively), (ii) local partial order
  `D_k` by a weighted first-occurrence pairwise-consensus estimator followed by likelihood-guided
  add/prune passes under a cover-edge complexity penalty. `pi` gets the Dirichlet(alpha/K_max)
  posterior mean.

Held-out scoring uses a *normalized* segmentation prior. The manuscript's ELBO
(Eq. structured-segmentation-elbo) carries `sum_l [log pi_{z_l} - lambda_seg]` without dividing by
its partition function over the lattice, so the implied prior over segmentations is improper and
raw `log Z` values are not comparable across models. We therefore report

    log p(x_{1:J}) = log Z(score) - log Z(prior-only score),

which is the exact marginal likelihood of the CPA sequence under the normalized joint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from hpop.inference.recurrent import hard_precedence_matrix, sigmoid
from hpop.inference.semi_markov import SemiMarkovLattice

_FLOOR = 1e-300


# ---------------------------------------------------------------------------------------------
# batched recurrent frontier likelihood: evaluates one CPA block under all K skills at once
# ---------------------------------------------------------------------------------------------
def batched_block_logp(D, theta, J, seq, beta, lam_rep, lam_back, eps, need_grad=False,
                       lam_comp=1.0, lam_po=1.0, seq_logits=None, J_fail=None, outcomes=None):
    """log p(seq | H_k) for every k. D:(K,V,V) theta:(K,V) J:(K,V,V). Returns (K,) [, (K,V) grad].

    Run 2 additions, all no-ops at their defaults:
      lam_comp / lam_po  weight the composition and partial-order halves of log W separately, so
                         `lam_po = 0` is exactly the HSMM and `lam_comp = 0` is order-only.
      seq_logits         (K, V+1, V) HPOP+Seq transition term, added inside the same softmax so the
                         model stays normalized and the term is automatically frontier-restricted.
      J_fail / outcomes  failure-conditioned invalidation: when the observed outcome of step t is a
                         FAILURE, the roles the executed role *depends on* are invalidated too —
                         the manuscript's forward-only J~ cannot express this.
    """
    K, V = theta.shape
    q = np.zeros((K, V))
    total = np.zeros(K)
    grad = np.zeros((K, V)) if need_grad else None
    eye = np.eye(V, dtype=bool)
    prev = V                                                   # start-of-block context row
    # Fast path: for a *binary* precedence matrix, 1 - D[z,x](1-q[z]) is q[z] where D = 1 and 1
    # elsewhere, so log F~(x) = sum_{z in Pred(x)} log q(z) is a single contraction. Exact, not an
    # approximation; the general relaxed path below handles D in (0,1).
    binary = bool(np.all((D == 0.0) | (D == 1.0)))
    for t, y in enumerate(seq):
        stale = 1.0 - q                                        # (K,V)
        if binary:
            log_F = np.einsum('kzx,kz->kx', D, np.log(np.maximum(q, _FLOOR)))
        else:
            A = 1.0 - D * stale[:, :, None]                    # A[k,z,x]
            A[:, eye] = 1.0
            log_F = np.log(np.clip(A, _FLOOR, None)).sum(axis=1)
        S = np.einsum('kxz,kz->kx', D, stale)
        C_back = np.einsum('kxz,kz->kx', J, q)
        log_W = (lam_comp * theta
                 + lam_po * (log_F + beta * np.log1p(S) - lam_rep * q - lam_back * C_back))
        if seq_logits is not None:
            log_W = log_W + seq_logits[:, prev, :]
        log_W -= log_W.max(axis=1, keepdims=True)
        W = np.exp(log_W)
        p_soft = W / W.sum(axis=1, keepdims=True)
        p_mix = (1.0 - eps) * p_soft + eps / V
        py = np.clip(p_mix[:, y], _FLOOR, None)
        total += np.log(py)
        if need_grad:
            onehot = np.zeros(V)
            onehot[y] = 1.0
            grad += (1.0 - eps) * (p_soft[:, y] / py)[:, None] * (onehot[None, :] - p_soft)
        q = q * (1.0 - J[:, y, :])
        q[:, y] = 1.0
        if J_fail is not None and outcomes is not None and t < len(outcomes) \
                and outcomes[t] == "FAILURE":
            # the verification failed: what it depended on is now suspect, and so is its own result
            q = q * (1.0 - J_fail[:, y, :])
            q[:, y] = 0.0
        prev = y
    return (total, grad) if need_grad else total


# ---------------------------------------------------------------------------------------------
@dataclass
class HPOPConfig:
    V: int
    K_max: int = 6
    D_max: int = 8
    beta: float = 1.5
    lam_rep: float = 1.5
    lam_back: float = 0.5
    omega: float = 2.5                 # invalidation logit; sigmoid(omega) is the invalidation rate
    eps: float = 0.02
    alpha: float = 1.0                 # Dirichlet concentration -> sparse library
    lam_seg: float = 1.0               # per-skill-instance penalty
    lam_edge: float = 0.8              # cover-edge complexity penalty
    tau_act: float = 1.0               # soft active-library temperature
    lam_K: float = 0.0
    use_order: bool = True             # False -> HSMM baseline (composition only, no partial order)
    use_recurrence: bool = True        # False -> ablation: no invalidation, nothing ever goes stale
    # --- Run 2 corrections. Every default below reproduces the Run 1 model exactly. ---
    normalized_duration: bool = False  # True -> proper duration+transition prior instead of lam_seg
    failure_invalidation: bool = False # True -> a FAILURE outcome invalidates upstream roles
    lam_comp: float = 1.0              # weight on the composition (emission) term
    lam_po: float = 1.0                # weight on the partial-order term
    seq_eta: float = 0.0               # HPOP+Seq: frontier-restricted transition term weight
    edge_threshold: float = 0.90       # pairwise-consensus threshold for a cover edge
    min_support: float = 0.25          # min expected occurrence share for a role to enter a skill
    theta_steps: int = 4
    theta_lr: float = 0.45
    mu_floor: float = 0.02             # ignore blocks with negligible posterior mass in the M-step
    max_theta_blocks: int = 700        # cap on blocks used for the theta gradient
    max_prune_blocks: int = 25         # cap on blocks used for likelihood-guided edge pruning


class HPOP:
    def __init__(self, cfg: HPOPConfig, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(0)
        K, V = cfg.K_max, cfg.V
        self.D = np.zeros((K, V, V))
        self.theta = 0.01 * self.rng.normal(size=(K, V))
        self.counts = np.full(K, cfg.alpha / K)
        # normalized segmentation prior (Run 2): duration, transition and start distributions.
        self.log_dur = np.full(cfg.D_max, -np.log(cfg.D_max))
        self.log_trans = np.log(np.where(np.eye(K, dtype=bool), 1e-12, 1.0 / max(K - 1, 1)))
        self.log_start = np.full(K, -np.log(K))
        self.seq_logits = np.zeros((K, V + 1, V)) if cfg.seq_eta else None
        self._outcomes = None
        self._refresh_J()

    # ---- parameter helpers ------------------------------------------------------------------
    def _refresh_J(self):
        nu = sigmoid(self.cfg.omega) if self.cfg.use_recurrence else 0.0
        self.J = self.D * nu
        # Failure-conditioned invalidation runs *backwards* along precedence: J_fail[y, z] > 0 when
        # z must precede y, so a failed y casts doubt on what produced its inputs.
        self.J_fail = (np.transpose(self.D, (0, 2, 1)) * sigmoid(self.cfg.omega)
                       if self.cfg.failure_invalidation else None)
        for k in range(self.cfg.K_max):
            np.fill_diagonal(self.J[k], 0.0)
            if self.J_fail is not None:
                np.fill_diagonal(self.J_fail[k], 0.0)

    def elogpi(self):
        c = self.counts
        return digamma(c) - digamma(c.sum())

    def active_skills(self, eps_use=None):
        """Slots with expected usage above eps_use (default: 1% of all expected instances)."""
        n = np.maximum(self.counts - self.cfg.alpha / self.cfg.K_max, 0.0)
        thr = 0.01 * n.sum() if eps_use is None else eps_use
        return np.where(n > thr)[0]

    def soft_K(self):
        n = np.maximum(self.counts - self.cfg.alpha / self.cfg.K_max, 0.0)
        return float(np.sum(1.0 - np.exp(-n / self.cfg.tau_act)))

    # ---- block enumeration ------------------------------------------------------------------
    def blocks(self, seeds):
        """blocks[(i, w)] = concatenated CPA list for seed block i .. i+w-1."""
        Jn, D = len(seeds), self.cfg.D_max
        out = {}
        for i in range(Jn):
            acc = []
            for w in range(1, min(D, Jn - i) + 1):
                acc = acc + seeds[i + w - 1]
                out[(i, w)] = list(acc)
        return out

    def block_outcomes(self, seeds, flat_outcomes):
        """Per-block slices of the per-occurrence outcome list, keyed like `blocks`."""
        if flat_outcomes is None:
            return None
        off = np.cumsum([0] + [len(s) for s in seeds])
        out = {}
        for i in range(len(seeds)):
            for w in range(1, min(self.cfg.D_max, len(seeds) - i) + 1):
                out[(i, w)] = flat_outcomes[off[i]:off[i + w]]
        return out

    def block_logliks(self, seeds, cache=None, outcomes=None):
        """(J, D_max, K) array of log p(block | H_k), -inf where the block runs off the end."""
        Jn, D, K = len(seeds), self.cfg.D_max, self.cfg.K_max
        out = np.full((Jn, D, K), -np.inf)
        Dmat = self.D if self.cfg.use_order else np.zeros_like(self.D)
        Jmat = self.J if self.cfg.use_order else np.zeros_like(self.J)
        ocache = self.block_outcomes(seeds, outcomes)
        sl = (self.cfg.seq_eta * self.seq_logits) if self.seq_logits is not None else None
        for (i, w), seq in (cache if cache is not None else self.blocks(seeds)).items():
            out[i, w - 1, :] = batched_block_logp(
                Dmat, self.theta, Jmat, seq,
                self.cfg.beta, self.cfg.lam_rep, self.cfg.lam_back, self.cfg.eps,
                lam_comp=self.cfg.lam_comp, lam_po=self.cfg.lam_po, seq_logits=sl,
                J_fail=self.J_fail, outcomes=(ocache[(i, w)] if ocache else None))
        return out

    def scores(self, seeds, logliks=None, cache=None, outcomes=None):
        ll = self.block_logliks(seeds, cache, outcomes) if logliks is None else logliks
        if self.cfg.normalized_duration:
            # a proper duration distribution replaces the free -lambda_seg factor; the transition
            # and start distributions are carried by the lattice itself
            prior = self.log_dur[None, :, None] * np.ones((1, 1, self.cfg.K_max))
        else:
            prior = self.elogpi()[None, None, :] - self.cfg.lam_seg
        prior = np.broadcast_to(prior, ll.shape).copy()
        return ll + prior, np.where(np.isfinite(ll), prior, -np.inf)

    def _lattice_kwargs(self):
        if not self.cfg.normalized_duration:
            return {}
        return {"log_trans": self.log_trans, "log_start": self.log_start}

    # ---- E-step ------------------------------------------------------------------------------
    def e_step(self, corpus, caches=None):
        """corpus: list of seed-segment lists. Returns per-trace marginals and log evidence."""
        out = []
        outs = getattr(self, "_outcomes", None)
        lk = self._lattice_kwargs()
        for n, seeds in enumerate(corpus):
            cache = caches[n] if caches is not None else self.blocks(seeds)
            oc = outs[n] if outs is not None else None
            ll = self.block_logliks(seeds, cache, oc)
            sc, prior = self.scores(seeds, ll, cache, oc)
            lat = SemiMarkovLattice(sc.copy(), **lk)
            mu = lat.segment_marginals()
            logZ = lat.log_partition()
            logZ0 = SemiMarkovLattice(prior.copy(), **lk).log_partition()
            rec = {"seeds": seeds, "cache": cache, "outcomes": oc, "mu": mu, "logZ": logZ,
                   "logZ_prior": logZ0, "logp": logZ - logZ0}
            if self.cfg.normalized_duration:
                rec["xi"], rec["start"] = lat.transition_marginals()
            out.append(rec)
        return out

    # ---- M-step ------------------------------------------------------------------------------
    def m_step(self, estep, update_structure=True):
        cfg = self.cfg
        K, V = cfg.K_max, cfg.V
        # --- expected usage / pi ---
        counts = np.zeros(K)
        for rec in estep:
            counts += rec["mu"].sum(axis=(0, 1))
        self.counts = counts + cfg.alpha / K

        # --- normalized segmentation prior: duration, transition, start (Run 2 correction E.1) ---
        if cfg.normalized_duration:
            dur = np.zeros(cfg.D_max)
            xi = np.zeros((K, K))
            start = np.zeros(K)
            for rec in estep:
                dur += rec["mu"].sum(axis=(0, 2))
                xi += rec["xi"]
                start += rec["start"]
            dur += 0.5
            self.log_dur = np.log(dur / dur.sum())
            np.fill_diagonal(xi, 0.0)
            xi = xi + 0.5 * (1.0 - np.eye(K))
            row = xi.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore"):
                self.log_trans = np.log(np.divide(xi, np.maximum(row, 1e-12)))
            self.log_trans[np.eye(K, dtype=bool)] = -np.inf
            start += 0.5
            self.log_start = np.log(start / start.sum())

        # --- collect blocks that carry meaningful posterior mass, with their (K,) weight vector ---
        active_blocks = []
        for rec in estep:
            cache = rec.get("cache") or self.blocks(rec["seeds"])
            ocache = self.block_outcomes(rec["seeds"], rec.get("outcomes"))
            mu = rec["mu"]
            for (i, w), seq in cache.items():
                m = mu[i, w - 1, :]
                if m.max() > cfg.mu_floor:
                    active_blocks.append((seq, m.copy(), ocache[(i, w)] if ocache else None))
        if len(active_blocks) > cfg.max_theta_blocks:            # keep the heaviest blocks
            order = np.argsort([-float(m.max()) for _, m, _ in active_blocks])
            active_blocks = [active_blocks[i] for i in order[:cfg.max_theta_blocks]]

        # --- theta: weighted gradient ascent (theta enters log W additively) ---
        Dmat = self.D if cfg.use_order else np.zeros_like(self.D)
        Jmat = self.J if cfg.use_order else np.zeros_like(self.J)
        sl = (cfg.seq_eta * self.seq_logits) if self.seq_logits is not None else None
        for _ in range(cfg.theta_steps):
            grad = np.zeros((K, V))
            for seq, m, oc in active_blocks:
                _, g = batched_block_logp(Dmat, self.theta, Jmat, seq, cfg.beta, cfg.lam_rep,
                                          cfg.lam_back, cfg.eps, need_grad=True,
                                          lam_comp=cfg.lam_comp, lam_po=cfg.lam_po,
                                          seq_logits=sl, J_fail=self.J_fail, outcomes=oc)
                grad += m[:, None] * g
            norm = np.maximum(np.abs(grad).max(axis=1, keepdims=True), 1e-9)
            self.theta += cfg.theta_lr * grad / norm
            self.theta -= self.theta.mean(axis=1, keepdims=True)     # remove the softmax null space
            self.theta = np.clip(self.theta, -8.0, 8.0)
        weighted = {k: [(seq, float(m[k])) for seq, m, _ in active_blocks if m[k] > cfg.mu_floor]
                    for k in range(K)}

        # --- HPOP+Seq: per-skill transition logits, estimated from expected adjacent-pair counts ---
        if self.seq_logits is not None:
            cnt = np.full((K, V + 1, V), 0.5)
            for seq, m, _ in active_blocks:
                prev = V
                for y in seq:
                    cnt[:, prev, y] += m
                    prev = y
            self.seq_logits = np.log(cnt / cnt.sum(axis=2, keepdims=True))

        # --- local partial orders ---
        if update_structure and cfg.use_order:
            for k in range(K):
                if len(weighted[k]) >= 3:
                    self.D[k] = self._fit_structure(k, weighted[k])
                else:
                    self.D[k] = np.zeros((V, V))
            self._refresh_J()

    def _fit_structure(self, k, blocks):
        """Weighted first-occurrence pairwise consensus, acyclified, then likelihood-pruned."""
        cfg = self.cfg
        V = cfg.V
        wsum = sum(w for _, w in blocks)
        occ = np.zeros(V)
        before = np.zeros((V, V))
        cooc = np.zeros((V, V))
        for seq, w in blocks:
            first = {}
            for t, c in enumerate(seq):
                first.setdefault(c, t)
            for c in first:
                occ[c] += w
            items = list(first)
            for a in items:
                for b in items:
                    if a == b:
                        continue
                    cooc[a, b] += w
                    if first[a] < first[b]:
                        before[a, b] += w
        support = occ / max(wsum, 1e-9)
        roles = np.where(support >= cfg.min_support)[0]
        if len(roles) < 2:
            return np.zeros((V, V))

        cand = []
        for a in roles:
            for b in roles:
                if a == b or cooc[a, b] <= 0:
                    continue
                frac = before[a, b] / cooc[a, b]
                pair_support = cooc[a, b] / max(wsum, 1e-9)
                if frac >= cfg.edge_threshold and pair_support >= cfg.min_support:
                    cand.append((frac * pair_support, int(a), int(b)))
        cand.sort(reverse=True)

        edges = []
        for _, a, b in cand:                                   # greedy acyclic insertion
            trial = hard_precedence_matrix(V, edges + [(a, b)])
            if np.any(np.diag(trial) > 0) or trial[b, a] > 0:
                continue
            edges.append((a, b))

        # likelihood-guided prune under the cover-edge complexity penalty
        if edges:
            edges = self._prune_edges(k, edges, blocks)
        return hard_precedence_matrix(V, edges)

    def _score_edges(self, k, edges, blocks):
        cfg = self.cfg
        D = hard_precedence_matrix(cfg.V, edges)[None, :, :]
        nu = sigmoid(cfg.omega) if cfg.use_recurrence else 0.0
        Jm = D * nu
        np.fill_diagonal(Jm[0], 0.0)
        th = self.theta[k:k + 1]
        tot = 0.0
        for seq, w in blocks:
            tot += w * float(batched_block_logp(D, th, Jm, seq, cfg.beta, cfg.lam_rep,
                                                cfg.lam_back, cfg.eps,
                                                lam_comp=cfg.lam_comp, lam_po=cfg.lam_po)[0])
        return tot - cfg.lam_edge * len(edges)

    def _prune_edges(self, k, edges, blocks, max_passes=2):
        cap = self.cfg.max_prune_blocks
        sub = blocks if len(blocks) <= cap else [blocks[i] for i in
                                                np.argsort([-w for _, w in blocks])[:cap]]
        best = self._score_edges(k, edges, sub)
        for _ in range(max_passes):
            improved = False
            for e in list(edges):
                trial = [x for x in edges if x != e]
                s = self._score_edges(k, trial, sub)
                if s > best:
                    best, edges, improved = s, trial, True
            if not improved:
                break
        return edges

    # ---- fitting -----------------------------------------------------------------------------
    def init_composition(self, corpus, window=3):
        """Initialize theta by k-means on the CPA histograms of short seed windows.

        EM from theta ~ 0 starts with a near-uniform emission model, so the first E-step carries
        almost no signal and the segmentation posterior is flat. Clustering block compositions is
        the standard fix and costs one pass over the corpus.
        """
        from sklearn.cluster import KMeans
        V, K = self.cfg.V, self.cfg.K_max
        rows = []
        for seeds in corpus:
            for i in range(len(seeds)):
                block = sum(seeds[i:i + window], [])
                if not block:
                    continue
                h = np.bincount(block, minlength=V).astype(float)
                rows.append(h / h.sum())
        if len(rows) < K:
            return
        X = np.array(rows)
        km = KMeans(n_clusters=K, n_init=4, random_state=int(self.rng.integers(1 << 30)))
        lab = km.fit_predict(X)
        for k in range(K):
            m = X[lab == k].mean(axis=0) if np.any(lab == k) else X.mean(axis=0)
            t = np.log(m + 1e-2)
            self.theta[k] = np.clip(t - t.mean(), -8.0, 8.0)

    def fit(self, corpus, iters=12, warmup=3, init=True, verbose=False, outcomes=None):
        """`outcomes` is an optional per-trace list of per-occurrence outcome strings, used only
        when cfg.failure_invalidation is set."""
        self._outcomes = outcomes
        if init:
            self.init_composition(corpus)
        caches = [self.blocks(seeds) for seeds in corpus]
        history = []
        for it in range(iters):
            estep = self.e_step(corpus, caches)
            evidence = sum(r["logp"] for r in estep)
            history.append(evidence)
            if verbose:
                print(f"  iter {it:2d}  log p(x) = {evidence:12.2f}  K+ = {len(self.active_skills())}")
            self.m_step(estep, update_structure=(it >= warmup))
        return history

    # ---- outputs -----------------------------------------------------------------------------
    def decode(self, seeds):
        sc, _ = self.scores(seeds)
        segs, _ = SemiMarkovLattice(sc.copy()).viterbi()
        return segs

    def heldout_logp(self, corpus, outcomes=None):
        """Normalized marginal log-likelihood of each trace's CPA sequence."""
        prev = getattr(self, "_outcomes", None)
        self._outcomes = outcomes
        try:
            return [r["logp"] for r in self.e_step(corpus)]
        finally:
            self._outcomes = prev

    def global_structure(self, corpus, edge_threshold=None):
        """Type-level global partial order over skills, from decoded instance sequences.

        The manuscript defines the global order over skill *instances*. Because the recurrent
        likelihood already lets a node be re-executed, the same machinery represents repeated
        instances at the *type* level without cycles, which is what we estimate here.
        """
        thr = self.cfg.edge_threshold if edge_threshold is None else edge_threshold
        K = self.cfg.K_max
        before = np.zeros((K, K))
        cooc = np.zeros((K, K))
        for seeds in corpus:
            labels = [k for _, _, k in self.decode(seeds)]
            first = {}
            for t, k in enumerate(labels):
                first.setdefault(k, t)
            for a in first:
                for b in first:
                    if a == b:
                        continue
                    cooc[a, b] += 1
                    if first[a] < first[b]:
                        before[a, b] += 1
        edges = []
        cand = []
        for a in range(K):
            for b in range(K):
                if a == b or cooc[a, b] < 3:
                    continue
                frac = before[a, b] / cooc[a, b]
                if frac >= thr:
                    cand.append((frac, a, b))
        cand.sort(reverse=True)
        for _, a, b in cand:
            trial = hard_precedence_matrix(K, edges + [(a, b)])
            if trial[b, a] > 0 or np.any(np.diag(trial) > 0):
                continue
            edges.append((a, b))
        return edges


# ---------------------------------------------------------------------------------------------
class FlatPoset:
    """Baseline: one global revisitable poset over the whole CPA vocabulary, no segmentation."""

    def __init__(self, cfg: HPOPConfig):
        self.cfg = cfg
        self.D = np.zeros((cfg.V, cfg.V))
        self.theta = np.zeros(cfg.V)

    def fit(self, corpus, iters=1):
        cfg = self.cfg
        traces = [sum(seeds, []) for seeds in corpus]
        blocks = [(t, 1.0) for t in traces]
        model = HPOP(cfg)
        model.theta = np.zeros((cfg.K_max, cfg.V))
        # composition
        counts = np.zeros(cfg.V)
        for t in traces:
            for c in t:
                counts[c] += 1
        self.theta = np.log(counts + 1.0)
        self.theta -= self.theta.mean()
        model.theta = np.repeat(self.theta[None, :], cfg.K_max, axis=0)
        self.D = model._fit_structure(0, blocks)
        return self

    def logp(self, corpus):
        cfg = self.cfg
        D = self.D[None, :, :]
        nu = sigmoid(cfg.omega) if cfg.use_recurrence else 0.0
        Jm = D * nu
        np.fill_diagonal(Jm[0], 0.0)
        th = self.theta[None, :]
        return [float(batched_block_logp(D, th, Jm, sum(seeds, []), cfg.beta, cfg.lam_rep,
                                         cfg.lam_back, cfg.eps)[0]) for seeds in corpus]
