"""Synthetic generator for the HPOP recovery experiment.

Ground truth produced here:

* a library of `K_true` reusable skill types, each a local DAG over a subset of the CPA vocabulary
  plus a composition profile;
* a type-level global DAG saying which skills must precede which;
* per trace: the skill-instance sequence, the true CPA-occurrence boundaries, and the CPA trace;
* a simulated LLM seed oversegmentation with controllable boundary recall and spurious-cut rate.

Two things are deliberately *not* drawn from the fitted model, so the experiment is not a
self-fulfilling well-specified check:

1. **Repair loops are exogenous.** After a verify role executes, a failure fires with probability
   `fail_prob` and invalidates the roles it depends on. HPOP has no failure signal: its invalidation
   `J = D * sigmoid(omega)` only flows forward along precedence, so a *failed test* can never
   invalidate the edit that preceded it. Under HPOP the loop restart is an unexplained move charged
   `lambda_rep`. Generating failures exogenously tests the model under exactly that mismatch.
2. **Seed boundaries are imperfect.** `boundary_recall < 1` deletes true boundaries from the seed
   set, which the merge-only model provably cannot repair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hpop.inference.recurrent import RecurrentFrontier, hard_precedence_matrix

CPA_VOCAB = [
    "PLAN_APPROACH", "SEARCH_CODE", "READ_SOURCE", "REPRODUCE_ISSUE", "RUN_TEST_SUITE",
    "INSPECT_FAILURE", "EDIT_CODE", "WRITE_TEST", "VERIFY_FIX", "RUN_LINTER",
    "SUMMARIZE_CHANGE", "SUBMIT_PATCH",
]
VERIFY_LIKE = {"REPRODUCE_ISSUE", "RUN_TEST_SUITE", "VERIFY_FIX", "RUN_LINTER"}


@dataclass
class SkillType:
    name: str
    roles: list                      # CPA indices in this skill's support
    edges: list                      # (u, v) cover edges over CPA indices, u must precede v
    theta: np.ndarray                # composition logits over the full vocab
    verify_roles: tuple = ()

    def precedence(self, V):
        return hard_precedence_matrix(V, self.edges)


@dataclass
class Trace:
    cpas: list                       # CPA indices, length T
    true_boundaries: list            # cut positions in CPA index space, strictly inside (0, T)
    skill_labels: list               # one per true segment
    seed_boundaries: list            # simulated LLM cuts, superset/subset of true_boundaries
    instance_spans: list = field(default_factory=list)   # (start, end) per true segment
    outcomes: list = field(default_factory=list)         # per-occurrence SUCCESS / FAILURE

    def failure_rate(self):
        return sum(o == "FAILURE" for o in self.outcomes) / max(len(self.outcomes), 1)


@dataclass
class SyntheticWorld:
    skills: list
    global_edges: list
    V: int
    vocab: list

    def local_matrices(self):
        return [s.precedence(self.V) for s in self.skills]

    def global_matrix(self):
        return hard_precedence_matrix(len(self.skills), self.global_edges)


def build_world(rng, K_true=4, V=None, roles_per_skill=(3, 5), extra_edge_prob=0.35,
                composition_strength=2.5):
    """Sample a ground-truth skill library and a type-level global DAG."""
    vocab = CPA_VOCAB if V is None else CPA_VOCAB[:V]
    V = len(vocab)
    skills = []
    for k in range(K_true):
        m = int(rng.integers(roles_per_skill[0], roles_per_skill[1] + 1))
        roles = sorted(rng.choice(V, size=m, replace=False).tolist())
        order = rng.permutation(roles).tolist()          # a topological order over the support
        edges = [(order[i], order[i + 1]) for i in range(m - 1)]   # backbone chain
        for i in range(m):                                # optional extra forward edges
            for j in range(i + 2, m):
                if rng.random() < extra_edge_prob:
                    edges.append((order[i], order[j]))
        # drop some backbone edges to create genuinely incomparable pairs
        keep = [e for e in edges[: m - 1] if rng.random() > 0.30] + edges[m - 1:]
        if not keep:
            keep = edges[:1]
        theta = np.full(V, -composition_strength)
        theta[roles] = composition_strength
        # only the *last* verification in the skill can fail; earlier ones are treated as setup.
        verify_candidates = [r for r in order if vocab[r] in VERIFY_LIKE]
        verify = (verify_candidates[-1],) if verify_candidates else ()
        skills.append(SkillType(f"SKILL_{k}", roles, keep, theta, verify))

    # global type-level DAG: a random DAG over skill indices with a topological backbone
    perm = rng.permutation(K_true).tolist()
    global_edges = [(perm[i], perm[i + 1]) for i in range(K_true - 1)
                    if rng.random() > 0.35]
    for i in range(K_true):
        for j in range(i + 2, K_true):
            if rng.random() < 0.25:
                global_edges.append((perm[i], perm[j]))
    return SyntheticWorld(skills, global_edges, V, vocab)


def _sample_skill_sequence(world, rng, n_instances, beta=1.0, lam_rep=2.0, eps=0.05):
    """Skill-instance labels for one trace: a repeated-execution trace over the global DAG."""
    Dg = world.global_matrix()
    model = RecurrentFrontier(Dg, omega=2.0, beta=beta, lam_rep=lam_rep, lam_back=0.5, eps=eps)
    q = np.zeros(Dg.shape[0])
    labels = []
    for _ in range(n_instances):
        p = np.exp(model.step_logprobs(q))
        if labels:                                        # canonical adjacent-label convention
            p[labels[-1]] = 0.0
        p = p / p.sum()
        k = int(rng.choice(len(p), p=p))
        labels.append(k)
        q = model.update(q, k)
        if np.all(q > 1 - 1e-9):
            q = np.zeros_like(q)                          # the workflow restarts a new pass
    return labels


def sample_trace(world, rng, n_instances=(3, 6), fail_prob=0.25, beta=1.5, lam_rep=1.5,
                 lam_back=0.5, eps=0.02, omega=2.5, boundary_recall=1.0, oversegment_rate=0.40,
                 max_steps=12):
    L = int(rng.integers(n_instances[0], n_instances[1] + 1))
    labels = _sample_skill_sequence(world, rng, L)
    Ds = world.local_matrices()

    cpas, spans, true_cuts, seed_cuts, outcomes = [], [], [], [], []
    for li, k in enumerate(labels):
        sk = world.skills[k]
        model = RecurrentFrontier(Ds[k], omega=omega, beta=beta, lam_rep=lam_rep,
                                  lam_back=lam_back, eps=eps, theta=sk.theta)
        seq, outs = _sample_instance(model, sk, rng, fail_prob, max_steps)
        start = len(cpas)
        cpas.extend(seq)
        outcomes.extend(outs)
        spans.append((start, len(cpas)))
        if li < L - 1:
            true_cuts.append(len(cpas))
        # simulated LLM seeds: keep this true boundary with prob `boundary_recall`
        if li < L - 1 and rng.random() < boundary_recall:
            seed_cuts.append(len(cpas))
        # spurious cuts strictly inside the instance
        for pos in range(start + 1, len(cpas)):
            if rng.random() < oversegment_rate:
                seed_cuts.append(pos)
    seed_cuts = sorted(set(seed_cuts))
    return Trace(cpas, true_cuts, labels, seed_cuts, spans, outcomes)


def _sample_instance(model, skill, rng, fail_prob, max_steps):
    """One skill instance: execute until everything is valid, with exogenous verify failures.

    Returns (sequence, outcomes). The outcome is the *observable* signal a real trace would carry:
    a verification that failed is recorded as FAILURE. HPOP's own invalidation cannot produce these
    events -- it only flows forward along precedence -- so this is the supervision that the
    failure-conditioned correction (E.4) is meant to consume.
    """
    q = np.zeros(model.M)
    allowed = np.zeros(model.M, dtype=bool)
    allowed[skill.roles] = True                            # an instance only realizes its own roles
    seq, outs = [], []
    for _ in range(max_steps):
        p = np.exp(model.step_logprobs(q))
        p = np.where(allowed, p, 0.0)
        if p.sum() <= 0:
            break
        y = int(rng.choice(model.M, p=p / p.sum()))
        seq.append(y)
        outs.append("SUCCESS")
        q = model.update(q, y)
        if y in skill.verify_roles and rng.random() < fail_prob:
            # a failed verification invalidates itself and everything it depends on
            outs[-1] = "FAILURE"
            for z in np.where(model.D[:, y] > 0)[0]:
                q[z] = 0.0
            q[y] = 0.0
        if np.all(q[allowed] > 1 - 1e-9):
            break
    if not seq:                                            # degenerate guard
        seq, outs = [int(rng.choice(skill.roles))], ["SUCCESS"]
    return seq, outs


def sample_corpus(seed=0, n_traces=60, K_true=4, V=12, **kwargs):
    rng = np.random.default_rng(seed)
    world = build_world(rng, K_true=K_true, V=V)
    traces = [sample_trace(world, rng, **kwargs) for _ in range(n_traces)]
    return world, traces


def seeds_of(trace):
    """Seed segments as lists of CPA indices, from the simulated LLM boundaries."""
    cuts = [0] + list(trace.seed_boundaries) + [len(trace.cpas)]
    return [trace.cpas[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]


def true_seed_index_boundaries(trace):
    """True boundaries expressed as indices into the seed-segment list (merge-only targets).

    Returns (indices, recoverable) where `recoverable` is False if some true boundary is missing
    from the seed set, i.e. no merge-only segmentation can reproduce the ground truth.
    """
    cuts = [0] + list(trace.seed_boundaries) + [len(trace.cpas)]
    pos_to_seed = {p: i for i, p in enumerate(cuts)}
    idx, ok = [], True
    for b in trace.true_boundaries:
        if b in pos_to_seed:
            idx.append(pos_to_seed[b])
        else:
            ok = False
    return idx, ok
