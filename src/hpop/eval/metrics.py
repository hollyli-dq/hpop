"""Recovery metrics for the HPOP synthetic experiment.

All structural comparisons are made after matching learned skill slots to ground-truth skills with
a Hungarian assignment on the occurrence-level contingency table, so metrics do not depend on the
arbitrary numbering of library slots.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


# ---- segment bookkeeping -----------------------------------------------------------------
def seed_cut_positions(trace):
    return [0] + list(trace.seed_boundaries) + [len(trace.cpas)]


def decoded_to_cpa_spans(trace, segs):
    """Seed-index segments -> (start, end, skill) in CPA-occurrence space."""
    cuts = seed_cut_positions(trace)
    return [(cuts[a], cuts[b], k) for a, b, k in segs]


def occurrence_labels(spans, T):
    lab = np.full(T, -1, dtype=int)
    for a, b, k in spans:
        lab[a:b] = k
    return lab


def boundary_prf(pred_cuts, true_cuts, tol=0):
    """Precision/recall/F1 of interior boundary positions (exact match by default)."""
    pred, true = set(pred_cuts), set(true_cuts)
    if tol > 0:
        matched = sum(any(abs(p - t) <= tol for p in pred) for t in true)
        tp_p = sum(any(abs(p - t) <= tol for t in true) for p in pred)
    else:
        matched = len(pred & true)
        tp_p = matched
    prec = tp_p / len(pred) if pred else (1.0 if not true else 0.0)
    rec = matched / len(true) if true else 1.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ---- structure comparison ----------------------------------------------------------------
def transitive_reduction(R):
    """Cover relation of a transitively closed precedence matrix."""
    R = np.asarray(R) > 0
    cover = R.copy()
    n = R.shape[0]
    for a in range(n):
        for b in range(n):
            if not R[a, b]:
                continue
            if np.any(R[a, :] & R[:, b]):
                cover[a, b] = False
    return cover


def edge_prf(pred, true):
    p = np.asarray(pred) > 0
    t = np.asarray(true) > 0
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    prec = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def match_skills(true_labels, pred_labels, K_true, K_pred):
    """Hungarian match on the occurrence contingency table. Returns pred_slot -> true_skill."""
    C = np.zeros((K_true, K_pred))
    for t, p in zip(true_labels, pred_labels):
        if t >= 0 and p >= 0:
            C[t, p] += 1
    n = max(K_true, K_pred)
    P = np.zeros((n, n))
    P[:K_true, :K_pred] = C
    row, col = linear_sum_assignment(-P)
    mapping = {}
    for r, c in zip(row, col):
        if r < K_true and c < K_pred and C[r, c] > 0:
            mapping[int(c)] = int(r)
    return mapping, C


# ---- top-level ---------------------------------------------------------------------------
def evaluate(world, traces, decoded, K_pred, pred_local, pred_global_edges=None):
    """decoded: list of seed-index segment lists, aligned with `traces`."""
    K_true = len(world.skills)
    V = world.V

    all_true, all_pred = [], []
    b_prec, b_rec, b_f1 = [], [], []
    for tr, segs in zip(traces, decoded):
        spans = decoded_to_cpa_spans(tr, segs)
        T = len(tr.cpas)
        pred_lab = occurrence_labels(spans, T)
        true_spans = [(a, b, k) for (a, b), k in zip(tr.instance_spans, tr.skill_labels)]
        true_lab = occurrence_labels(true_spans, T)
        all_true.extend(true_lab.tolist())
        all_pred.extend(pred_lab.tolist())
        pred_cuts = [a for a, _, _ in spans if a > 0]
        p, r, f = boundary_prf(pred_cuts, tr.true_boundaries)
        b_prec.append(p), b_rec.append(r), b_f1.append(f)

    ari = adjusted_rand_score(all_true, all_pred)
    mapping, _ = match_skills(all_true, all_pred, K_true, K_pred)

    # local structure, over matched skills only
    rel_p, rel_r, rel_f, cov_f = [], [], [], []
    true_local = world.local_matrices()
    for slot, k_true in mapping.items():
        pr = np.asarray(pred_local[slot]) > 0
        tr_ = true_local[k_true] > 0
        p, r, f = edge_prf(pr, tr_)
        rel_p.append(p), rel_r.append(r), rel_f.append(f)
        cov_f.append(edge_prf(transitive_reduction(pr), transitive_reduction(tr_))[2])

    out = {
        "boundary_precision": float(np.mean(b_prec)),
        "boundary_recall": float(np.mean(b_rec)),
        "boundary_f1": float(np.mean(b_f1)),
        "skill_ari": float(ari),
        "local_rel_f1": float(np.mean(rel_f)) if rel_f else 0.0,
        "local_rel_precision": float(np.mean(rel_p)) if rel_p else 0.0,
        "local_rel_recall": float(np.mean(rel_r)) if rel_r else 0.0,
        "local_cover_f1": float(np.mean(cov_f)) if cov_f else 0.0,
        "n_matched_skills": len(mapping),
    }

    if pred_global_edges is not None:
        Gp = np.zeros((K_true, K_true))
        for a, b in pred_global_edges:
            if a in mapping and b in mapping:
                Gp[mapping[a], mapping[b]] = 1
        from hpop.inference.recurrent import hard_precedence_matrix
        Gp = hard_precedence_matrix(K_true, [(a, b) for a in range(K_true)
                                             for b in range(K_true) if Gp[a, b] > 0])
        Gt = world.global_matrix()
        p, r, f = edge_prf(Gp, Gt)
        out.update({"global_rel_precision": p, "global_rel_recall": r, "global_rel_f1": f})
    return out


def merge_only_feasible(traces):
    """Share of traces whose true boundaries are all present in the LLM seed set."""
    from hpop.synth.generator import true_seed_index_boundaries
    ok = [true_seed_index_boundaries(t)[1] for t in traces]
    return float(np.mean(ok))
