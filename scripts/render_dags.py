"""Render the learned partial orders as inline SVG DAGs and inject them into docs/experiments.html.

Three figures, all generated from result data rather than drawn by hand:

  A. the skill library learned from the real SWE-rebench pilot, one DAG per active skill;
  B. ground truth vs recovered local posets on one synthetic seed, with every edge classified
     recovered / missed / spurious (this is what "local edge F1 0.848" looks like);
  C. a real decoded skill instance replayed through the recurrent validity state, showing a
     re-execution cascade on an acyclic latent order.

The script replaces everything between the <!--DAG_FIGURES--> and <!--/DAG_FIGURES--> markers, so it
is safe to re-run after new results land.

Run:
    PYTHONPATH=src .venv/bin/python scripts/render_dags.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "experiments.html"
REAL = ROOT / "data" / "experiments" / "real_pilot.json"

# Validated against the dark surface #161b22 (six checks: chroma, contrast, CVD, normal floor).
# Status colours are always paired with a line style + legend label, never colour alone.
INK, MUT, LINE, PANEL = "#e6edf3", "#9aa7b4", "#2b333d", "#1c232c"
ACC, GOOD, BAD, WARN = "#7c9cff", "#5fd0a8", "#f0776c", "#e3b341"
EDGE = "#6b7785"

NODE_H, LAYER_GAP, ROW_GAP, PAD = 30, 152, 40, 12


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short(label):
    """Compact CPA names so nodes stay readable at 11px."""
    return {
        "EXPLORE_REPOSITORY": "EXPLORE", "LOCATE_CODE": "LOCATE", "READ_SOURCE": "READ",
        "REPRODUCE_ISSUE": "REPRODUCE", "RUN_TEST_SUITE": "RUN_TESTS",
        "DIAGNOSE_FAILURE": "DIAGNOSE", "EDIT_SOURCE": "EDIT", "WRITE_TEST": "WRITE_TEST",
        "VERIFY_FIX": "VERIFY", "WRITE_REPRODUCTION_SCRIPT": "WRITE_REPRO",
        "SUBMIT_SOLUTION": "SUBMIT", "CLEANUP_ARTIFACTS": "CLEANUP",
        "INSPECT_CHANGES": "INSPECT", "INSTALL_DEPENDENCY": "INSTALL",
        "REVERT_CHANGE": "REVERT", "BUILD_PROJECT": "BUILD",
    }.get(label, label)


# ---------------------------------------------------------------------------------------------
def layer_nodes(nodes, edges):
    """Longest-path layering: rank(v) = longest chain of edges ending at v."""
    rank = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for a, b in edges:
            if a in rank and b in rank and rank[b] < rank[a] + 1:
                rank[b] = rank[a] + 1
                changed = True
        if not changed:
            break
    layers = {}
    for n in nodes:
        layers.setdefault(rank[n], []).append(n)
    return [layers[r] for r in sorted(layers)]


def node_width(label):
    return max(76, 8 + 7.0 * len(short(label)))


def dag_svg(nodes, edges, classified=None, title=None, subtitle=None, width=None):
    """One DAG. `classified` maps (a,b) -> 'recovered'|'missed'|'spurious'."""
    layers = layer_nodes(nodes, edges)
    pos, max_w = {}, 0
    for li, layer in enumerate(layers):
        for ri, n in enumerate(layer):
            w = node_width(n)
            x = PAD + li * LAYER_GAP
            y = PAD + ri * (NODE_H + ROW_GAP) + (24 if title else 0)
            pos[n] = (x, y, w)
            max_w = max(max_w, x + w)
    rows = max(len(l) for l in layers) if layers else 1
    h = PAD * 2 + rows * (NODE_H + ROW_GAP) - ROW_GAP + (24 if title else 0) + (14 if subtitle else 0)
    w = width or (max_w + PAD)

    # Keep the natural size and let the card scroll: scaling a 680px DAG into a 340px column would
    # render 11px labels at ~5px.
    out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           f'style="min-width:{w:.0f}px;display:block" '
           f'role="img" aria-label="{esc(title or "partial order")}">']
    out.append(
        '<defs>'
        f'<marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
        f'orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="{EDGE}"/></marker>'
        f'<marker id="ahg" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
        f'orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="{GOOD}"/></marker>'
        f'<marker id="ahr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
        f'orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="{BAD}"/></marker>'
        f'<marker id="ahw" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
        f'orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="{WARN}"/></marker>'
        '</defs>')
    if title:
        out.append(f'<text x="{PAD}" y="14" fill="{INK}" font-size="12" font-weight="700">{esc(title)}</text>')
    if subtitle:
        out.append(f'<text x="{PAD}" y="{h - 3:.0f}" fill="{MUT}" font-size="10.5">{esc(subtitle)}</text>')

    style = {"recovered": (GOOD, "none", "url(#ahg)", 2.0),
             "missed": (BAD, "5 4", "url(#ahr)", 2.0),
             "spurious": (WARN, "1.5 3.5", "url(#ahw)", 2.0)}
    for a, b in edges:
        if a not in pos or b not in pos:
            continue
        ax, ay, aw = pos[a]
        bx, by, _ = pos[b]
        x1, y1 = ax + aw, ay + NODE_H / 2
        x2, y2 = bx - 3, by + NODE_H / 2
        col, dash, mark, sw = style.get((classified or {}).get((a, b)), (EDGE, "none", "url(#ah)", 1.6))
        mx = (x1 + x2) / 2
        out.append(f'<path d="M{x1:.0f} {y1:.0f} C{mx:.0f} {y1:.0f} {mx:.0f} {y2:.0f} {x2:.0f} {y2:.0f}" '
                   f'fill="none" stroke="{col}" stroke-width="{sw}" stroke-dasharray="{dash}" '
                   f'marker-end="{mark}"/>')
    for n, (x, y, wd) in pos.items():
        out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{wd:.0f}" height="{NODE_H}" rx="7" '
                   f'fill="{PANEL}" stroke="{LINE}"/>'
                   f'<text x="{x + wd / 2:.0f}" y="{y + NODE_H / 2 + 4:.0f}" fill="{INK}" font-size="11" '
                   f'text-anchor="middle" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">'
                   f'{esc(short(n))}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
def figure_real_library():
    d = json.loads(REAL.read_text())
    skills = d["skills"]
    total = sum(v["usage"] for v in skills.values())
    cards = []
    for k, v in sorted(skills.items(), key=lambda kv: -kv[1]["usage"]):
        edges = [tuple(e) for e in v["edges"]]
        nodes = list(dict.fromkeys([n for e in edges for n in e] + v["composition"][:4]))
        share = v["usage"] / total
        sub = "no learned order — genuinely interchangeable" if not edges else \
              f"{len(edges)} learned cover edge{'s' if len(edges) > 1 else ''}"
        cards.append(
            f'<div class="dagcard"><div class="dagh"><b>skill {k}</b>'
            f'<span class="usage">{share:.1%} of instances</span></div>'
            f'{dag_svg(nodes, edges, subtitle=sub)}</div>')
    return ('<div class="daggrid">' + "".join(cards) + "</div>")


def figure_synthetic_recovery():
    """Fit one synthetic seed and classify every ground-truth / learned edge."""
    from hpop.eval.metrics import evaluate, match_skills, occurrence_labels, decoded_to_cpa_spans
    from hpop.inference.hpop import HPOP, HPOPConfig
    from hpop.synth.generator import sample_corpus, seeds_of

    world, traces = sample_corpus(seed=0, n_traces=40, K_true=4, V=12)
    split = int(0.7 * len(traces))
    corpus = [seeds_of(t) for t in traces[:split]]
    m = HPOP(HPOPConfig(V=12, K_max=6, D_max=8), rng=np.random.default_rng(0))
    m.fit(corpus, iters=12, warmup=3)

    decoded = [m.decode(s) for s in corpus]
    all_true, all_pred = [], []
    for tr, segs in zip(traces[:split], decoded):
        T = len(tr.cpas)
        all_pred.extend(occurrence_labels(decoded_to_cpa_spans(tr, segs), T).tolist())
        all_true.extend(occurrence_labels(
            [(a, b, k) for (a, b), k in zip(tr.instance_spans, tr.skill_labels)], T).tolist())
    mapping, _ = match_skills(all_true, all_pred, 4, 6)
    res = evaluate(world, traces[:split], decoded, 6, m.D, m.global_structure(corpus))

    from hpop.eval.metrics import transitive_reduction
    vocab = world.vocab
    cards = []
    for slot, k_true in sorted(mapping.items(), key=lambda kv: kv[1]):
        true_cov = transitive_reduction(world.local_matrices()[k_true] > 0)
        pred_cov = transitive_reduction(np.asarray(m.D[slot]) > 0)
        t_edges = {(vocab[a], vocab[b]) for a, b in zip(*np.where(true_cov))}
        p_edges = {(vocab[a], vocab[b]) for a, b in zip(*np.where(pred_cov))}
        cls = {}
        for e in t_edges & p_edges:
            cls[e] = "recovered"
        for e in t_edges - p_edges:
            cls[e] = "missed"
        for e in p_edges - t_edges:
            cls[e] = "spurious"
        edges = list(cls)
        nodes = list(dict.fromkeys([n for e in edges for n in e] +
                                   [vocab[r] for r in world.skills[k_true].roles]))
        n_ok, n_miss, n_sp = (sum(1 for v in cls.values() if v == x)
                              for x in ("recovered", "missed", "spurious"))
        cards.append(
            f'<div class="dagcard"><div class="dagh"><b>true skill {k_true}</b>'
            f'<span class="usage">→ slot {slot}</span></div>'
            f'{dag_svg(nodes, edges, classified=cls, subtitle=f"{n_ok} recovered · {n_miss} missed · {n_sp} spurious")}'
            f'</div>')
    return ('<div class="daggrid">' + "".join(cards) + "</div>"), res


def figure_recurrent_trace():
    """A latent DAG plus a repeated execution, with the validity state drawn under each step."""
    from hpop.inference.recurrent import RecurrentFrontier, hard_precedence_matrix
    roles = ["READ", "EDIT", "RUN_TESTS", "DIAGNOSE"]
    edges = [(0, 1), (1, 2), (2, 3)]
    D = hard_precedence_matrix(4, edges)
    m = RecurrentFrontier(D, omega=3.0, beta=1.5, lam_rep=1.5, lam_back=0.5, eps=0.02)
    trace = [0, 1, 2, 3, 1, 2, 3, 1, 2]          # edit-test-diagnose repair loop

    q = np.zeros(4)
    states = [q.copy()]
    for y in trace:
        q = m.update(q, y)
        states.append(q.copy())

    cw, x0, y0, rh = 62, 108, 34, 22
    w = x0 + cw * len(trace) + 16
    h = y0 + rh * 4 + 54
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'style="min-width:{w}px;display:block" role="img" '
           f'aria-label="validity state over a repeated execution">']
    out.append(f'<text x="0" y="14" fill="{INK}" font-size="12" font-weight="700">'
               f'Validity state q_t over a repeated execution</text>')
    out.append(f'<text x="0" y="{y0 - 8}" fill="{MUT}" font-size="10">step</text>')
    for t, y in enumerate(trace):
        out.append(f'<text x="{x0 + cw * t + cw / 2:.0f}" y="{y0 - 8}" fill="{MUT}" font-size="10" '
                   f'text-anchor="middle">{t + 1}</text>')
    for r, name in enumerate(roles):
        yy = y0 + r * rh
        out.append(f'<text x="{x0 - 8}" y="{yy + 15}" fill="{INK}" font-size="10.5" text-anchor="end" '
                   f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace">{name}</text>')
        for t in range(len(trace)):
            v = states[t + 1][r]
            executed = trace[t] == r
            fill = GOOD if v > 0.99 else (WARN if v > 0.05 else PANEL)
            op = 1.0 if v > 0.99 else (0.35 + 0.5 * v if v > 0.05 else 1.0)
            out.append(f'<rect x="{x0 + cw * t + 4:.0f}" y="{yy + 3}" width="{cw - 8}" height="{rh - 6}" '
                       f'rx="4" fill="{fill}" fill-opacity="{op:.2f}" stroke="{ACC if executed else LINE}" '
                       f'stroke-width="{2 if executed else 1}"/>')
            if executed:
                out.append(f'<text x="{x0 + cw * t + cw / 2:.0f}" y="{yy + rh / 2 + 4:.0f}" '
                           f'fill="#0e1116" font-size="9.5" font-weight="700" text-anchor="middle">RUN</text>')
    ly = y0 + 4 * rh + 16
    out.append(f'<text x="0" y="{ly + 10}" fill="{MUT}" font-size="10.5">'
               f'RUN = executed at this step · solid = output valid · faded = stale after invalidation'
               f' · empty = never produced</text>')
    out.append(f'<text x="0" y="{ly + 26}" fill="{MUT}" font-size="10.5">'
               f'Steps 5–9 re-execute EDIT → RUN_TESTS → DIAGNOSE. The latent order stays acyclic;'
               f' only the execution recurs.</text>')
    out.append("</svg>")
    return "".join(out), dag_svg([roles[i] for i in range(4)],
                                 [(roles[a], roles[b]) for a, b in edges],
                                 subtitle="latent local partial order (acyclic)")


# ---------------------------------------------------------------------------------------------
CSS = """
<style>
  .daggrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;margin-top:12px}
  .dagcard{background:#12171e;border:1px solid #2b333d;border-radius:11px;padding:12px 14px;overflow-x:auto}
  .dagh{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px;font-size:13px}
  .dagh .usage{color:#9aa7b4;font-size:11.5px}
  .daglegend{display:flex;flex-wrap:wrap;gap:18px;margin-top:12px;font-size:12px;color:#9aa7b4;align-items:center}
  .daglegend span{display:inline-flex;align-items:center;gap:7px}
  .daglegend svg{display:block}
</style>
"""


def legend():
    def swatch(col, dash, label):
        return (f'<span><svg width="34" height="10" aria-hidden="true">'
                f'<path d="M1 5 H31" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>'
                f'<path d="M27 1 L33 5 L27 9 z" fill="{col}"/></svg>{label}</span>')
    return ('<div class="daglegend">'
            + swatch(GOOD, "none", "recovered (solid)")
            + swatch(BAD, "5 4", "missed (dashed)")
            + swatch(WARN, "1.5 3.5", "spurious (dotted)")
            + "</div>")


def main():
    print("rendering figure A — learned library from the real pilot ...", flush=True)
    fig_a = figure_real_library()
    print("rendering figure C — recurrent validity state ...", flush=True)
    fig_c_trace, fig_c_dag = figure_recurrent_trace()
    print("rendering figure B — synthetic ground truth vs recovered (fitting one seed) ...", flush=True)
    fig_b, res = figure_synthetic_recovery()

    body = f"""{CSS}
  <section><h2>Learned structure — what the partial orders actually look like</h2><div class="panel">

    <h3 style="margin-top:0">A · Skill library learned from the real SWE-rebench pilot</h3>
    <div style="color:var(--mut);font-size:12.5px">
      Every DAG below is the local partial order HPOP learned for one library slot, drawn straight from
      <code>data/experiments/real_pilot.json</code>. Nodes are CPA roles, edges are learned cover relations
      (<i>must precede</i>); anything not connected is inferred to be <b>incomparable</b> — executable in either order.
      No supervision: the annotator supplies CPA labels only, never dependencies.
    </div>
    {fig_a}

    <h3>B · Ground truth vs recovered — what local edge F1 {res['local_rel_f1']:.3f} looks like</h3>
    <div style="color:var(--mut);font-size:12.5px">
      One synthetic seed, skills matched to ground truth by Hungarian assignment on occurrence labels. Each edge is
      classified against the true poset. Line style carries the same information as colour, so the figure survives
      greyscale and colour-vision deficiency.
    </div>
    {legend()}
    {fig_b}

    <h3>C · Why the latent graph stays acyclic while execution repeats</h3>
    <div style="color:var(--mut);font-size:12.5px">
      The repair loop is <i>not</i> a cycle in the partial order. It is re-execution driven by the validity state
      <code>q_t</code>: running an upstream role invalidates everything it dominates, and the stale roles must be
      recomputed. This is the component whose removal costs 0.17 local edge F1 in experiment ①.
    </div>
    <div class="daggrid" style="grid-template-columns:minmax(300px,0.9fr) minmax(420px,1.6fr)">
      <div class="dagcard">{fig_c_dag}</div>
      <div class="dagcard">{fig_c_trace}</div>
    </div>
    <div class="warnbox" style="margin-top:14px">
      <b>What this figure also exposes.</b> Under the manuscript's <code>J̃ = D<sub>U</sub>·σ(ω)</code>, invalidation
      flows only <i>forward</i> along precedence — so the jump back to <code>EDIT</code> at step 5 is
      <b>never predicted</b>, only charged λ<sub>rep</sub>. A failed test cannot invalidate the edit that preceded it.
      That is finding (b), and this is what it looks like.
    </div>
  </div></section>
"""
    html = HTML.read_text()
    a, b = html.index("<!--DAG_FIGURES-->"), html.index("<!--/DAG_FIGURES-->")
    HTML.write_text(html[:a] + "<!--DAG_FIGURES-->\n" + body + "  " + html[b:])
    print(f"injected {len(body)} chars into {HTML}")


if __name__ == "__main__":
    main()
