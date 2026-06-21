"""Render a CPA view directly from `opencode` output (the finalized occurrence-level schema).

Consumes `<prefix>.jsonl` (one annotation object per trajectory) + the interim traces, and writes a
self-contained HTML with: (1) a corpus summary — induced candidate-CPA frequency, decision mix,
review load, and the accepted CPA library if supplied; (2) a per-trajectory view — action tokens
coloured by induced CPA + the occurrence table + one raw occurrence record.

This is DATA-PREP output only (CPAs); skills/partial orders are downstream (HPOP) and not shown here.
Re-run after the real LLM batch and the page auto-fills.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/render_from_opencode.py \
        --opencode data/annotated/swe_rebench/cpa_A \
        --traces   data/interim/swe_rebench/pilot100.jsonl \
        --out      docs/cpa_view.html [--library rules/cpa_library_v0.1.json] [--trajectory <instance_id>]
"""
import argparse, hashlib, html, json, os
from collections import Counter

esc = lambda s: html.escape(str(s if s is not None else ""))
PALETTE = ["#7c9cff", "#5fd0a8", "#f78bb0", "#e3b341", "#d98a5f", "#9b8cff", "#56c2d6", "#c98bf7", "#8fce6b", "#e57f7f"]


def color_for(label):
    return PALETTE[int(hashlib.md5((label or "").encode()).hexdigest(), 16) % len(PALETTE)]


def eid_to_i(e):
    try:
        return int(str(e).lstrip("e"))
    except Exception:
        return None


def load_ann(prefix):
    objs = []
    with open(prefix + ".jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objs.append(json.loads(line))
    return objs


def _label(c):
    return c.get("canonical_label") or c.get("candidate_label") or "(abstain)"


def _conf(c):
    return float(c.get("label_confidence", c.get("confidence", 0.0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render CPA view from opencode output.")
    ap.add_argument("--opencode", required=True, help="prefix of <prefix>.jsonl")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument("--trajectory", default=None)
    args = ap.parse_args(argv)

    objs = load_ann(args.opencode)
    traces = {json.loads(l)["trace_id"]: json.loads(l) for l in open(args.traces) if l.strip()}
    library = json.load(open(args.library)) if args.library and os.path.exists(args.library) else []

    # ---- corpus aggregates ----
    occ = [c for o in objs for c in o.get("cpa_instances", [])]
    freq = Counter(_label(c) for c in occ)
    dec = Counter(c.get("decision") for c in occ)
    n_rev = sum(1 for c in occ if c.get("review_required"))
    n_excl = sum(len(o.get("excluded_events", [])) for o in objs)

    def bars(counter, colorf, top=14):
        items = counter.most_common(top)
        m = max((v for _, v in items), default=1)
        return "".join('<div class="brow"><div class="blab" title="{0}">{0}</div><div class="btrack"><div class="bfill" '
            'style="width:{1:.0f}px;background:{2}"></div></div><div class="bval">{3}</div></div>'.format(
            esc(k), 360 * v / m, colorf(k), v) for k, v in items)
    freqbars = bars(freq, color_for)
    decbars = bars(dec, lambda k: {"PROPOSE_NEW": "#7c9cff", "MATCH_EXISTING": "#5fd0a8", "ABSTAIN": "#e3b341"}.get(k, "#777"))
    libtable = "".join("<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
        esc(c.get("id", "")), esc(c.get("name", "")), esc((c.get("definitions") or [c.get("definition", "")])[0])) for c in library)

    # ---- pick a trajectory ----
    annby = {o.get("_instance_id") or o.get("trajectory_id"): o for o in objs}
    tid = args.trajectory or max(annby, key=lambda k: len(annby[k].get("cpa_instances", [])))
    ann = annby[tid]
    trace = traces.get(ann.get("_instance_id") or tid) or traces.get(tid)

    seq = occtab = examplejson = ""
    if trace:
        tok2lab = {}
        for c in ann.get("cpa_instances", []):
            for e in c.get("source_event_ids", []):
                i = eid_to_i(e)
                if i is not None:
                    tok2lab[i] = _label(c)
        for x in trace["action_tokens"]:
            lab = tok2lab.get(x["i"]); col = color_for(lab) if lab else "#2b333d"
            fl = "outline:2px solid #d62728;" if x.get("after_fail") else ""
            seq += '<span class="tk" style="background:{};{}" title="{} -> {}">{}</span>'.format(
                col, fl, esc(x.get("command")), esc(x.get("observation")), x["i"])
        for c in ann.get("cpa_instances", []):
            conf = _conf(c); rev = ' <span class="pill todo">review</span>' if c.get("review_required") else ""
            ev = "[{}-{}]".format(c.get("start_event_id"), c.get("end_event_id"))
            occtab += ('<tr><td><span class="dot" style="background:{}"></span>{}{}</td><td><span class="pill {}">{}</span></td>'
                '<td>{}</td><td>{:.2f}</td><td>{}</td><td>{}</td></tr>').format(
                color_for(_label(c)), esc(_label(c)), rev,
                "done" if c.get("decision") == "MATCH_EXISTING" else "demo", esc(c.get("decision")),
                esc(c.get("outcome")), conf, esc(ev), esc((c.get("procedural_function") or "")[:80]))
        if ann.get("cpa_instances"):
            examplejson = json.dumps(ann["cpa_instances"][0], indent=2, ensure_ascii=False)

    PRE = "background:#0b0e13;border:1px solid #2b333d;border-radius:8px;padding:10px 13px;overflow:auto;font-size:11.5px;line-height:1.5;color:#cdd6df"
    doc = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>HPOP — CPA view (from opencode)</title><style>
body{{margin:0;background:#0e1116;color:#e6edf3;font:14px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif}}
header{{padding:22px 26px;border-bottom:1px solid #2b333d;background:#11161d}} h1{{margin:0;font-size:21px}} .tag{{color:#9aa7b4;font-size:13px;margin-top:6px}}
.wrap{{max-width:1120px;margin:0 auto;padding:20px}} section{{margin:24px 0}} h2{{font-size:12px;text-transform:uppercase;letter-spacing:1.3px;color:#9aa7b4;margin:0 0 10px}}
.panel{{background:#161b22;border:1px solid #2b333d;border-radius:12px;padding:16px}} .lead{{color:#9aa7b4;font-size:12px;margin-bottom:10px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:860px){{.cols{{grid-template-columns:1fr}}}}
.brow{{display:flex;align-items:center;gap:8px;margin:3px 0}} .blab{{width:160px;color:#cdd6df;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.btrack{{flex:1;background:#0e1116;border-radius:3px}} .bfill{{height:13px;border-radius:3px}} .bval{{width:36px;color:#9aa7b4;font-size:11px;text-align:right}}
.stat{{display:inline-block;margin-right:22px}} .stat b{{font-size:20px;color:#7c9cff}} .stat span{{color:#9aa7b4;font-size:12px}}
.tk{{display:inline-block;width:22px;height:20px;line-height:20px;text-align:center;border-radius:4px;margin:2px;font-size:9px;color:#06090d;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:12px}} td,th{{border-bottom:1px solid #20262e;padding:5px 8px;text-align:left;vertical-align:top}}
th{{color:#9aa7b4;text-transform:uppercase;font-size:10px}} code{{background:#0b0e13;border:1px solid #2b333d;border-radius:5px;padding:0 4px;font-size:11px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}} .pill{{display:inline-block;padding:0 7px;border-radius:20px;border:1px solid #2b333d;font-size:10px}}
.done{{color:#5fd0a8;border-color:#2f5d4c}} .demo{{color:#7c9cff;border-color:#33425e}} .todo{{color:#e3b341;border-color:#5d4f25}}
</style></head><body>
<header><h1>HPOP — CPA view (rendered from opencode output)</h1><div class="tag">{src} · {ntraj} trajectories · {nocc} CPA occurrences · auto-generated by scripts/render_from_opencode.py</div></header>
<div class="wrap">
<section><h2>Corpus summary</h2><div class="panel">
  <div style="margin-bottom:12px"><span class="stat"><b>{ntraj}</b><br><span>trajectories</span></span>
    <span class="stat"><b>{nocc}</b><br><span>CPA occurrences</span></span>
    <span class="stat"><b>{ncpa}</b><br><span>distinct candidate CPAs</span></span>
    <span class="stat"><b>{nrev}</b><br><span>flagged for review</span></span>
    <span class="stat"><b>{nexcl}</b><br><span>excluded (non-action)</span></span></div>
  <div class="cols">
    <div><div class="lead">Induced candidate-CPA frequency</div>{freqbars}</div>
    <div><div class="lead">Decision mix</div>{decbars}
      {libsec}</div>
  </div></div></section>
<section><h2>Per-trajectory · <code>{tid}</code></h2><div class="panel">
  <div class="lead">action tokens coloured by induced CPA · red outline = follows a failed observation</div>
  <div>{seq}</div>
  <table style="margin-top:12px"><tr><th>candidate / canonical CPA</th><th>decision</th><th>outcome</th><th>conf</th><th>span</th><th>procedural function</th></tr>{occtab}</table>
  <div class="lead" style="margin-top:14px">Example raw occurrence record (finalized schema):</div>
  <pre style="{pre}">{examplejson}</pre>
</div></section>
<div class="tag">Data-prep only — skills &amp; partial orders are learned downstream (HPOP), not shown here.</div>
</div></body></html>""".format(
        src=esc(args.opencode), ntraj=len(objs), nocc=len(occ), ncpa=len(freq), nrev=n_rev, nexcl=n_excl,
        freqbars=freqbars, decbars=decbars,
        libsec=('<div class="lead" style="margin-top:14px">Accepted CPA library</div><table><tr><th>id</th><th>name</th><th>definition</th></tr>{}</table>'.format(libtable) if library else ""),
        tid=esc(tid), seq=seq, occtab=occtab, pre=PRE, examplejson=esc(examplejson))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote {}  ({} trajectories, {} occurrences, {} distinct CPAs)".format(args.out, len(objs), len(occ), len(freq)))


if __name__ == "__main__":
    main()
