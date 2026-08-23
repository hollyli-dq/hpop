"""Render docs/cpa_dictionary.html from rules/cpa_dictionary.json + the annotated frequency.

Keeps the codebook HTML in sync with the JSON (definitions, distinguish-from notes, in/out) and the
live occurrence distribution. Run after any dictionary change or re-annotation.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/render_cpa_dictionary.py \
        --library rules/cpa_dictionary.json \
        --instances data/annotated/swe_rebench/cpa_rule.cpa_instances.jsonl \
        --out docs/cpa_dictionary.html
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter

import sys as _sys
_sys.path.insert(0, "src")
from hpop.annotate.cpa_v2 import PHASE_ORDER, PHASE_LEVEL, PHASE_LEVEL_NAME

_LEVEL_COLOR = {"L1": "#d29922", "L2": "#7c9cff", "L3": "#3fb950", "L4": "#f778ba"}
PHASES = [(p, "{} · {} ({})".format(PHASE_LEVEL[p], p, PHASE_LEVEL_NAME[PHASE_LEVEL[p]]),
           _LEVEL_COLOR[PHASE_LEVEL[p]]) for p in PHASE_ORDER]


def esc(s):
    return html.escape(str(s or ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    lib = json.load(open(args.library))
    freq = Counter()
    trajs = set()
    for l in open(args.instances):
        if not l.strip():
            continue
        r = json.loads(l)
        freq[r["candidate_label"]] += 1
        trajs.add(r.get("instance_id"))
    total = sum(freq.values())
    mx = max(freq.values()) if freq else 1
    n_fire = sum(1 for c in lib if freq.get(c["name"], 0) > 0)

    css = (".cpa{background:#0e1116;border:1px solid #2b333d;border-radius:10px;padding:11px 13px;margin:6px 0}"
           "body{background:#0b0e13;color:#d7dde5;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:0 auto;padding:24px}"
           "h1{margin:0 0 4px}.tag{color:#8b949e;font-size:13px}"
           ".grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}"
           ".name{font-weight:700;color:#e6edf3}.bar{height:6px;background:#1f6feb;border-radius:3px;margin-top:6px}"
           ".n{color:#8b949e;font-size:12px;float:right}.d{color:#adbac7;margin:3px 0}"
           ".dist{color:#f0883e}.stat{display:inline-block;background:#11161d;border:1px solid #2b333d;border-radius:8px;padding:8px 14px;margin:4px 6px 12px 0;text-align:center}"
           "h2{font-size:16px;margin:20px 0 6px}.new{background:#1c2a1c;border:1px solid #2ea043;color:#3fb950;font-size:11px;padding:1px 6px;border-radius:6px;margin-left:6px}")

    out = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>CPA dictionary &amp; distribution</title><style>',
           css, "</style></head><body>",
           "<header><h1>CPA dictionary &amp; distribution</h1>",
           '<div class="tag">{} canonical procedural actions · silver labels over {} SWE-rebench trajectories ({} occurrences) · rules/cpa_dictionary.json</div></header>'.format(
               len(lib), len(trajs), total),
           '<div style="margin:14px 0"><span class="stat"><b>{}</b><br><span>CPAs ({} fire)</span></span>'.format(len(lib), n_fire),
           '<span class="stat"><b>{}</b><br><span>occurrences</span></span>'.format(total),
           '<span class="stat"><b>{}</b><br><span>trajectories</span></span>'.format(len(trajs)),
           '<span class="stat"><b>4</b><br><span>phases</span></span></div>']

    NEW = {"INSPECT_HISTORY", "CHECK_TYPES", "RUN_LINTER", "FORMAT_CODE", "CONFIGURE_ENVIRONMENT",
           "MANAGE_FILESYSTEM", "COMMIT_CHANGES", "MEASURE_PERFORMANCE", "COMPARE_OUTPUT",
           "INTERACTIVE_DEBUG", "INSPECT_ENVIRONMENT"}
    for key, title, color in PHASES:
        items = [c for c in lib if c.get("phase") == key]
        if not items:
            continue
        out.append('<section><h2 style="color:{}">{}</h2><div class="grid">'.format(color, esc(title)))
        for c in sorted(items, key=lambda c: -freq.get(c["name"], 0)):
            n = freq.get(c["name"], 0)
            w = int(100 * n / mx) if mx else 0
            badge = '<span class="new">NEW</span>' if c["name"] in NEW else ""
            out.append('<div class="cpa"><span class="n">{} occ</span>'
                       '<span class="name">{}</span>{}'
                       '<div class="d">{}</div>'
                       '<div class="d dist">↔ {}</div>'
                       '<div class="bar" style="width:{}%"></div></div>'.format(
                           n, esc(c["name"]), badge, esc(c.get("definition")), esc(c.get("distinguish")), w))
        out.append("</div></section>")
    out.append("</body></html>")
    open(args.out, "w", encoding="utf-8").write("".join(out))
    print("wrote {}  ({} CPAs, {} fire, {} occurrences, {} trajectories)".format(
        args.out, len(lib), n_fire, total, len(trajs)))


if __name__ == "__main__":
    main()
