"""Reusable renderer: plot a trajectory across three levels (Trajectory / CPA / Skill).

render(trace, cpa_list, dep_list, skills, outpath, note) writes a self-contained HTML and the
provenance artifacts (cpa_instances.jsonl + dependencies.jsonl) in the collaborator format.

cpa_list item: dict(label, canonical, status, tokens=[event ids], inp, outp, conf, skill)
dep_list item: (src_cpa_idx, tgt_cpa_idx, relation, conf)
skills:        list of (skill_id, name, color)

NOTE: this is a visualization driven by an annotation. For real results, feed CPA instances from
`annotate/opencode.py` output (and skills from HPOP inference); hand-authored annotations are
illustrative mock-ups, not pipeline output.
"""
import html, json, os

FAMC = {"read": "#7c9cff", "execute": "#f78bb0", "search": "#e3b341", "edit": "#5fd0a8",
        "test": "#d98a5f", "think": "#9aa7b4", "submit": "#9b8cff", "install": "#56c2d6", "other": "#777"}
RELC = {"DATA_FLOW": "#1f77b4", "STATE_CHANGE": "#2ca02c", "PRECONDITION": "#ff7f0e",
        "VERIFY_OF": "#9467bd", "REPAIR_OF": "#d62728", "ELABORATES": "#8c564b", "INCOMPARABLE": "#7f7f7f"}
esc = lambda s: html.escape(str(s if s is not None else ""))


def render(trace, cpa_list, dep_list, skills, outpath, note=""):
    toks = trace["action_tokens"]; tid = trace["trace_id"]
    skillcol = {s[0]: s[2] for s in skills}
    tok2skill = {}
    for c in cpa_list:
        for e in c["tokens"]:
            tok2skill[e] = c["skill"]
    cid = lambda i: "{}-CPA{:03d}".format(tid, i + 1)

    # provenance artifacts
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    base = os.path.splitext(outpath)[0]
    with open(base + ".cpa_instances.jsonl", "w") as f:
        for i, c in enumerate(cpa_list):
            f.write(json.dumps({"trajectory_id": trace.get("trajectory_id"), "cpa_instance_id": cid(i),
                "source_event_ids": c["tokens"], "proposed_cpa": c["label"], "canonical_cpa": c.get("canonical"),
                "status": c["status"], "input_artifact": c.get("inp", ""), "output_artifact": c.get("outp", ""),
                "confidence": c["conf"], "human_review": bool(c["status"] == "PROPOSE_NEW" or c["conf"] < 0.7)}) + "\n")
    with open(base + ".dependencies.jsonl", "w") as f:
        for s, t, rel, cf in dep_list:
            f.write(json.dumps({"trajectory_id": trace.get("trajectory_id"), "source_cpa_id": cid(s),
                "target_cpa_id": cid(t), "proposed_relation": rel, "confidence": cf}) + "\n")

    # A: trajectory
    trj = ""
    for x in toks:
        sk = tok2skill.get(x["i"], "")
        bb = "border-bottom:3px solid {};".format(skillcol.get(sk, "#333"))
        fl = "outline:2px solid #d62728;" if x.get("after_fail") else ""
        trj += '<span class="tk" style="background:{};{}{}" title="{} -> {}">{}</span>'.format(
            FAMC.get(x["tool_family"], "#777"), bb, fl, esc(x["command"]), esc(x["observation"]), x["i"])

    # B: CPA table
    cparows = ""
    for i, c in enumerate(cpa_list):
        badge = "MATCH" if c["status"] == "MATCH" else "NEW"; bcl = "done" if c["status"] == "MATCH" else "demo"
        chips = "".join('<span class="ix">{}</span>'.format(e) for e in c["tokens"])
        rev = ' <span class="pill todo">review</span>' if (c["status"] == "PROPOSE_NEW" or c["conf"] < 0.7) else ""
        cparows += ('<tr><td><code>{}</code></td><td><span class="dot" style="background:{}"></span>{}{}</td>'
            '<td><span class="pill {}">{}</span></td><td>{}</td><td>{} → <b>{}</b></td><td>{}</td></tr>').format(
            esc(cid(i)), skillcol[c["skill"]], esc(c["label"]), rev, bcl, badge, c["conf"], esc(c.get("inp", "")), esc(c.get("outp", "")), chips)

    # C: skill DAGs
    def skill_svg(sid):
        idxs = [i for i, c in enumerate(cpa_list) if c["skill"] == sid]
        pos = {i: (70 + j * 140, 55 + (38 if j % 2 else 0)) for j, i in enumerate(idxs)}
        w = max(220, 70 + len(idxs) * 140)
        nodes = ""
        for i in idxs:
            x, y = pos[i]
            nodes += ('<circle cx="{}" cy="{}" r="17" fill="#10151c" stroke="{}" stroke-width="2"/>'
                '<text x="{}" y="{}" text-anchor="middle" fill="#e6edf3" font-size="10" font-weight="700">c{}</text>'
                '<text x="{}" y="{}" text-anchor="middle" fill="#9aa7b4" font-size="9">{}</text>').format(
                x, y, skillcol[sid], x, y + 3, i, x, y + 31, esc(cpa_list[i]["label"].split()[0]))
        edges = ""
        for s, t, rel, cf in dep_list:
            if s in pos and t in pos:
                x1, y1 = pos[s]; x2, y2 = pos[t]; dash = "4 3" if rel == "INCOMPARABLE" else "0"
                head = "" if rel == "INCOMPARABLE" else "marker-end='url(#ar)'"
                edges += '<path d="M{},{} C{},{} {},{} {},{}" fill="none" stroke="{}" stroke-width="1.8" stroke-dasharray="{}" {}/>'.format(
                    x1 + 17, y1, (x1 + x2) / 2, y1 - 28, (x1 + x2) / 2, y2 - 28, x2 - 17, y2, RELC[rel], dash, head)
        return ('<svg viewBox="0 0 {} 120"><defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" '
            'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#9aa7b4"/></marker></defs>{}{}</svg>').format(w, edges, nodes)
    skillblocks = "".join('<div class="skill"><h4><span class="dot" style="background:{}"></span>{} '
        '<span class="sub">(skill {} — local partial order)</span></h4>{}</div>'.format(c, esc(n), sid, skill_svg(sid))
        for sid, n, c in skills)

    famleg = "".join('<span class="chip"><span class="sw" style="background:{}"></span>{}</span>'.format(FAMC[k], k)
                     for k in FAMC if any(x["tool_family"] == k for x in toks))
    relleg = "".join('<span class="chip"><span class="sw" style="background:{}"></span>{}</span>'.format(RELC[k], k) for k in RELC)
    doc = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>HPOP three levels</title><style>
body{{margin:0;background:#0e1116;color:#e6edf3;font:14px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif}}
header{{padding:20px 26px;border-bottom:1px solid #2b333d;background:#11161d}} h1{{margin:0;font-size:20px}}
.tag{{color:#9aa7b4;font-size:13px;margin-top:6px}} .wrap{{max-width:1180px;margin:0 auto;padding:20px}}
.banner{{background:#1c1a10;border:1px solid #5d4f25;color:#e3b341;border-radius:10px;padding:10px 14px;font-size:12.5px;margin:14px 0}}
section{{margin:22px 0}} h2{{font-size:12px;text-transform:uppercase;letter-spacing:1.3px;color:#9aa7b4;margin:0 0 6px}}
.lead{{color:#9aa7b4;font-size:12px;margin-bottom:10px}} .panel{{background:#161b22;border:1px solid #2b333d;border-radius:12px;padding:16px}}
.tk{{display:inline-block;width:22px;height:20px;line-height:20px;text-align:center;border-radius:4px;margin:2px;font-size:9px;color:#06090d;font-weight:700}}
.chip{{display:inline-block;margin:3px 8px 0 0;font-size:11px;color:#cdd6df}} .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
table{{width:100%;border-collapse:collapse;font-size:12px}} td,th{{border-bottom:1px solid #20262e;padding:5px 8px;text-align:left;vertical-align:top}}
th{{color:#9aa7b4;text-transform:uppercase;font-size:10px}} code{{background:#0b0e13;border:1px solid #2b333d;border-radius:5px;padding:0 4px;font-size:11px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}} .ix{{display:inline-block;background:#0b0e13;border:1px solid #2b333d;border-radius:3px;padding:0 4px;margin:1px;font-size:10px;color:#9aa7b4}}
.pill{{display:inline-block;padding:0 7px;border-radius:20px;border:1px solid #2b333d;font-size:10px}} .done{{color:#5fd0a8;border-color:#2f5d4c}} .demo{{color:#7c9cff;border-color:#33425e}} .todo{{color:#e3b341;border-color:#5d4f25}}
.skill{{background:#0e1116;border:1px solid #2b333d;border-radius:10px;padding:10px 12px;margin:10px 0;overflow-x:auto}} .skill h4{{margin:0 0 4px;font-size:13px}} .sub{{color:#9aa7b4;font-weight:400;font-size:11px}} svg{{max-width:100%}}
</style></head><body>
<header><h1>HPOP — Trajectory · CPA · Skill</h1><div class="tag"><code>{tid}</code> · repo {repo} · {n} actions · resolved={res}</div></header>
<div class="wrap"><div class="banner">{note}</div>
<section><h2>① Trajectory — raw OpenHands tool calls</h2><div class="lead">box = tool call (number=order), fill=tool family, underline=owning skill, red outline=follows a failed observation.</div>
<div class="panel"><div>{trj}</div><div style="margin-top:10px">{famleg}</div></div></section>
<section><h2>② CPA — induced canonical procedural actions</h2><div class="lead">tokens grouped into procedural actions; NEW=PROPOSE_NEW, MATCH=recurs. Labels induced, not predefined.</div>
<div class="panel"><table><tr><th>cpa id</th><th>proposed CPA (· skill)</th><th>status</th><th>conf</th><th>input → output</th><th>tokens</th></tr>{cparows}</table></div></section>
<section><h2>③ Skill — local partial orders over CPA instances</h2><div class="lead">edge colour = dependency relation; INCOMPARABLE (dashed)=order-free/parallel.</div>
<div class="panel"><div style="margin-bottom:8px">{relleg}</div>{skillblocks}</div></section>
</div></body></html>""".format(tid=esc(tid), repo=esc(trace.get("repo")), n=len(toks), res=trace.get("resolved"),
        note=note, trj=trj, famleg=famleg, cparows=cparows, relleg=relleg, skillblocks=skillblocks)
    with open(outpath, "w") as f:
        f.write(doc)
    return outpath
