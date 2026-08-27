#!/usr/bin/env python3
"""Validate the paper evidence pack.

Read-only consistency checker for ``paper/evidence``, ``paper/drafts`` and
``paper/tables``.  It never launches, stops, resumes or inspects a running
experiment, never touches an inference kernel, and never reads a sealed test
artifact.

Checks performed
----------------
1.  every numerical ledger entry names a source artifact that exists on disk
2.  every commit referenced by the ledger and the manifest exists in git history
3.  every corpus / benchmark hash in the manifest matches its own manifest file
4.  no pending result is populated with a number
5.  no TaskBench claim is labelled full HPOP
6.  no tau3 test artifact is read by the pack
7.  all generated LaTeX fragments compile in a minimal wrapper (skipped, with a
    clear message, when no LaTeX toolchain is installed)
8.  all tables round consistently (three decimals unless justified)
9.  no undefined citation placeholder is silently introduced

Usage
-----
    python3 scripts/validate_paper_evidence.py            # all checks
    python3 scripts/validate_paper_evidence.py --no-latex # skip check 7
    python3 scripts/validate_paper_evidence.py -v         # list every pass

Exit status is 0 when every check passes or is skipped, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "paper" / "evidence"
DRAFTS = REPO / "paper" / "drafts"
TABLES = REPO / "paper" / "tables"

MANIFEST = EVIDENCE / "evidence_manifest.json"
LEDGER_CSV = EVIDENCE / "result_ledger.csv"
FIGURE_MANIFEST = EVIDENCE / "taskbench_figure_manifest.json"

# Artifacts live in sibling worktrees; a repo-relative path is resolved against
# whichever worktree actually holds it.
WORKTREE_ROOTS = [
    REPO,
    REPO.parent / "hpop-taskbench",
    REPO.parent / "hpop-tau3",
]

# Paths this validator must never open, and never encourage anything else to.
FORBIDDEN_READS = [
    "tau3_retail_hpop_pilot/test_trajectories.jsonl",
    "matched_condition_c/formal_chains",
    "matched_condition_c/formal_registration.json",
]

PENDING_CLAIMS = {"C7", "C10"}

# Tables whose cells are metric values. The strict three-decimal rule and the
# strict pending-row rule apply to these. Prose tables are listed separately and
# their exemption is printed, never silent.
NUMERIC_RESULT_TABLES = {
    "table_matched_synthetic.tex",
    "table_taskbench_main.tex",
    "table_taskbench_appendix.tex",
}
PROSE_TABLES = {
    # Cells are sentences. Its Condition C row deliberately carries frozen
    # DESIGN facts (proposal scales, cadence) beside a pending location cell,
    # and its prose quotes z-scores and scientific notation.
    "table_claim_scope.tex",
}
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class Report:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.failures: list[str] = []
        self.skips: list[str] = []
        self.checks = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks += 1
        if ok:
            if self.verbose:
                print(f"  pass  {name}")
        else:
            self.failures.append(f"{name}: {detail}")
            print(f"  FAIL  {name}\n        {detail}")
        return ok

    def skip(self, name: str, why: str) -> None:
        self.skips.append(f"{name}: {why}")
        print(f"  skip  {name} — {why}")

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def resolve_artifact(rel: str) -> Path | None:
    """Resolve a repo-relative artifact path against every known worktree.

    Ledger source fields sometimes carry a ``::`` field suffix or a trailing
    parenthetical; strip those before resolving.
    """
    rel = rel.split("::")[0].strip()
    rel = re.sub(r"\s*\(.*\)\s*$", "", rel).strip()
    if not rel or rel in {"n/a", ""}:
        return None
    # Glob-ish ledger entries (chain*.json, per_skill[*]) name a directory.
    if "*" in rel:
        rel = rel.rsplit("/", 1)[0]
    for root in WORKTREE_ROOTS:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def git_commits() -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "--all", "--format=%H"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def commit_exists(ref: str, known: set[str]) -> bool:
    ref = ref.strip()
    if not ref:
        return False
    if any(full.startswith(ref) for full in known):
        return True
    # Fall back to git itself for annotated/abbreviated forms.
    res = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{ref}^{{commit}}"],
        capture_output=True,
    )
    return res.returncode == 0


def iter_commit_refs(text: str) -> list[str]:
    return re.findall(r"\b[0-9a-f]{7,40}\b", text)


def tex_files() -> list[Path]:
    return sorted(DRAFTS.glob("*.tex")) + sorted(TABLES.glob("*.tex"))


def tabular_bodies(text: str) -> list[str]:
    """Return the row area of every tabular/tabularx environment."""
    out = []
    pattern = re.compile(
        r"\\begin\{(tabular\*?|tabularx)\}(.*?)\\end\{\1\}", re.S
    )
    for m in pattern.finditer(text):
        out.append(m.group(2))
    return out


# Numbers that are legitimately not three-decimal metric values.
EXEMPT_NUMBER_PATTERNS = [
    # scientific-notation mantissas: 2.6\times10^{-15}
    re.compile(r"\d+(?:\.\d+)?\s*\\times\s*10\^\{[^}]*\}"),
    # percentages: -52.4\%
    re.compile(r"[-+]?\d+(?:\.\d+)?\s*\\?%"),
    # fixed model settings and hyperparameters: \rho_0 = 0.5, \sigma_U = 1.0
    re.compile(r"(?:\\[A-Za-z]+(?:_[A-Za-z0-9{}]+)?|[A-Za-z])\s*=\s*[-+]?\d+(?:\.\d+)?"),
    # z-scores, conventionally two decimals: z = +2.86, (z = +5.76)
    re.compile(r"z\s*=?\s*[-+]?\d+(?:\.\d+)?"),
    # digit-grouped counts: 500{,}000
    re.compile(r"\d+\{,\}\d+"),
    # ratios and fractions of counts: 132/132, 24/26, 17/18
    re.compile(r"\d+\s*/\s*\d+"),
    # trace-length and node-count labels: J = 48, m = 5, d = 3 (covered above),
    # plus bare superscripts and subscripts
    re.compile(r"\^\{?-?\d+\}?"),
    re.compile(r"_\{?\d+\}?"),
    # arXiv identifiers and dates in prose tables
    re.compile(r"arXiv:\d+\.\d+"),
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
]


def strip_exempt_numbers(text: str) -> str:
    for pat in EXEMPT_NUMBER_PATTERNS:
        text = pat.sub(" ", text)
    return text


def strip_tex_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        idx = 0
        while True:
            idx = line.find("%", idx)
            if idx == -1:
                out.append(line)
                break
            if idx > 0 and line[idx - 1] == "\\":
                idx += 1
                continue
            out.append(line[:idx])
            break
    return "\n".join(out)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_ledger_sources(rep: Report, rows: list[dict]) -> None:
    rep.section("1. Ledger source artifacts exist")
    missing: list[str] = []
    checked = 0
    for i, row in enumerate(rows, start=2):
        src = (row.get("source_artifact") or "").strip()
        if not src:
            # Only pending placeholder rows may omit a source.
            if str(row.get("value", "")).strip().upper() != "PENDING":
                missing.append(f"line {i}: no source_artifact and value is not PENDING")
            continue
        checked += 1
        if resolve_artifact(src) is None:
            missing.append(f"line {i}: {src}")
    rep.check(
        f"every ledger source artifact resolves ({checked} checked)",
        not missing,
        "; ".join(missing[:10]),
    )


def check_commits(rep: Report, rows: list[dict], manifest: dict) -> None:
    rep.section("2. Referenced commits exist in git history")
    known = git_commits()

    bad: list[str] = []
    refs = set()
    for row in rows:
        for ref in iter_commit_refs(row.get("commit") or ""):
            refs.add(ref)
    for ref in sorted(refs):
        if not commit_exists(ref, known):
            bad.append(f"ledger: {ref}")
    rep.check(f"ledger commits exist ({len(refs)} distinct)", not bad, "; ".join(bad))

    mbad: list[str] = []
    mrefs = set()
    for claim in manifest["claims"]:
        for ref in iter_commit_refs(claim.get("commit") or ""):
            mrefs.add(ref)
    for hashes in manifest["corpus_and_benchmark_hashes"].values():
        for key, val in hashes.items():
            if key.endswith("_commit") and isinstance(val, str):
                mrefs.update(iter_commit_refs(val))
    # Upstream third-party commits are not in this repository's history.
    upstream = {
        "7624cf388b47334ff8a0868e7d862dde18cfda86",
        "3b005ddbdb4127c60cf2100e894807b6f6786a7a",
        "157415de88947ca2efec6ab16c6a6820fb6dc962",
    }
    for ref in sorted(mrefs - upstream):
        if not commit_exists(ref, known):
            mbad.append(f"manifest: {ref}")
    rep.check(
        f"manifest commits exist ({len(mrefs - upstream)} local, {len(upstream & mrefs)} upstream exempt)",
        not mbad,
        "; ".join(mbad),
    )


def check_hashes(rep: Report, manifest: dict) -> None:
    rep.section("3. Corpus / benchmark hashes match their manifests")
    problems: list[str] = []
    verified = 0
    for name, block in manifest["corpus_and_benchmark_hashes"].items():
        manifest_rel = block.get("manifest")
        if not manifest_rel:
            continue
        path = resolve_artifact(manifest_rel)
        if path is None:
            problems.append(f"{name}: manifest file {manifest_rel} not found")
            continue
        # A hash may be recorded in the primary manifest or in any manifest the
        # block explicitly declares alongside it.
        haystacks: list[tuple[str, str]] = []
        for rel in [manifest_rel, *block.get("additional_manifests", [])]:
            p = resolve_artifact(rel)
            if p is None:
                problems.append(f"{name}: manifest file {rel} not found")
                continue
            haystacks.append((rel, json.dumps(json.loads(p.read_text()))))
        if not haystacks:
            continue
        for key, claimed in block.items():
            if not (key.endswith("sha256") or key.endswith("_hash")):
                continue
            if not isinstance(claimed, str) or len(claimed) != 64:
                continue
            if any(claimed in blob for _, blob in haystacks):
                verified += 1
            else:
                where = ", ".join(rel for rel, _ in haystacks)
                problems.append(f"{name}.{key} = {claimed[:16]}… absent from {where}")
    rep.check(f"declared hashes present in their manifests ({verified} verified)", not problems, "; ".join(problems))


def check_pending_unpopulated(rep: Report, rows: list[dict], manifest: dict) -> None:
    rep.section("4. No pending result carries a number")
    problems: list[str] = []

    for claim in manifest["claims"]:
        cid = claim["claim_id"]
        if claim["status"] != "pending":
            if cid in PENDING_CLAIMS:
                problems.append(f"{cid} must be status 'pending', found '{claim['status']}'")
            continue
        if claim.get("metrics"):
            problems.append(f"{cid} is pending but carries {len(claim['metrics'])} metric entries")
        if claim["scope"] != "pending":
            problems.append(f"{cid} is pending but scope is '{claim['scope']}'")
    rep.check("pending claims carry no metrics", not problems, "; ".join(problems))

    ledger_problems: list[str] = []
    for i, row in enumerate(rows, start=2):
        if str(row.get("value", "")).strip().upper() != "PENDING":
            continue
        for field in ("uncertainty_ci", "unit", "source_field"):
            if (row.get(field) or "").strip():
                ledger_problems.append(f"line {i}: pending row has non-empty {field}")
    rep.check("pending ledger rows carry no values", not ledger_problems, "; ".join(ledger_problems))

    # A pending cell must not share a TABLE ROW with a numeric result cell.
    # Prose around a table (captions, notes) may legitimately mention both.
    tex_problems: list[str] = []
    prose_skipped: list[str] = []
    for path in tex_files():
        if path.name in PROSE_TABLES:
            prose_skipped.append(path.name)
            continue
        for body in tabular_bodies(strip_tex_comments(path.read_text())):
            for row in re.split(r"\\\\", body):
                if "\\textit{pending}" not in row:
                    continue
                for cell in row.split("&"):
                    if "\\textit{pending}" in cell:
                        continue
                    if NUMBER_RE.search(strip_exempt_numbers(cell)):
                        tex_problems.append(
                            f"{path.name}: numeric cell {cell.strip()[:40]!r} in a pending row"
                        )
    rep.check(
        "no LaTeX pending cell was filled in"
        + (f" (prose tables exempt: {', '.join(sorted(prose_skipped))})" if prose_skipped else ""),
        not tex_problems,
        "; ".join(tex_problems),
    )
    # A prose table that carries a pending cell must say the numbers beside it
    # are design facts, so the exemption above cannot hide a populated result.
    prose_problems: list[str] = []
    for name in prose_skipped:
        body = strip_tex_comments((TABLES / name).read_text())
        if "\\textit{pending}" in body and "design fact" not in body.lower():
            prose_problems.append(f"{name}: pending row without a 'design facts' qualifier")
    rep.check("prose tables label pending-row numbers as design facts", not prose_problems, "; ".join(prose_problems))

    # Condition C and tau3 smoke fit must be absent from any results table with numbers.
    banned = re.compile(r"(condition[ _-]?c|smoke[ _-]?fit)[^\n]{0,120}?\d\.\d{3}", re.I)
    leak: list[str] = []
    for path in tex_files():
        body = strip_tex_comments(path.read_text())
        for m in banned.finditer(body):
            snippet = m.group(0)
            # Design-time scales quoted in prose are permitted only in comments,
            # which strip_tex_comments has already removed.
            leak.append(f"{path.name}: {snippet[:70]!r}")
    rep.check("no numeric result attributed to Condition C or the smoke fit", not leak, "; ".join(leak))


def check_taskbench_scope(rep: Report, manifest: dict) -> None:
    rep.section("5. No TaskBench claim is labelled full HPOP")
    problems: list[str] = []
    for claim in manifest["claims"]:
        blob = json.dumps(claim).lower()
        if "taskbench" not in blob:
            continue
        if claim["scope"] == "full HPOP":
            problems.append(f"{claim['claim_id']} mentions TaskBench with scope 'full HPOP'")
    rep.check("manifest scopes", not problems, "; ".join(problems))

    tex_problems: list[str] = []
    forbidden = re.compile(
        r"(full[- ]hpop|recurrent hpop|hierarchical segmentation|real (agent )?trajector)",
        re.I,
    )
    tb = TABLES / "table_taskbench_main.tex"
    for path in [DRAFTS / "results_taskbench.tex", tb, TABLES / "table_taskbench_appendix.tex"]:
        if not path.exists():
            continue
        body = strip_tex_comments(path.read_text())
        for m in forbidden.finditer(body):
            # Legitimate uses are negations: "not a full-HPOP experiment",
            # "not observed agent trajectories", "neither ... nor full HPOP".
            window = body[max(0, m.start() - 160) : m.start()].lower()
            window = window.replace("\n", " ")
            if re.search(
                r"\b(not|never|no|none|nothing|non-|rather than|neither|nor|only|beyond)\b[^.]*$",
                window,
            ):
                continue
            tex_problems.append(f"{path.name}: unqualified {m.group(0)!r}")
    rep.check("TaskBench drafts qualify every full-HPOP phrase", not tex_problems, "; ".join(tex_problems))


def check_no_tau3_test_reads(rep: Report) -> None:
    rep.section("6. No sealed / live artifact is referenced as a source")
    problems: list[str] = []
    targets = list(EVIDENCE.glob("*")) + tex_files()
    for path in targets:
        if path.is_dir():
            continue
        text = path.read_text(errors="replace")
        for forbidden in FORBIDDEN_READS:
            for m in re.finditer(re.escape(forbidden), text):
                # A prohibition may sit on either side of the mention, e.g.
                # "X was never opened" or "never opened: X".
                window = (
                    text[max(0, m.start() - 240) : m.start()]
                    + " "
                    + text[m.end() : m.end() + 240]
                ).lower()
                if re.search(
                    r"(not\b|never|forbidden|sealed|untracked|do not|must not|was not|"
                    r"deliberately|live run|running|pending|currently)",
                    window,
                ):
                    continue
                problems.append(f"{path.name}: unqualified reference to {forbidden}")
    rep.check("sealed and live artifacts appear only in prohibitions", not problems, "; ".join(problems))

    # And prove it directly: no ledger row may source from them.
    ledger_hits = []
    if LEDGER_CSV.exists():
        for i, row in enumerate(csv.DictReader(LEDGER_CSV.open()), start=2):
            src = row.get("source_artifact") or ""
            if any(f in src for f in FORBIDDEN_READS):
                ledger_hits.append(f"line {i}: {src}")
    rep.check("no ledger row sources a sealed or live artifact", not ledger_hits, "; ".join(ledger_hits))


def check_latex(rep: Report, run_latex: bool) -> None:
    rep.section("7. LaTeX fragments compile in a minimal wrapper")
    if not run_latex:
        rep.skip("latex compile", "disabled with --no-latex")
        return
    engine = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("lualatex")
    if engine is None:
        rep.skip(
            "latex compile",
            "no LaTeX engine on PATH (install TeX Live/MacTeX, or rerun with --no-latex)",
        )
        return

    # Optional packages are loaded only if present, so a minimal TeX install
    # still exercises the fragments' own syntax rather than failing on a
    # missing style file.
    preamble = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,tabularx,array,multirow}
\makeatletter
\IfFileExists{pdflscape.sty}{\usepackage{pdflscape}}%
  {\newenvironment{landscape}{}{}}
\IfFileExists{graphicx.sty}{\usepackage{graphicx}}{}
\IfFileExists{xcolor.sty}{\usepackage{xcolor}}{}
\makeatother
\begin{document}
"""
    # Fragments cross-reference labels defined in sibling fragments and in the
    # (absent) main paper. This check is about SYNTAX, so \ref is made inert
    # here; dangling references are caught statically by check 9 instead.
    stubs = r"""
\makeatletter
\renewcommand{\ref}[1]{0}
\renewcommand{\pageref}[1]{0}
\makeatother
"""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for path in tex_files():
            wrapper = tmpdir / f"wrap_{path.stem}.tex"
            wrapper.write_text(preamble + stubs + f"\\input{{{path}}}\n" + "\\end{document}\n")
            res = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(tmpdir), str(wrapper)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                tail = [l for l in res.stdout.splitlines() if l.startswith("!")][:3]
                failures.append(f"{path.name}: {' | '.join(tail) or 'nonzero exit'}")
        rep.check(f"all {len(tex_files())} fragments compile", not failures, "; ".join(failures))


def check_rounding(rep: Report) -> None:
    """Reported metric values in table bodies must use exactly three decimals.

    Exempt, because they are not metric values: scientific-notation mantissas,
    percentages, fixed model settings of the form ``symbol = value``, z-scores
    (conventionally two decimals), digit-grouped counts, and count ratios.
    The exemption count is printed so the check cannot quietly excuse
    everything.
    """
    rep.section("8. Table rounding is consistent")
    problems: list[str] = []
    exempted = 0
    checked = 0
    skipped: list[str] = []
    for path in sorted(TABLES.glob("*.tex")):
        if path.name not in NUMERIC_RESULT_TABLES:
            skipped.append(path.name)
            continue
        body = strip_tex_comments(path.read_text())
        for cell_text in tabular_bodies(body):
            raw = len(re.findall(r"(?<![0-9])\d+\.\d+(?![0-9])", cell_text))
            kept = strip_exempt_numbers(cell_text)
            literals = re.findall(r"(?<![0-9])(\d+)\.(\d+)(?![0-9])", kept)
            exempted += raw - len(literals)
            checked += len(literals)
            bad = sorted({f"{a}.{b}" for a, b in literals if len(b) != 3})
            if bad:
                problems.append(f"{path.name}: {', '.join(bad[:8])}")
    rep.check(
        f"every reported table value uses three decimals "
        f"({checked} checked across {len(NUMERIC_RESULT_TABLES)} result tables, "
        f"{exempted} non-metric literals exempt"
        + (f"; prose tables not rounded-checked: {', '.join(sorted(skipped))}" if skipped else "")
        + ")",
        not problems,
        "; ".join(problems),
    )


def check_citations(rep: Report) -> None:
    rep.section("9. No undefined citation placeholder")
    problems: list[str] = []
    placeholder = re.compile(r"\\cite[tp]?\*?\{([^}]*)\}")
    todo = re.compile(r"\\(citation|CITE|TODOcite)\b|\[\s*(CITATION|CITE|REF)\s*\]|\?\?\?", re.I)
    for path in tex_files():
        body = strip_tex_comments(path.read_text())
        for m in placeholder.finditer(body):
            keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
            if not keys:
                problems.append(f"{path.name}: empty \\cite{{}}")
            for k in keys:
                if re.fullmatch(r"(TODO|XXX|FIXME|REF|CITE|\?+)", k, re.I):
                    problems.append(f"{path.name}: placeholder citation key {k!r}")
        for m in todo.finditer(body):
            problems.append(f"{path.name}: citation placeholder {m.group(0)!r}")
    rep.check("no placeholder citation keys", not problems, "; ".join(problems))

    # Every \ref target should either be defined in the pack or be a known stub.
    known_labels = set()
    for path in tex_files():
        known_labels.update(re.findall(r"\\label\{([^}]*)\}", strip_tex_comments(path.read_text())))
    external = {"sec:inference", "app:generator", "app:A", "app:B", "app:C", "app:D"}
    dangling: list[str] = []
    for path in tex_files():
        body = strip_tex_comments(path.read_text())
        for ref in re.findall(r"\\ref\{([^}]*)\}", body):
            if ref not in known_labels and ref not in external:
                dangling.append(f"{path.name}: \\ref{{{ref}}}")
    rep.check(
        f"cross-references resolve within the pack or a declared stub ({len(known_labels)} labels defined)",
        not dangling,
        "; ".join(dangling),
    )


def check_figure_manifest(rep: Report) -> None:
    rep.section("10. Figure manifest paths exist (extra)")
    if not FIGURE_MANIFEST.exists():
        rep.check("figure manifest present", False, "missing taskbench_figure_manifest.json")
        return
    data = json.loads(FIGURE_MANIFEST.read_text())
    problems: list[str] = []
    for fig in data["figures"]:
        for key in ("source_data_path", "rendered_figure_path"):
            if resolve_artifact(fig[key]) is None:
                problems.append(f"figure {fig['figure_id']}: {fig[key]} not found")
        if not fig.get("deterministic_selection_rule"):
            problems.append(f"figure {fig['figure_id']}: no selection rule recorded")
        if not fig.get("caption_draft"):
            problems.append(f"figure {fig['figure_id']}: no caption draft")
    rep.check(f"all {len(data['figures'])} figures resolve with rules and captions", not problems, "; ".join(problems))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-latex", action="store_true", help="skip the LaTeX compile check")
    ap.add_argument("-v", "--verbose", action="store_true", help="print passing checks too")
    args = ap.parse_args()

    for required in (MANIFEST, LEDGER_CSV):
        if not required.exists():
            print(f"FATAL: {required} not found. Run this from the repository root.")
            return 1

    manifest = json.loads(MANIFEST.read_text())
    rows = list(csv.DictReader(LEDGER_CSV.open()))

    print(f"Validating evidence pack under {EVIDENCE.parent}")
    print(f"  manifest claims : {len(manifest['claims'])}")
    print(f"  ledger rows     : {len(rows)}")
    print(f"  latex fragments : {len(tex_files())}")

    rep = Report(verbose=args.verbose)
    check_ledger_sources(rep, rows)
    check_commits(rep, rows, manifest)
    check_hashes(rep, manifest)
    check_pending_unpopulated(rep, rows, manifest)
    check_taskbench_scope(rep, manifest)
    check_no_tau3_test_reads(rep)
    check_latex(rep, run_latex=not args.no_latex)
    check_rounding(rep)
    check_citations(rep)
    check_figure_manifest(rep)

    print("\n" + "=" * 60)
    print(f"{rep.checks} checks run, {len(rep.failures)} failed, {len(rep.skips)} skipped")
    if rep.skips:
        print("\nSkipped:")
        for s in rep.skips:
            print(f"  - {s}")
    if rep.failures:
        print("\nFailures:")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print("\nEvidence pack is internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
