# HPOP — Hierarchical Partial-Order Skill Programs

Turn raw **software-agent trajectories** into reusable **skill programs represented as partial
orders**, then use those partial orders for fast, parallel, reusable inference.

Pipeline: **raw trajectory → events → canonical procedural actions (CPAs) → skills → local posets →
global poset**. CPAs are *induced from data* (LLM-assisted, not a predefined ontology); skills and
their partial orders are *learned* by a hierarchical Bayesian model (BPOP frontier-softmax + a
Dirichlet-process skill library with a new-skill penalty).

Primary benchmark: [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories).
Open [`docs/index.html`](docs/index.html) (the dashboard) and read [`docs/DESIGN.md`](docs/DESIGN.md) /
[`docs/MODEL.md`](docs/MODEL.md) / [`docs/cpa_induction.md`](docs/cpa_induction.md).

## Setup
```bash
git clone https://github.com/hollyli-dq/hpop.git && cd hpop
python3.13 -m venv .venv
.venv/bin/python -m pip install anthropic matplotlib   # anthropic only needed for the real LLM annotation
# every command below assumes:  PYTHONPATH=src .venv/bin/python ...
```

## CLI — end-to-end (no API key needed)
```bash
# 1. INGEST: sample SWE-rebench trajectories -> normalized events
PYTHONPATH=src .venv/bin/python -m hpop.ingest.swe_rebench \
    --resolved 50 --unresolved 50 --output data/interim/swe_rebench/pilot100.jsonl

# 2a. CPA ANNOTATION (silver, rule-based, no key) -> occurrence-level CPAs
PYTHONPATH=src .venv/bin/python -m hpop.annotate.rule_apply \
    --input data/interim/swe_rebench/pilot100.jsonl \
    --library rules/cpa_library_seed.json \
    --output data/annotated/swe_rebench/cpa_rule

# 3. EXTRACT: CPA annotations -> modelling-ready CPA sequences + corpus vocab
PYTHONPATH=src .venv/bin/python -m hpop.extract.sequences \
    --opencode data/annotated/swe_rebench/cpa_rule \
    --out data/modelling/swe_rebench/pilot100

# 4. SKILL ANNOTATION: segment CPA sequences into skill instances (weak-supervision prior)
PYTHONPATH=src .venv/bin/python -m hpop.annotate.skill_segment \
    --sequences data/modelling/swe_rebench/pilot100.sequences.jsonl \
    --library rules/skill_library_seed.json \
    --out data/modelling/swe_rebench/pilot100

# 5. VISUALIZE: render the CPA view from annotations
PYTHONPATH=src .venv/bin/python scripts/render_from_opencode.py \
    --opencode data/annotated/swe_rebench/cpa_rule \
    --traces data/interim/swe_rebench/pilot100.jsonl \
    --library rules/cpa_library_seed.json --out docs/cpa_view.html
```
Modelling input lands in `data/modelling/swe_rebench/`: `pilot100.sequences.jsonl` (CPA sequences =
local level) and `pilot100.skills.jsonl` (global skill sequence + per-skill CPA subsequences).

## CLI — real LLM annotation (needs `ANTHROPIC_API_KEY`)
Replaces step 2a with semantic open-coding (model `claude-opus-4-8`). INDUCE proposes CPAs; APPLY
matches a frozen library.
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# INDUCE with two independent configs, then consolidate -> CPA library v0.1
PYTHONPATH=src .venv/bin/python -m hpop.annotate.opencode --input data/interim/swe_rebench/pilot100.jsonl \
    --output data/annotated/swe_rebench/cpa_A --mode INDUCE --annotator cfgA
PYTHONPATH=src .venv/bin/python -m hpop.annotate.opencode --input data/interim/swe_rebench/pilot100.jsonl \
    --output data/annotated/swe_rebench/cpa_B --mode INDUCE --annotator cfgB
PYTHONPATH=src .venv/bin/python -m hpop.annotate.consolidate \
    --inputs data/annotated/swe_rebench/cpa_A.cpa_instances.jsonl data/annotated/swe_rebench/cpa_B.cpa_instances.jsonl \
    --out-library rules/cpa_library_v0.1.json --out-review data/annotated/swe_rebench/review.jsonl --m 3 --r 2 --tau 0.5
# APPLY the frozen library (consistent labels); add --dry-run to preview the prompt without an API call
PYTHONPATH=src .venv/bin/python -m hpop.annotate.opencode --input data/interim/swe_rebench/pilot100.jsonl \
    --output data/annotated/swe_rebench/cpa_apply --mode APPLY --library rules/cpa_library_v0.1.json --library-version v0.1
```

## Model primitives + tests
```bash
PYTHONPATH=src .venv/bin/python scripts/inference_demo.py        # frontier-softmax + new-skill penalty demo
PYTHONPATH=src .venv/bin/python -m unittest tests.test_inference -v   # 13 sanity tests
```
`src/hpop/inference/` = the BPOP frontier-softmax likelihood (`likelihood.py`), latent-U posets
(`poset.py`), and the CRP/Ewens new-skill penalty (`library.py`). The hierarchical MCMC learning loop
is the model author's part.

## Layout
| Path | Holds |
|------|-------|
| `src/hpop/ingest/` | `swe_rebench.py` — OpenHands trajectories → events |
| `src/hpop/annotate/` | `opencode` (LLM CPA), `rule_apply` (silver CPA), `skill_segment`, `consolidate` |
| `src/hpop/extract/` | annotations → modelling-ready CPA sequences |
| `src/hpop/inference/` | BPOP likelihood, posets, CRP penalty (model primitives) |
| `rules/` | `cpa_library_seed.json`, `skill_library_seed.json` (+ `cpas.yaml`, `skill_library.yaml`) |
| `data/` | `interim/` events · `annotated/` CPA labels · `modelling/` CPA + skill sequences |
| `docs/` | dashboard (`index.html`), design/model/methodology docs, visualizations |
| `scripts/` | `render_from_opencode.py`, `inference_demo.py`, `plot_three_levels.py` |
| `notebooks/` | `swe_rebench_analysis.ipynb` — the 6-stage walkthrough |

> Note: `rule_apply` / `skill_segment` produce **silver / heuristic** labels (LLM-designed rules, no
> API) — a starting prior. Swap in the real LLM annotation (`opencode`) once a key is available.
