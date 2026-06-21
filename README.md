# HPOP

Turning raw agent interaction traces into **reusable skill programs represented as partial
orders**, then using those partial orders for **fast inference**.

The pipeline takes WebLINX agent traces, annotates them with a **linguistically-grounded,
cognitive-science-backed taxonomy** (via human decision rules applied at scale by an LLM), and
extracts two nested partial orders per task:

- a **local** partial order over canonical action tokens within each skill, and
- a **workflow** partial order over skill instances within the whole task.

Each recovered partial order is a DAG — a precompiled "recipe" for completing a task. Inference
then becomes **DAG traversal** (parallel, search-pruned, reusable) instead of from-scratch
generation by an LLM.

> Read [`docs/DESIGN.md`](docs/DESIGN.md) first — it is the source of truth for the data model,
> taxonomy, annotation pipeline, and the core thesis.

## Layout

| Path | Holds |
|------|-------|
| `docs/` | Design doc, taxonomy, and human annotation guidelines (the rulebook). |
| `rules/` | Machine-actionable rule artifacts the LLM annotator consumes. |
| `data/` | `raw/` WebLINX traces → `interim/` parsed events → `annotated/` LLM output → `gold/` human validation set. |
| `skill_library/` | Reusable skill programs (DAG artifacts), indexed by phase. |
| `src/hpop/` | Pipeline: `ingest`, `annotate`, `extract`, `library`, `inference`, `eval`. |
| `tests/` | Tests. |
| `archive/` | Previous unsupervised-structure-learning prototype (kept for reference/reuse). |

## Status

Project foundation. The data model and thesis are settled (see `docs/DESIGN.md`); the taxonomy,
annotation rules, and pipeline code are being built. The skill-type inventory and annotation
schema are open (see §7 of the design doc).
