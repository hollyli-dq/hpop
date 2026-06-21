# HPOP — Design

**HPOP**: turning raw agent interaction traces into **reusable skill programs represented as
partial orders**, then using those partial orders for fast inference.

This document is the source of truth for the data model, the taxonomy, the annotation pipeline,
and the core thesis. Code and rule files build on it.

---

## 1. The core thesis

**The recovered partial order *is* the product.**

When you extract the partial order from a trace, you have a *precompiled execution plan* — a
"recipe" for how the task gets done. You have already paid the expensive cost of *figuring out
how to do the task*. At inference time you are not re-deriving a plan token-by-token with an LLM;
you are **traversing a DAG**.

Speed comes from four concrete properties of the partial order:

1. **Parallelism** — incomparable nodes have no edge between them, so they can execute
   concurrently instead of being forced into one sequential chain.
2. **No re-planning** — the recipe is retrieved and reused across tasks that share a workflow /
   phase grammar, instead of being regenerated from scratch.
3. **Search pruning** — a free-form agent searches over *all* action sequences; a DAG restricts
   you to its **linear extensions** only, collapsing the combinatorial space to valid orderings.
4. **Cheap legality checks** — "is this next step allowed?" reduces to "are its predecessors
   done?" — a frontier check on the DAG, not an LLM call.

**Offline (expensive):** traces → annotation → partial orders → skill library.
**Online (cheap):** retrieve the matching skill program → execute by DAG traversal.

---

## 2. The abstraction stack

Levels, from atomic to whole, over WebLINX traces:

| Level | Unit | Definition | Span / boundary rule |
|------:|------|------------|----------------------|
| 0 | **Raw trace event** | Atomic log entry: a single tool call, tool result, assistant message, or system event. One line in the execution log. | 1 line |
| 1 | **Canonical action token** | Minimal meaningful action unit — a single purposeful step that cannot be decomposed further without losing semantic coherence (an "intent act"). | Spans 1–3 raw events. A **new token** begins when the agent's intent shifts to a different atomic goal. |
| 2 | **Skill instance** | A complete invocation of a reusable skill *pattern*. The same skill *type* may recur multiple times in a trace. Tagged with a **phase** (see §3). | Ends when the reusable procedure completes — success or failure. |
| 3 | **Workflow** | The entire trace, task receipt → final output delivery. The complete execution of a task. | Whole trace |

### Two nested partial orders

- **Local partial order** — over **canonical action tokens** *within* a skill instance: which
  steps must precede which; the rest are free (incomparable).
- **Workflow partial order** — over **skill instances** *within* a workflow: which skills must
  precede which; the rest are free.

A workflow is therefore a **partial order over partial orders**: a PO over skills, each of which
is itself a PO over tokens.

---

## 3. Phase — the cognitive-science-grounded category

**A phase is a category *tag* on a skill instance, not a level in the stack.** It is the
"part-of-speech" of a skill, drawn from the cognitive-science literature on how agents solve
complex tasks (orient, explore, plan, act, monitor/verify, deliver, …). The phase classifies and
guides a skill instance; it does not compose it.

The linguistic analogy that drives the whole design:

- **skill instance : phase  ::  word : part-of-speech**
- **workflow : phase grammar  ::  sentence : syntax**

**Constructing a workflow from phases.** Project the workflow's partial-order-of-skills onto the
phase tags and you get a **phase-level partial order** — the cognitive shape of how the task is
solved (e.g. *Orient → Explore → {Act ∥ Verify}\* → Deliver*). This projection is the reusable
abstraction:

- Different concrete skills can realize the **same** phase, so different workflows share a **phase
  grammar** even when their surface skills differ — this is the source of cross-task generalization.
- The phase categories supply the **rule system's constraints**: which phases may precede which
  (e.g. you cannot *Verify* before you *Act*).
- The **skill library is indexed by phase**, so "what skills can fill the Explore slot" is a
  first-class query.

---

## 4. The annotation pipeline: human rules → LLM annotator → structured skill programs

- A **linguistics PhD authors the human decision rules** (`docs/annotation-guidelines.md`): how to
  cut raw events into tokens, group tokens into skill instances, assign phase tags and skill
  types, and lay down the precedence edges.
- An **LLM applies those rules at scale** to annotate WebLINX traces, emitting the four-level
  structure as labeled, schema-conformant output.

The rulebook does double duty: it is both the human-readable annotation standard *and* the spec
operationalized into the LLM annotator (system prompt + decision procedure + output schema).

The LLM performs four sub-decisions per trace:
1. segment raw events → canonical action tokens,
2. label each token's canonical action,
3. bracket tokens into skill instances + assign a **phase** and **skill type**,
4. emit the within-skill (local) and across-skill (workflow) precedence edges.

**Validation.** A small **human-gold** subset (`data/gold/`) measures how faithfully the LLM
follows the rules — agreement on boundaries, phase/skill labels, and edges — before trusting it at
scale.

---

## 5. End-to-end pipeline

```
WebLINX traces
  → ingest            (data/raw → data/interim: parse execution log into raw events)
  → annotate          (rules + LLM → data/annotated: tokens, skills, phases, edges)
  → extract           (annotations → two nested partial orders / DAGs)
  → library           (skill_library/: store, index by phase, retrieve reusable skill programs)
  → inference         (fast DAG traversal: parallel, pruned, reusable execution on new tasks)
  +  eval             (validate annotations vs data/gold; measure extraction + inference)
```

---

## 6. Repository layout

| Path | Holds |
|------|-------|
| `docs/` | This design doc, the taxonomy, and the human annotation guidelines (rulebook). |
| `rules/` | Machine-actionable rule artifacts the LLM annotator consumes (phase/skill inventories, annotation prompt). |
| `data/raw/` | Raw WebLINX traces. |
| `data/interim/` | Parsed/canonicalized events and tokens. |
| `data/annotated/` | LLM-annotated, schema-conformant structured output. |
| `data/gold/` | Human-annotated validation subset. |
| `skill_library/` | The reusable skill programs (DAG artifacts), indexed by phase. |
| `src/hpop/ingest/` | WebLINX loading → raw events. |
| `src/hpop/annotate/` | LLM annotation pipeline. |
| `src/hpop/extract/` | Annotations → partial orders / DAGs. |
| `src/hpop/library/` | Skill library: store, index, retrieve. |
| `src/hpop/inference/` | Fast DAG-based inference / execution. |
| `src/hpop/eval/` | Validation vs gold, metrics. |
| `archive/` | The previous unsupervised-structure-learning prototype (paper + poset/linext code). Kept for reference and possible reuse (e.g. the C++ linear-extension counter for DAG inference). |

---

## 7. Open questions (to resolve as the design firms up)

- **Skill-type inventory** — closed taxonomy authored by the PhD, induced from data, or hybrid?
- **Canonical action token vocabulary** — fixed set or open?
- **Annotation output schema** — exact JSON shape for tokens/skills/phases/edges (so partial
  orders are well-formed: acyclic, boundaries aligned).
- **LLM choice and prompting** — which model, single-pass vs. staged decisions, self-consistency.
- **Retrieval** — how a new task is matched to a skill program (phase grammar match? embedding?).
