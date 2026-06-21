# Annotation Guidelines (the rulebook)

The **human decision rules** for annotating WebLINX traces into the four-level structure. Authored
by the linguistics PhD. This document is both the human annotation standard and the spec that gets
operationalized into the LLM annotator (`rules/annotation_prompt.md`).

> **Ownership:** linguistics PhD. Sections marked _TBD_ await that content.

The annotator (human or LLM) makes four sub-decisions per trace. Each needs an explicit decision
procedure below.

---

## Decision 1 — Segment raw events → canonical action tokens

A canonical action token is the minimal meaningful action unit (an "intent act"), spanning 1–3 raw
events. **Boundary rule:** a new token begins when the agent's intent shifts to a different atomic
goal.

_TBD — operational rules: how to recognize an intent shift; how to merge 1–3 raw events; edge
cases (retries, tool errors, no-ops, system events)._

---

## Decision 2 — Label each token's canonical action

_TBD — labeling scheme and the canonical action vocabulary (see `docs/taxonomy.md`)._

---

## Decision 3 — Bracket tokens into skill instances + assign phase and skill type

A skill instance is a complete invocation of a reusable skill pattern; it ends when the procedure
completes (success or failure). Each instance gets a **phase** tag and a **skill type**.

_TBD — rules for: where a skill instance starts/ends; how to assign the phase (cognitive category);
how to assign / name the skill type; handling recurrence of the same skill type in one trace._

---

## Decision 4 — Emit precedence (partial-order) edges

Two partial orders:

- **Local** — over tokens within a skill instance: which steps *must* precede which (the rest are
  free / incomparable).
- **Workflow** — over skill instances within the trace: which skills *must* precede which.

_TBD — rules for deciding a "must-precede" edge vs. incidental ordering; ensuring the result is a
well-formed DAG (acyclic, boundaries aligned); how phase precedence constraints (`rules/phases.yaml`)
bound the workflow edges._

---

## Output

The annotator emits one schema-conformant record per trace (see `rules/` for the schema once
defined): raw events, tokens (with labels), skill instances (with phase + type + member tokens),
and the two edge lists.

## Validation

A human-gold subset (`data/gold/`) is annotated by hand to the same rules. LLM annotations are
scored against it on: token boundaries, phase labels, skill-type labels, and edge agreement.
