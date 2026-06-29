# Layer A — The 9-Phase Cognitive-Control Taxonomy

The annotation framework from the linguistics coauthor (PDAF v3). It replaces the ad-hoc 4-phase
grouping (`orient / reproduce_diagnose / modify / verify_finalize`) with **nine cognitive-control
phases** grounded in **Miller & Cohen (2001)**, organised across **four abstraction levels**. In v3 the
the phase label is an **optional, coarse procedural descriptor** for interpretability and diagnostics —
**not** a fixed skill type. HPOP does not take phases as skill labels: it receives the CPA occurrence
sequences and learns a reusable library of **local partial-order skills**, and a learned skill may
contain CPAs from one phase or several.

Machine-readable form: [`rules/phases.yaml`](../rules/phases.yaml).

---

## The three layers

| Layer | Name | Does |
|---|---|---|
| **A** | Neuro-cognitive abstraction | The **9 phases** (this doc) = an optional coarse control-state descriptor for interpretability — *not* the model's skill types. |
| **B** | Linguistic formalization | **B-1a**: normalize each event → *action token*. **B-1b+**: segment into *procedural phrases* by boundary cut-rules; then tag each phrase's **phase**. |
| **C** | Annotation pipeline | The executable steps: normalize → CPA-abstract → classify skill → label dependencies → score → review. |

> **phase ≠ phrase.** A **phase** is the cognitive *control state* (PLAN, VERIFY, …). A **procedural
> phrase** is a *local functional chunk* of consecutive/near-consecutive actions realising one proximate
> goal. The same phrase type may recur across phases; **a phase boundary always constrains a phrase span
> (phase precedes phrase).**

---

## The nine phases (4 abstraction levels)

| L | Level | Phase | Cognitive state | Reference |
|---|---|---|---|---|
| **L1** | goal-setting | **PLAN** | Goal decomposition — PFC sets the goal hierarchy | Miller & Cohen (2001); Grosz & Sidner (1986) |
| **L2** | execution | **RETRIEVE** | Directed subgoal pursuit — externally-directed acquisition | Botvinick et al. (2009) |
| **L2** | execution | **INSPECT** | Metacognitive monitoring — passive scanning, *no proposition tested* | Nelson & Narens (1990) |
| **L2** | execution | **EXTRACT** | Focused subgoal pursuit — central executive locks on target | Baddeley (2000) |
| **L2** | execution | **VERIFY** | Metacognitive control — *explicit proposition; output is a truth value* | Nelson & Narens (1990); Wason (1960) |
| **L2** | execution | **WRITE** | Generative production — externalization mode | Flower & Hayes (1981) |
| **L3** | integration | **SYNTHESIZE** | Multi-source integration — assembles a representation no single unit possessed | Hutchins (1995) |
| **L4** | interrupt | **REPAIR** | Error monitoring — SAS override after a failure signal | Norman & Shallice (1986) |
| **L4** | interrupt | **HANDOFF** | Distributed cognitive load transfer (agent identity change) | Hutchins (1995); Monsell (2003) |

**Phase grammar** (recurrence, not a fixed line): PLAN opens an episode; the five L2 states
(RETRIEVE, INSPECT, EXTRACT, VERIFY, WRITE) freely interleave and recur; SYNTHESIZE follows several L2
acquisitions; REPAIR and HANDOFF are **interrupts** that may fire at any point and then return. HPOP
infers the global poset over phases itself — we supply only the per-instance phase label.

---

## Layer B-1a — Physical normalization (action tokens)

Normalize every raw event into an **action token** with four fields — **no A-layer or B-layer rule is
applied here; this is pure data cleaning + format unification** (ISO 24612:2012):

| Field | Meaning | Coding-domain source |
|---|---|---|
| `action_type` | the act (click, scroll, type → **view, edit, run, search, …**) | tool / command verb |
| `tool_family` | tool class (browser, keyboard → **editor, shell, vcs, test_runner, …**) | tool name |
| `agent_id` | which agent executed it | role / agent field (single-agent ⇒ constant) |
| `artifact_id` | the object operated on (which page/file) | normalized file / symbol (our `ground.py` role:id) |

Then **collapse identical repeated actions within a 5-second window** to remove logging noise. (Our
SWE-rebench traces have ordering but not always wall-clock timestamps — the dedup becomes "merge
adjacent identical `(action_type, artifact_id)` tokens"; flagged for the coauthor.)

## Layer B — Procedural phrase boundaries (cut rules)

A **procedural phrase** is bounded by functional cohesion. Cut when **any** fires (priority order):

1. **Proximate-goal change** → must cut *(strongest signal)*.
2. **Artifact-identity change** → must cut *(most reliable in agent traces — drives our `ground.py` chains)*.
3. **Phase boundary** → must cut *(phase precedes phrase)*.
4. **Tool-family change combined with a goal shift** → cut.

After boundaries are fixed, consult the 9 phase definitions above and **tag each phrase's phase** by its
cognitive control state.

---

## Bridge: our 29 coding CPAs → the 9 phases (proposed)

This re-buckets the current dictionary under the 9-phase taxonomy (was 4 phases). Ambiguous cells are
flagged for coauthor sign-off.

| Phase | Coding CPAs |
|---|---|
| **PLAN** | *(none yet — planning is latent/`think`, excluded; future FORMULATE_GOAL)* |
| **RETRIEVE** | EXPLORE_REPOSITORY, LOCATE_CODE |
| **INSPECT** | READ_SOURCE, INSPECT_CHANGES, INSPECT_HISTORY, INSPECT_ENVIRONMENT, READ_DOCUMENTATION |
| **EXTRACT** | DIAGNOSE_FAILURE *(⚠ could be INSPECT — coauthor to confirm)* |
| **VERIFY** | RUN_TEST_SUITE, VERIFY_FIX, REPRODUCE_ISSUE, CHECK_TYPES, RUN_LINTER, COMPARE_OUTPUT, MEASURE_PERFORMANCE |
| **WRITE** | EDIT_SOURCE, WRITE_TEST, WRITE_REPRODUCTION_SCRIPT, ADD_DEBUG_INSTRUMENTATION, FORMAT_CODE, REFACTOR_CODE, CONFIGURE_ENVIRONMENT, MANAGE_FILESYSTEM, INSTALL_DEPENDENCY, BUILD_PROJECT, APPLY_PATCH, CLEANUP_ARTIFACTS, SUBMIT_SOLUTION |
| **SYNTHESIZE** | *(⚠ rare/undetectable by rules — pending LLM pass)* |
| **REPAIR** | REVERT_CHANGE, INTERACTIVE_DEBUG *(control state = REPAIR even when the realizing CPA is a WRITE, e.g. ADD_DEBUG_INSTRUMENTATION after a failure)* |
| **HANDOFF** | *(none — single-agent traces; fires only on `agent_id` change)* |

### Three open questions for the coauthor
1. **EXTRACT in coding** — is DIAGNOSE_FAILURE an EXTRACT (locks onto the failing value) or an INSPECT?
2. **SYNTHESIZE** — does it occur in coding (cross-file integrating edits), or is it web/research-specific?
3. **REPAIR vs WRITE** — REPAIR is a *control state*; its realizing CPA is often a WRITE. Do we tag the
   phase (REPAIR) independently of the CPA (WRITE), i.e. phase and CPA on separate axes? (v3 says yes.)

---

*Next step (per coauthor): rewrite the CPA annotation rule on top of this — Layer B action tokens →
phrase segmentation → phase tagging → CPA abstraction — as the new annotation method.*
