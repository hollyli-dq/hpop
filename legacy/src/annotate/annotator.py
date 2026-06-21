"""PDAF Steps 1b + 2 + 3 as one rule-guided LLM call (Report 3 §2.3).

`build_system_prompt` assembles the PDAF rulebook (phases, CPAs, dependency labels, skill
library — read verbatim from rules/) into a system prompt. `annotate_trace` sends one
normalized trace and returns the validated PDAF object (schema in schema.py). The model does:
Step 1b (group action tokens -> CPA instances), Step 2 (classify the skill), Step 3 (typed
dependency edges), and emits per-item confidences (Step 4 input). `run.py` does the
id-assignment, output split, and distribution monitoring around this.

Model: claude-opus-4-8, adaptive thinking, effort=high, structured output, prompt-cached system.
"""
from __future__ import annotations

import json
import os

from hpop.annotate.schema import (
    PDAF_ANNOTATION_SCHEMA, PHASES, CPAS, DEPENDENCY_LABELS,
)

MODEL = "claude-opus-4-8"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RULE_FILES = ["phases.yaml", "cpas.yaml", "dependencies.yaml", "skill_library.yaml"]

_INSTRUCTIONS = """\
You are a PDAF v3.0 annotator. You convert a single normalized agent interaction trace (a list
of action tokens) into a structured TWO-LEVEL procedural-skill annotation for the HPOP model.

The trace is one WebLINX demonstration: a `navigator` agent carrying out a task for an
`instructor`, interleaved with dialogue (`say`) turns. A single demonstration usually contains
MORE THAN ONE skill (e.g. search_and_summarize, then verify_fact). Work strictly from the rulebook.

Make exactly these decisions, in order:

STEP 1b — CPA ABSTRACTION. Group the action tokens into one ordered list of CPA instances (across
the whole trace). Each CPA instance is one of the 13 Canonical Procedural Actions and carries its
owning phase (use the CPA->phase mapping in the rulebook; the two must agree). Give each a 0-based
`index`, the `event_indices` (the `i` of the action tokens it covers), `turn_start`/`turn_end`
(min/max of those indices), a short `artifact_id`, a `confidence` in [0,1], and a one-line
`rationale` naming the trigger/signal that fired. Use the physical triggers + signals as
priority-ordered contextual rules with window-based lookahead. Identical physical sequences can be
different CPAs depending on context (click+type+click = SEARCH_SOURCE when retrieving vs FILL_FIELD
when completing a form) — use intent, not surface form. If genuinely ambiguous, set confidence < 0.5
(ABSTAIN) rather than forcing a label.

STEP 2 — SEGMENTATION + SKILL CLASSIFICATION. Partition the CPA sequence into contiguous SKILL
INSTANCES (each is a complete invocation of one reusable skill; the same skill TYPE may recur, so
give it the same `skill_name` again as a new instance). Emit `skill_instances` (each with a 0-based
`index`, a `skill_name`, `skill_is_new`, `skill_confidence`) and set every CPA instance's
`skill_index` to its owning skill. Match each skill against SKILL_LIBRARY by phase-sequence overlap
(Jaccard over phase sets + domain fit); reuse the exact library `skill_name` and set
`skill_is_new=false`, otherwise emit a new lowercase snake_case name and `skill_is_new=true`.

STEP 3a — LOCAL DEPENDENCIES. Emit `local_edges` between CPA-instance pairs that lie in the SAME
skill (reference them by CPA `index`). For each ordered pair (A before B) apply the 8 dependency
labels in STRICT PRIORITY ORDER (rank 1 first); the FIRST matching rule wins. Include the WebLINX
STATE_CHANGE patches R1-R3. Use INCOMPARABLE for same-parent-goal pairs whose order is arbitrary
(this is what licenses parallelism). Use ADJACENT_ONLY only as a last resort (well under 15%).

STEP 3b — GLOBAL DEPENDENCIES. Emit `global_edges` between SKILL-INSTANCE pairs (reference them by
skill `index`) using the SAME 8 labels and priority order. This is the workflow-level partial order
over skills (e.g. a retrieval skill PRECONDITION/ELABORATES a verification skill; two independent
retrieval skills are INCOMPARABLE).

Every edge gets a `rationale`, a `confidence`, and `artifact_transferred` for DATA_FLOW / VERIFY_OF
(else ""). Output ONLY the structured object required by the response schema. No prose.
"""


def _read_rulebook():
    parts = []
    rules_dir = os.path.join(_REPO_ROOT, "rules")
    for fn in _RULE_FILES:
        p = os.path.join(rules_dir, fn)
        try:
            with open(p, encoding="utf-8") as f:
                parts.append("### rules/{}\n{}".format(fn, f.read().rstrip()))
        except FileNotFoundError:
            parts.append("### rules/{} (missing)".format(fn))
    return "\n\n".join(parts)


def build_system_prompt():
    """Full PDAF system prompt: instructions + verbatim rulebook + canonical enums."""
    enums = (
        "CANONICAL ENUMS (use these exact strings):\n"
        "  PHASES = {}\n  CPAS = {}\n  DEPENDENCY_LABELS (priority order) = {}\n".format(
            PHASES, CPAS, DEPENDENCY_LABELS
        )
    )
    return "{}\n\n=== PDAF RULEBOOK ===\n\n{}\n\n{}".format(_INSTRUCTIONS, _read_rulebook(), enums)


def build_user_message(trace):
    """Serialize one normalized trace's action tokens compactly for the model."""
    lines = [
        "trace_id: {}".format(trace.get("trace_id")),
        "instructor_goal: {}".format(trace.get("instructor_goal")),
        "num_action_tokens: {}".format(trace.get("num_action_tokens", len(trace.get("action_tokens", [])))),
        "",
        "ACTION TOKENS (i | action_type | tool_family | agent | artifact | note):",
    ]
    for t in trace.get("action_tokens", []):
        note = ""
        if t.get("action_type") == "say":
            note = "[{}] {}".format(t.get("agent_id"), (t.get("utterance") or "")[:80])
        elif t.get("collapsed"):
            note = "(x{} collapsed)".format(t["collapsed"])
        lines.append("{:>3} | {:<11} | {:<10} | {:<10} | {:<22} | {}".format(
            t["i"], t.get("action_type", ""), t.get("tool_family", ""),
            t.get("agent_id", ""), (t.get("artifact_id") or "")[:22], note))
    lines.append("\nAnnotate this trace per the PDAF steps and return the structured object.")
    return "\n".join(lines)


def annotate_trace(client, trace, system_prompt=None, max_tokens=16000):
    """Run one PDAF annotation call. Returns the validated dict (schema PDAF_ANNOTATION_SCHEMA)."""
    system_prompt = system_prompt or build_system_prompt()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": PDAF_ANNOTATION_SCHEMA},
        },
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_message(trace)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("annotation refused: {}".format(getattr(resp, "stop_details", None)))
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)
