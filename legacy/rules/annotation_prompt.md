# LLM Annotation Prompt

The operationalized version of `docs/annotation-guidelines.md` — the system prompt + decision
procedure the LLM annotator runs. Keep this in sync with the guidelines: the guidelines are the
human standard, this is its machine-executable form.

> **Status:** scaffold. Fill once the guidelines (Decisions 1–4) and the output schema are settled.

---

## System prompt (draft skeleton)

```
You annotate raw agent interaction traces (WebLINX) into a four-level structure.

LEVELS
  0 raw event           — one line of the execution log (tool call / result / message / system event)
  1 canonical action token — minimal intent act; spans 1–3 raw events
  2 skill instance      — a complete invocation of a reusable skill pattern; carries a PHASE tag and a SKILL TYPE
  3 workflow            — the whole trace

You make four decisions (see rules below):
  1. segment raw events into canonical action tokens   (boundary = intent shift to a new atomic goal)
  2. label each token's canonical action
  3. bracket tokens into skill instances; assign a phase (from PHASES) and a skill type
  4. emit precedence edges: local (tokens within a skill) and workflow (across skills)

CONSTRAINTS
  - Phase precedence must respect the phase grammar (PHASES.may_follow).
  - Both edge sets must be acyclic (valid DAGs) with boundaries aligned.
  - Output must conform exactly to the OUTPUT SCHEMA. Emit only the schema object.

PHASES
  {{ injected from rules/phases.yaml }}

SKILL TYPES
  {{ injected from rules/skills.yaml }}

RULES
  {{ Decisions 1–4, injected from docs/annotation-guidelines.md }}

OUTPUT SCHEMA
  {{ injected — see below }}
```

## Output schema (TBD)

JSON object per trace: `raw_events`, `tokens` (with labels + member event ids), `skill_instances`
(with phase, skill_type, member token ids), `local_edges`, `workflow_edges`. Exact shape to be
finalized alongside the extraction code (`src/hpop/extract/`).
