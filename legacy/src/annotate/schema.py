"""PDAF v3.0 structured-output schema — TWO-LEVEL (skills + CPAs).

One Claude call per trace performs:
  Step 1b  CPA Abstraction      -> ordered CPA instances (canonical action tokens)
  Step 2   Segmentation+Skill   -> partition CPAs into skill INSTANCES, each labelled with a
                                   skill type (recurring vocabulary R = skill_name)
  Step 3a  Local dependencies   -> typed edges over CPA pairs WITHIN a skill
  Step 3b  Global dependencies  -> typed edges over SKILL-INSTANCE pairs (the workflow poset)

This matches the two BPOP/HPOP levels:
  LOCAL  poset nodes = CPA instances within a skill   (-> local_orders.jsonl)
  GLOBAL poset nodes = skill instances (types = R)     (-> global_orders.jsonl)
Phases are a cognitive TAG on each CPA, not a modeling layer; there is no phrase layer.

Flat design: every CPA carries a `skill_index`; edges are flat index-referencing lists. `run.py`
assigns canonical ids and splits into the output files. Schema obeys structured-output limits
(no recursion, no numeric/length bounds, additionalProperties:false everywhere).
"""

PHASES = [
    "PLAN", "RETRIEVE", "INSPECT", "EXTRACT", "VERIFY",
    "WRITE", "SYNTHESIZE", "REPAIR", "HANDOFF",
]
CPAS = [
    "FORMULATE_GOAL", "SEARCH_SOURCE", "OPEN_DOCUMENT", "NAVIGATE_PAGE",
    "READ_SOURCE", "SCAN_RESULTS", "EXTRACT_EVIDENCE", "FILL_FIELD",
    "VERIFY_CLAIM", "WRITE_RESPONSE", "SUBMIT_FORM", "SYNTHESIZE_FINDINGS",
    "REPAIR_SOURCE",
]
DEPENDENCY_LABELS = [  # priority order (first match wins); used at BOTH levels
    "REPAIR_OF", "VERIFY_OF", "DATA_FLOW", "STATE_CHANGE",
    "PRECONDITION", "ELABORATES", "INCOMPARABLE", "ADJACENT_ONLY",
]
CPA_TO_PHASE = {
    "FORMULATE_GOAL": "PLAN",
    "SEARCH_SOURCE": "RETRIEVE", "OPEN_DOCUMENT": "RETRIEVE", "NAVIGATE_PAGE": "RETRIEVE",
    "READ_SOURCE": "INSPECT", "SCAN_RESULTS": "INSPECT",
    "EXTRACT_EVIDENCE": "EXTRACT", "FILL_FIELD": "EXTRACT",
    "VERIFY_CLAIM": "VERIFY",
    "WRITE_RESPONSE": "WRITE", "SUBMIT_FORM": "WRITE",
    "SYNTHESIZE_FINDINGS": "SYNTHESIZE",
    "REPAIR_SOURCE": "REPAIR",
}

_EDGE_PROPS = {
    "label": {"type": "string", "enum": DEPENDENCY_LABELS},
    "rationale": {"type": "string"},
    "confidence": {"type": "number", "description": "0..1; <0.5 means ABSTAIN."},
    "artifact_transferred": {
        "type": "string",
        "description": "Artifact handle flowing A->B for DATA_FLOW/VERIFY_OF ('' if none).",
    },
}

PDAF_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_instances": {
            "type": "array",
            "description": "Step 2: the skill instances the trace segments into (a skill type may recur).",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "0-based skill-instance index."},
                    "skill_name": {
                        "type": "string",
                        "description": "snake_case skill type from SKILL_LIBRARY, or a new snake_case name.",
                    },
                    "skill_is_new": {"type": "boolean", "description": "True if not in SKILL_LIBRARY."},
                    "skill_confidence": {"type": "number"},
                },
                "required": ["index", "skill_name", "skill_is_new", "skill_confidence"],
                "additionalProperties": False,
            },
        },
        "cpa_instances": {
            "type": "array",
            "description": "Step 1b: ordered CPA instances covering the whole trace.",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "0-based position across the trace."},
                    "skill_index": {"type": "integer", "description": "index of the owning skill instance."},
                    "cpa": {"type": "string", "enum": CPAS},
                    "phase": {"type": "string", "enum": PHASES, "description": "Phase that owns this CPA."},
                    "event_indices": {"type": "array", "items": {"type": "integer"}},
                    "turn_start": {"type": "integer"},
                    "turn_end": {"type": "integer"},
                    "artifact_id": {"type": "string"},
                    "confidence": {"type": "number", "description": "0..1; <0.5 means ABSTAIN."},
                    "rationale": {"type": "string"},
                },
                "required": ["index", "skill_index", "cpa", "phase", "event_indices",
                             "turn_start", "turn_end", "artifact_id", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
        "local_edges": {
            "type": "array",
            "description": "Step 3a: typed dependency edges over CPA-instance pairs WITHIN a skill (reference cpa index).",
            "items": {
                "type": "object",
                "properties": dict({"source_index": {"type": "integer"},
                                    "target_index": {"type": "integer"}}, **_EDGE_PROPS),
                "required": ["source_index", "target_index", "label", "rationale",
                             "confidence", "artifact_transferred"],
                "additionalProperties": False,
            },
        },
        "global_edges": {
            "type": "array",
            "description": "Step 3b: typed dependency edges over SKILL-INSTANCE pairs (the workflow poset; reference skill index).",
            "items": {
                "type": "object",
                "properties": dict({"source_skill_index": {"type": "integer"},
                                    "target_skill_index": {"type": "integer"}}, **_EDGE_PROPS),
                "required": ["source_skill_index", "target_skill_index", "label", "rationale",
                             "confidence", "artifact_transferred"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["skill_instances", "cpa_instances", "local_edges", "global_edges"],
    "additionalProperties": False,
}
