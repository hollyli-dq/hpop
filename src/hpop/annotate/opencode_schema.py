"""Structured-output schema for the occurrence-level CPA annotator (collaborator spec).

DATA PREPARATION ONLY: occurrence-level Canonical Procedural Actions. The annotator must NOT infer
skills, local/global partial orders, phases, or hidden reasoning — those are learned downstream by
BPOP/HPOP. Modes: INDUCE (propose candidate CPAs) / APPLY (match a frozen library).

This JSON Schema mirrors the exact output structure in the spec; it is passed via
`output_config.format` so every annotation conforms. Nullable fields use ["string","null"].
"""

DECISIONS = ["MATCH_EXISTING", "PROPOSE_NEW", "ABSTAIN"]
OUTCOMES = ["SUCCESS", "FAILURE", "PARTIAL", "UNKNOWN"]
EXCLUDE_REASONS = ["NON_ACTION_METADATA", "CONVERSATIONAL_TEXT", "LOGGING_DUPLICATE", "OTHER"]
REVIEW_REASONS = ["LOW_CONFIDENCE", "NEW_CPA", "AMBIGUOUS_BOUNDARY", "OTHER"]
PRIORITIES = ["HIGH", "MEDIUM", "LOW"]

_S = {"type": "string"}
_SN = {"type": ["string", "null"]}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}
_SARR = {"type": "array", "items": _S}

_ARTIFACT = {
    "type": "object",
    "properties": {"artifact": _S, "evidence_event_ids": _SARR},
    "required": ["artifact", "evidence_event_ids"], "additionalProperties": False,
}

_CPA_INSTANCE = {
    "type": "object",
    "properties": {
        "occurrence_id": _S,
        "source_event_ids": _SARR,
        "start_event_id": _S,
        "end_event_id": _S,
        "decision": {"type": "string", "enum": DECISIONS},
        "canonical_label": _SN,
        "candidate_label": _SN,
        "definition": _S,
        "procedural_function": _S,
        "input_artifacts": {"type": "array", "items": _ARTIFACT},
        "output_artifacts": {"type": "array", "items": _ARTIFACT},
        "state_before": _SN,
        "state_after": _SN,
        "outcome": {"type": "string", "enum": OUTCOMES},
        "boundary_confidence": _NUM,
        "label_confidence": _NUM,
        "evidence": _SARR,
        "ambiguity": _SN,
        "review_required": _BOOL,
    },
    "required": ["occurrence_id", "source_event_ids", "start_event_id", "end_event_id", "decision",
                 "canonical_label", "candidate_label", "definition", "procedural_function",
                 "input_artifacts", "output_artifacts", "state_before", "state_after", "outcome",
                 "boundary_confidence", "label_confidence", "evidence", "ambiguity", "review_required"],
    "additionalProperties": False,
}

CPA_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "trajectory_id": _S,
        "mode": {"type": "string", "enum": ["INDUCE", "APPLY"]},
        "library_version": _SN,
        "cpa_instances": {"type": "array", "items": _CPA_INSTANCE},
        "excluded_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"event_id": _S, "reason": {"type": "string", "enum": EXCLUDE_REASONS}},
                "required": ["event_id", "reason"], "additionalProperties": False,
            },
        },
        "candidate_library_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_label": _S, "definition": _S, "occurrence_ids": _SARR,
                    "reason_new": _S, "nearest_existing_labels": _SARR,
                },
                "required": ["candidate_label", "definition", "occurrence_ids", "reason_new", "nearest_existing_labels"],
                "additionalProperties": False,
            },
        },
        "review_queue": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "occurrence_id": _S, "reason": {"type": "string", "enum": REVIEW_REASONS},
                    "priority": {"type": "string", "enum": PRIORITIES},
                },
                "required": ["occurrence_id", "reason", "priority"], "additionalProperties": False,
            },
        },
        "quality_checks": {
            "type": "object",
            "properties": {
                "action_event_coverage_complete": _BOOL,
                "spans_sorted": _BOOL,
                "spans_non_overlapping": _BOOL,
                "repeated_occurrences_preserved": _BOOL,
                "skill_labels_inferred": _BOOL,
                "unassigned_action_event_ids": _SARR,
            },
            "required": ["action_event_coverage_complete", "spans_sorted", "spans_non_overlapping",
                         "repeated_occurrences_preserved", "skill_labels_inferred", "unassigned_action_event_ids"],
            "additionalProperties": False,
        },
    },
    "required": ["trajectory_id", "mode", "library_version", "cpa_instances", "excluded_events",
                 "candidate_library_updates", "review_queue", "quality_checks"],
    "additionalProperties": False,
}
