"""v2 CPA vocabulary loader — single source of truth = rules/cpas_v2.yaml (phase-derived).

Exposes the CPA -> 9-phase map and the flat CPA list so the annotator can tag each occurrence with its
cognitive phase (Layer A, Miller & Cohen 2001) and downstream tools can group by phase/level.
"""
from __future__ import annotations

import os

import yaml

_YAML = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rules", "cpas_v2.yaml")

PHASE_ORDER = ["PLAN", "RETRIEVE", "INSPECT", "EXTRACT", "VERIFY", "WRITE", "SYNTHESIZE", "REPAIR", "HANDOFF"]
PHASE_LEVEL = {"PLAN": "L1", "RETRIEVE": "L2", "INSPECT": "L2", "EXTRACT": "L2", "VERIFY": "L2",
               "WRITE": "L2", "SYNTHESIZE": "L3", "REPAIR": "L4", "HANDOFF": "L4"}
PHASE_LEVEL_NAME = {"L1": "goal-setting", "L2": "execution", "L3": "integration", "L4": "interrupt"}


def _load():
    d = yaml.safe_load(open(_YAML))
    cpa_phase, cpas = {}, []
    for ph in PHASE_ORDER:
        for it in (d.get(ph) or []):
            if not isinstance(it, dict):
                continue
            cpa_phase[it["name"]] = ph
            cpas.append({"name": it["name"], "phase": ph, "level": PHASE_LEVEL[ph],
                         "definition": it.get("proximate_goal", ""), "distinguish": it.get("distinguish", ""),
                         "trigger": it.get("trigger", ""), "status": it.get("status", "active")})
    return cpa_phase, cpas


CPA_PHASE, CPAS_V2 = _load()
