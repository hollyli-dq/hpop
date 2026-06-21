"""HPOP — raw agent traces -> reusable skill programs as partial orders -> fast DAG inference.

Pipeline subpackages:
    ingest     WebLINX loading -> raw events
    annotate   LLM annotation pipeline (rules -> structured labels)
    extract    annotations -> two nested partial orders / DAGs
    library    skill library: store, index by phase, retrieve
    inference  fast DAG-based inference / execution
    eval       validation vs human gold, metrics

See docs/DESIGN.md for the data model and the core thesis.
"""

__version__ = "0.0.1"
