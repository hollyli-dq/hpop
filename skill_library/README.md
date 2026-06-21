# Skill Library

The reusable skill programs recovered from traces, stored as **DAG artifacts** and **indexed by
phase**. This is the offline output of the pipeline and the input to fast inference.

Each entry is a skill *type* (a reusable pattern), holding:

- its **phase** tag (from `rules/phases.yaml`),
- the **local partial order** over its canonical action tokens (the DAG / recipe),
- provenance (which trace instances it was recovered from),
- statistics (recurrence count, success/failure outcomes).

Indexed by phase so that "what skills can fill the *Explore* slot" is a first-class query
(DESIGN.md §3). Storage format is defined alongside `src/hpop/library/`.

> Empty until the extraction pipeline runs. Generated artifacts; not hand-edited.
