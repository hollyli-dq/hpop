#!/usr/bin/env bash
# Full HPOP CPA pipeline, end-to-end, reproducible from the raw ingested batches.
# (Re-fetching from HuggingFace is NOT done here — it is rate-limited and non-deterministic;
#  the 5 pilot batches on disk are the fixed substrate. To re-ingest: see hpop.ingest.swe_rebench.)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH=src
D=data/interim/swe_rebench
A=data/annotated/swe_rebench
M=data/modelling/swe_rebench

echo "==================== 0 · merge raw batches -> pilot500 ===================="
cat $D/pilot100.jsonl $D/pilot100_b.jsonl $D/pilot100_c.jsonl $D/pilot100_d.jsonl $D/pilot100_e.jsonl > $D/pilot500.jsonl
echo "  $(wc -l < $D/pilot500.jsonl) trajectories"

echo "==================== 1 · Layer B-1a · normalize -> action tokens ===================="
$PY -m hpop.annotate.normalize --input $D/pilot500.jsonl --output $D/tokens500.jsonl | sed 's/^/  /'

echo "==================== 2-3 · Layer A phase tag + B-1b CPA (v2) ===================="
$PY -m hpop.annotate.rule_apply --input $D/tokens500.jsonl --library rules/cpa_dictionary_v2.json --output $A/cpa_rule | sed -n '1,2p;$p' | sed 's/^/  /'

echo "==================== 4 · sequences (+ phase_sequence) ===================="
$PY -m hpop.extract.sequences --opencode $A/cpa_rule --out $M/sequences | sed -n '1,3p' | sed 's/^/  /'

echo "==================== 5 · recover v1 (29-CPA) sequences from git ===================="
git show 146fa46:data/modelling/swe_rebench/sequences.sequences.jsonl > /tmp/seq_v1.jsonl
echo "  /tmp/seq_v1.jsonl ($(wc -l < /tmp/seq_v1.jsonl) traj)"

echo "==================== 6 · construct no-instruction baselines + ladder ===================="
$PY scripts/build_baselines.py --annot $A/cpa_rule.jsonl --tokens $D/tokens500.jsonl --v1 /tmp/seq_v1.jsonl --outdir $M | sed 's/^/  /'

echo "==================== 7 · head-to-head metrics (physical/CPA/phase) ===================="
$PY scripts/eval_dictionary.py --annot $A/cpa_rule.jsonl --tokens $D/tokens500.jsonl | sed 's/^/  /'

echo "==================== 8 · render dictionary + CPA view HTML ===================="
$PY scripts/render_cpa_dictionary.py --library rules/cpa_dictionary_v2.json --instances $A/cpa_rule.cpa_instances.jsonl --out docs/cpa_dictionary.html | sed 's/^/  /'
$PY scripts/render_from_opencode.py --opencode $A/cpa_rule --traces $D/tokens500.jsonl --library rules/cpa_dictionary_v2.json --out docs/cpa_view.html | sed 's/^/  /'

echo "==================== DONE ===================="
