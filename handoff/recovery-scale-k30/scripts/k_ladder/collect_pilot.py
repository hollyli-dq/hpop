#!/usr/bin/env python3
"""Merge per-machine pilot output trees into one, refusing any inconsistency.

Each machine writes only the jobs in its own slice, so the four trees are disjoint and
merging is a copy. What this adds is the checking: an operator who reruns a slice on the
wrong commit, or copies a stale tree, produces a summary that looks fine and is not one
experiment. Every mismatch below is a blocking failure, not a warning.

    python scripts/k_ladder/collect_pilot.py --from machine0/ machine1/ machine2/ machine3/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "pilot_manifest.json")
    p.add_argument("--from", dest="sources", type=Path, nargs="+", required=True,
                   help="each machine's results/k_ladder_pilot/factorial directory")
    p.add_argument("--into", type=Path,
                   default=ROOT / "results" / "k_ladder_pilot" / "factorial")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected = {j["key"]: j for j in manifest["jobs"]}
    by_key, problems = defaultdict(list), []
    provenance = defaultdict(set)

    for source in args.sources:
        if not source.exists():
            problems.append(f"source tree missing: {source}")
            continue
        for path in sorted(source.rglob("chain*.json")):
            record = json.loads(path.read_text())
            job = record.get("job", {})
            key = job.get("key")
            if key not in expected:
                problems.append(f"{path}: key {key!r} is not in the manifest")
                continue
            if job.get("job_hash") != expected[key]["job_hash"]:
                problems.append(f"{key}: job hash differs from the manifest "
                                f"(built from a different manifest or edited)")
            by_key[key].append((source, path, record))
            provenance["code_tag"].add(job.get("code_tag"))
            provenance["code_commit"].add(job.get("code_commit"))
            provenance["crn_root"].add(job.get("crn_root"))

    for field, values in provenance.items():
        if len(values) > 1:
            problems.append(f"mixed {field} across machines: {sorted(map(str, values))}")
    if 6_500_000 in provenance.get("crn_root", set()):
        problems.append("a pilot output used the PRODUCTION RNG root 6500000")

    for key, entries in by_key.items():
        if len(entries) > 1:
            digests = {digest(path) for _, path, _ in entries}
            where = [str(s) for s, _, _ in entries]
            if len(digests) > 1:
                problems.append(f"{key}: produced by {where} with DIFFERENT content")
            else:
                problems.append(f"{key}: duplicated across {where} (slices overlap)")

    missing = sorted(set(expected) - set(by_key))
    failed = sorted(k for k, e in by_key.items() if e[0][2].get("status") != "complete")

    print(f"manifest {len(expected)}   found {len(by_key)}   "
          f"missing {len(missing)}   failed {len(failed)}")
    per_source = Counter(str(s) for entries in by_key.values() for s, _, _ in entries)
    for source, n in sorted(per_source.items()):
        print(f"  {n:>4} from {source}")
    for problem in problems[:20]:
        print(f"  BLOCKING  {problem}")
    for key in missing[:10]:
        print(f"  missing   {key}")
    for key in failed[:10]:
        print(f"  failed    {key}")

    if problems or missing or failed:
        print("\nRESULT: BLOCKED -- do not aggregate. Fix the cause and re-collect; a "
              "partial or\n        inconsistent merge is not a pilot.")
        return 1

    if args.dry_run:
        print("\ndry run: nothing copied. All checks passed.")
        return 0

    copied = 0
    for key, entries in by_key.items():
        _, path, _ = entries[0]
        target = Path(expected[key]["output_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(path):
            print(f"  BLOCKING  {key}: refusing to overwrite differing existing output")
            return 1
        if not target.exists():
            shutil.copy2(path, target)
            copied += 1
    print(f"\ncollected {len(by_key)} jobs ({copied} newly copied) into {args.into}")
    print("RESULT: READY -- run aggregate_pilot.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
