"""Extract (instance_id, repo, problem_statement) from SWE-rebench task parquet -> issues jsonl.

The 21k task instances are the question space; the 67k trajectories are repeated runs over them.
We cluster the instances, then map trajectories to clusters via instance_id downstream.

Usage:
    .venv/bin/python scripts/parquet_to_issues.py \
        data/interim/swe_rebench/parquet/test_0000.parquet data/interim/swe_rebench/parquet/test_0001.parquet \
        --output data/interim/swe_rebench/issues_all.jsonl
"""
from __future__ import annotations
import argparse, json, re
import pyarrow.parquet as pq

COLS = ["instance_id", "repo", "problem_statement"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.output, "w") as out:
        for path in args.parquet:
            tbl = pq.read_table(path, columns=COLS)
            d = tbl.to_pydict()
            for iid, repo, ps in zip(d["instance_id"], d["repo"], d["problem_statement"]):
                txt = re.sub(r"\s+", " ", ps or "").strip()
                if len(txt) < 20:
                    continue
                out.write(json.dumps({"instance_id": iid, "repo": repo,
                                      "issue": txt[:6000]}) + "\n")
                n += 1
    print(f"wrote {n} issues -> {args.output}")


if __name__ == "__main__":
    main()
