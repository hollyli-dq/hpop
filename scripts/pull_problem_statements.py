"""Pull problem_statement for a sample of SWE-rebench TASK instances (light dataset, no trajectories).

nebius/SWE-rebench has a clean `problem_statement` column. We stride offsets across the split so the
sample spans the whole question space. These are the questions; the 67k trajectories map in by
instance_id downstream.

Usage:
    .venv/bin/python scripts/pull_problem_statements.py --n 3000 --split test \
        --output data/interim/swe_rebench/issues_3k.jsonl
"""
from __future__ import annotations
import argparse, json, re, time, urllib.parse, urllib.request

DATASET = "nebius/SWE-rebench"
SPLIT_ROWS = {"test": 21336, "filtered": 6542}
BASE = "https://datasets-server.huggingface.co/rows"


def _get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            print("  fetch err:", str(e)[:60]); time.sleep(2 * (k + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    total = SPLIT_ROWS[args.split]
    page = 100
    n_pages = min((args.n + page - 1) // page, total // page)
    stride = max(1, total // n_pages)
    offsets = [min(i * stride, total - page) for i in range(n_pages)]

    cfg = urllib.parse.quote(DATASET, safe="")
    out, seen = [], set()
    with open(args.output, "w") as fh:
        for j, off in enumerate(offsets):
            url = f"{BASE}?dataset={cfg}&config=default&split={args.split}&offset={off}&length={page}"
            data = _get(url)
            if not data:
                continue
            for rw in data.get("rows", []):
                row = rw.get("row", {})
                iid = row.get("instance_id")
                if iid in seen:
                    continue
                seen.add(iid)
                txt = re.sub(r"\s+", " ", row.get("problem_statement") or "").strip()
                if len(txt) < 20:
                    continue
                fh.write(json.dumps({"instance_id": iid, "repo": row.get("repo"),
                                     "issue": txt[:6000]}) + "\n")
            fh.flush()
            print(f"  {j+1}/{len(offsets)} pages, {len(seen)} issues", flush=True)
            time.sleep(0.1)
    print(f"wrote {len(seen)} issues -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
