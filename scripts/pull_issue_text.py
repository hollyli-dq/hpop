"""Pull issue-description text for a sample of SWE-rebench traces, for question clustering.

We only need the issue statement (the `<issue_description>` block in the first user message) plus
light metadata (repo, resolved). Spread the sample across the 67k by striding the offset space so the
cluster structure reflects the whole dataset, not just the head.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/pull_issue_text.py --n 5000 --output data/interim/swe_rebench/issues_5k.jsonl
"""
from __future__ import annotations
import argparse, json, re, time, urllib.parse, urllib.request

DATASET = "nebius/SWE-rebench-openhands-trajectories"
TOTAL = 67074
BASE = "https://datasets-server.huggingface.co/rows"
ISSUE_RE = re.compile(r"<issue_description>(.*?)</issue_description>", re.S)


def _get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(1.5 * (k + 1))
    return None


def extract_issue(row):
    msgs = row.get("messages") or []
    issue = next((m.get("content", "") for m in msgs
                  if m.get("role") == "user" and m.get("content")), "")
    m = ISSUE_RE.search(issue)
    body = m.group(1) if m else re.sub(r"<uploaded_files>.*?</uploaded_files>", "", issue, flags=re.S)
    return re.sub(r"\s+", " ", body).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    page = 100
    n_pages = (args.n + page - 1) // page
    # stride offsets across the whole dataset
    stride = max(1, TOTAL // n_pages)
    offsets = [min(i * stride, TOTAL - page) for i in range(n_pages)]

    out, seen = [], set()
    cfg = urllib.parse.quote(DATASET, safe="")
    for j, off in enumerate(offsets):
        url = f"{BASE}?dataset={cfg}&config=default&split=train&offset={off}&length={page}"
        data = _get(url)
        if not data:
            print(f"  skip offset {off}"); continue
        for rw in data.get("rows", []):
            row = rw.get("row", {})
            iid = row.get("instance_id")
            if iid in seen:
                continue
            seen.add(iid)
            txt = extract_issue(row)
            if len(txt) < 20:
                continue
            out.append({
                "instance_id": iid,
                "repo": row.get("repo"),
                "resolved": row.get("resolved"),
                "issue": txt[:4000],
            })
        if j % 10 == 0:
            print(f"  {j+1}/{len(offsets)} pages, {len(out)} issues")
        time.sleep(0.15)

    with open(args.output, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} issues -> {args.output}")


if __name__ == "__main__":
    main()
