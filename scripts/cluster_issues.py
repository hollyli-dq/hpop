"""Cluster SWE-rebench issue statements to define question-type strata for trace sampling.

TF-IDF (no API key) -> KMeans over a sweep of k, pick k by silhouette, then report per-cluster:
size, top terms, example issues, repo spread, and resolved rate. The cluster sizes give the natural
proportions; allocation across clusters (floor + oversample failure-heavy ones) is decided downstream.

Usage:
    .venv/bin/python scripts/cluster_issues.py --input data/interim/swe_rebench/issues_5k.jsonl \
        --kmin 6 --kmax 16 --output data/interim/swe_rebench/issue_clusters.json
"""
from __future__ import annotations
import argparse, json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# domain-noise terms that swamp task semantics (repo names, boilerplate)
STOP_EXTRA = {"issue", "description", "repository", "workspace", "code", "file", "files",
              "change", "changes", "implement", "make", "following", "consider", "use", "using",
              "test", "tests", "python", "self", "def", "class", "return", "value", "values"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    docs = [r["issue"] for r in rows]
    print(f"{len(docs)} issues")

    stop = list(TfidfVectorizer(stop_words="english").get_stop_words() | STOP_EXTRA)
    vec = TfidfVectorizer(stop_words=stop, max_features=4000, ngram_range=(1, 2),
                          min_df=5, max_df=0.4, sublinear_tf=True)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    best = None
    for k in range(args.kmin, args.kmax + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(3000, len(docs)), random_state=0)
        print(f"  k={k:2d}  silhouette={sil:.4f}")
        if best is None or sil > best[0]:
            best = (sil, k, km)
    sil, k, km = best
    print(f"chosen k={k} (silhouette={sil:.4f})")

    labels = km.labels_
    order = np.argsort(km.cluster_centers_, axis=1)[:, ::-1]
    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        top = terms[order[c][:12]].tolist()
        repos = {}
        res = 0
        for i in idx:
            repos[rows[i]["repo"]] = repos.get(rows[i]["repo"], 0) + 1
            res += 1 if rows[i].get("resolved") else 0
        ex = [rows[i]["issue"][:160] for i in idx[:3]]
        clusters.append({
            "cluster": c, "size": int(len(idx)), "frac": round(len(idx) / len(docs), 3),
            "resolved_rate": round(res / max(1, len(idx)), 3),
            "n_repos": len(repos), "top_terms": top, "examples": ex,
        })
    clusters.sort(key=lambda d: -d["size"])

    out = {"k": k, "silhouette": round(sil, 4), "n": len(docs), "clusters": clusters}
    json.dump(out, open(args.output, "w"), indent=2)

    print("\n=== CLUSTERS ===")
    for cl in clusters:
        print(f"\n[{cl['cluster']}] n={cl['size']} ({cl['frac']:.1%})  "
              f"resolved={cl['resolved_rate']:.0%}  repos={cl['n_repos']}")
        print("   terms:", ", ".join(cl["top_terms"]))
        print("   e.g. :", cl["examples"][0][:120])
    print(f"\nwrote -> {args.output}")


if __name__ == "__main__":
    main()
