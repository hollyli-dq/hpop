"""Verify the vendored copies are unedited, and still match the source project.

    python -m hpop.vendored.check_drift

Exit status 0 when clean. Two distinct failures are reported separately:

* **edited** — a vendored file no longer matches the sha256 recorded at vendoring
  time. That is a bug: nothing here may be modified.
* **drifted** — the source project has changed since vendoring. Not necessarily a
  bug, but it means the verified implementation has moved on and the vendored copy
  should be refreshed deliberately.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from hpop.vendored import PROJECT_ROOT, VENDOR_ROOT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((VENDOR_ROOT / "PROVENANCE.json").read_text())
    source_root = Path(manifest["vendored_from"]["project"])
    edited, drifted, missing = [], [], []

    for relative, record in manifest["files"].items():
        local = PROJECT_ROOT / relative
        if not local.exists():
            missing.append(relative)
            continue
        if _sha256(local) != record["sha256"]:
            edited.append(relative)
        source = source_root / relative
        if source.exists() and _sha256(source) != record["sha256"]:
            drifted.append(relative)

    print(f"vendored from : {source_root}")
    print(f"source commit : {manifest['vendored_from']['git_commit']}")
    print(f"files checked : {len(manifest['files'])}")
    print(f"source reachable: {source_root.exists()}")
    print()
    if missing:
        print(f"MISSING ({len(missing)}): {missing}")
    if edited:
        print(f"EDITED ({len(edited)}) — vendored code must never be modified:")
        for name in edited:
            print(f"   {name}")
    if drifted:
        print(f"DRIFTED ({len(drifted)}) — source has changed since vendoring:")
        for name in drifted:
            print(f"   {name}")
    if not (missing or edited or drifted):
        print("clean: every vendored file is unedited and matches the source project")
    return 1 if (missing or edited) else 0


if __name__ == "__main__":
    raise SystemExit(main())
