"""Vendored third-party research code — DO NOT EDIT anything under this package.

`po_inference_agent/` holds a byte-identical copy of the **verified** partial-order
MCMC from a separate project. It is vendored, not reimplemented, because Stage 5 must
reuse the already-validated update rules for `U`, `rho` and the softmax `beta` rather
than substitute new ones.

Provenance, including the source commit and a sha256 for every file, is in
`PROVENANCE.json`. Run `python -m hpop.vendored.check_drift` to verify the copies still
match the source project (when it is reachable) and that nothing here has been edited.

Why an import shim instead of rewriting imports
-----------------------------------------------
The vendored modules import each other as `src.utils.po_fun`, matching their original
repository layout. Rewriting those imports would modify the files, which defeats the
point of vendoring a verified implementation. Instead :func:`ensure_importable` puts
the vendored root on ``sys.path`` so ``src.utils.*`` resolves to the copies here.

That does introduce a top-level ``src`` package name. This repository has no such
package of its own (its layout is ``src/hpop/...`` with ``src`` as a source *root*,
not a package), so there is no shadowing — but :func:`ensure_importable` verifies the
resolved file really is the vendored one and raises if anything else claims the name.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VENDOR_ROOT / "po_inference_agent"

__all__ = ["VENDOR_ROOT", "PROJECT_ROOT", "ensure_importable"]


def ensure_importable() -> Path:
    """Make the vendored ``src.utils.*`` / ``src.mcmc.*`` modules importable.

    Idempotent. Returns the path that was added to ``sys.path``.

    Raises:
        ImportError: if a *different* top-level ``src`` package has already been
            imported, which would silently shadow the vendored code.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    existing = sys.modules.get("src")
    if existing is not None:
        origin = getattr(existing, "__file__", None) or ""
        if not str(Path(origin).resolve()).startswith(str(PROJECT_ROOT)):
            raise ImportError(
                f"a different top-level 'src' package is already imported from {origin!r}; "
                "it would shadow the vendored partial-order sampler"
            )
    return PROJECT_ROOT
