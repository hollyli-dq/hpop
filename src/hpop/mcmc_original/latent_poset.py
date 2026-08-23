"""The original latent partial-order representation: ``U_k -> h(U_k)``.

For skill k the latent variable is ``U_k in R^{m_k x d}``, one row per role and
one column per coordinate of a d-dimensional product order. The induced strict
partial order is coordinate-wise dominance:

    i > j   iff   U_k[i, r] > U_k[j, r]  for every r = 1, ..., d.

This is a *strict product order*, and it is a strict partial order for any real
U whatsoever: irreflexivity, asymmetry and transitivity are inherited from ``>``
on the reals coordinate by coordinate. Order structure is therefore never
learned as edges and never needs an acyclicity check — moving U can only ever
move between valid partial orders.

The matrix returned by :func:`precedence_from_u` is already transitively closed:
``P[i, j]`` is the full precedence relation, not a Hasse/cover relation. The
transitive reduction is a presentation concern for later visualisation and is
deliberately not computed here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["precedence_from_u", "predecessors", "successors", "incomparable"]


def _as_precedence_matrix(precedence: np.ndarray) -> np.ndarray:
    """Validate and return a square boolean precedence matrix."""
    p = np.asarray(precedence)
    if p.dtype != np.bool_:
        raise ValueError(f"precedence must be a boolean matrix, got dtype {p.dtype}")
    if p.ndim != 2:
        raise ValueError(f"precedence must be 2-D, got ndim={p.ndim}")
    if p.shape[0] != p.shape[1]:
        raise ValueError(f"precedence must be square, got shape {p.shape}")
    if p.shape[0] < 1:
        raise ValueError("precedence must cover at least one role")
    return p


def _check_role(precedence: np.ndarray, x: int, name: str = "x") -> int:
    """Validate that ``x`` is an in-range role index."""
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer role index, got {type(x).__name__}")
    x = int(x)
    m = precedence.shape[0]
    if not (0 <= x < m):
        raise ValueError(f"{name}={x} is out of range for {m} roles")
    return x


def precedence_from_u(u: np.ndarray) -> np.ndarray:
    """Map a latent embedding ``U`` to the strict partial order it induces.

    Args:
        u: Latent positions of shape ``(m, d)``: m roles in a d-dimensional
            product order. Must be finite and convertible to float.

    Returns:
        Boolean array ``P`` of shape ``(m, m)`` where ``P[i, j]`` is True iff
        ``u[i, r] > u[j, r]`` for every coordinate r — i.e. role i strictly
        precedes role j. The diagonal is False. ``P`` is the transitive closure,
        not a cover relation.

    Raises:
        ValueError: If ``u`` is not a finite 2-D array with at least one role
            and latent dimension at least 1.
    """
    try:
        arr = np.asarray(u, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"u must be convertible to a float array: {exc}") from exc

    if arr.ndim != 2:
        raise ValueError(f"u must be 2-D of shape (m, d), got ndim={arr.ndim}")
    m, d = arr.shape
    if m < 1:
        raise ValueError("u must contain at least one role")
    if d < 1:
        raise ValueError(f"latent dimension must be >= 1, got d={d}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("u must not contain NaN or inf values")

    # p[i, j] = AND over coordinates of (u[i, r] > u[j, r]).
    p = np.all(arr[:, None, :] > arr[None, :, :], axis=-1)
    # Strict '>' already gives a False diagonal; set it explicitly so
    # irreflexivity does not depend on floating-point behaviour.
    np.fill_diagonal(p, False)
    return p


def predecessors(precedence: np.ndarray, x: int) -> tuple[int, ...]:
    """Roles that strictly precede ``x``: ``{z : P[z, x]}``, in index order."""
    p = _as_precedence_matrix(precedence)
    x = _check_role(p, x)
    return tuple(int(z) for z in np.flatnonzero(p[:, x]))


def successors(precedence: np.ndarray, x: int) -> tuple[int, ...]:
    """Roles that ``x`` strictly precedes: ``{z : P[x, z]}``, in index order."""
    p = _as_precedence_matrix(precedence)
    x = _check_role(p, x)
    return tuple(int(z) for z in np.flatnonzero(p[x, :]))


def incomparable(precedence: np.ndarray, i: int, j: int) -> bool:
    """True iff ``i`` and ``j`` are distinct and neither precedes the other."""
    p = _as_precedence_matrix(precedence)
    i = _check_role(p, i, "i")
    j = _check_role(p, j, "j")
    if i == j:
        return False
    return not bool(p[i, j]) and not bool(p[j, i])
