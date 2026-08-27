"""Immutable dataclasses for the original latent partial-order model.

Indexing convention
-------------------
All spans are half-open ``[start, end)`` in Python's own convention, so a
segment maps directly onto a slice::

    trace[segment.start:segment.end]

This module contains data containers and their validation only. No probability,
likelihood, or inference logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Segment", "Segmentation", "SkillTemplate"]


@dataclass(frozen=True)
class Segment:
    """A half-open span ``[start, end)`` of a trace, labelled with one skill.

    Attributes:
        start: First index of the span, inclusive. Must be >= 0.
        end: One past the last index of the span. Must be > ``start``.
        skill: Index of the skill generating this span. Must be >= 0.
    """

    start: int
    end: int
    skill: int

    def __post_init__(self) -> None:
        for name in ("start", "end", "skill"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"Segment.{name} must be an integer, got {type(value).__name__}"
                )
            # Normalise numpy integers to plain ints so equality/hashing behave.
            object.__setattr__(self, name, int(value))

        if self.start < 0:
            raise ValueError(f"Segment.start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"Segment.end must be > start (half-open span), "
                f"got start={self.start}, end={self.end}"
            )
        if self.skill < 0:
            raise ValueError(f"Segment.skill must be >= 0, got {self.skill}")

    @property
    def length(self) -> int:
        """Number of trace positions covered by this segment."""
        return self.end - self.start


@dataclass(frozen=True)
class Segmentation:
    """A complete, gap-free, non-overlapping tiling of a trace by segments.

    The segments must start at 0 and be contiguous: ``segments[i].end`` equals
    ``segments[i + 1].start``. This rules out both gaps and overlaps, and makes
    the end positions strictly increasing.
    """

    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        segments = self.segments
        if isinstance(segments, Segment):
            raise ValueError("Segmentation.segments must be a sequence of Segment")
        if not isinstance(segments, tuple):
            try:
                segments = tuple(segments)
            except TypeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "Segmentation.segments must be a sequence of Segment"
                ) from exc
            object.__setattr__(self, "segments", segments)

        if len(segments) == 0:
            raise ValueError("Segmentation must contain at least one segment")

        for index, segment in enumerate(segments):
            if not isinstance(segment, Segment):
                raise ValueError(
                    f"Segmentation.segments[{index}] must be a Segment, "
                    f"got {type(segment).__name__}"
                )

        if segments[0].start != 0:
            raise ValueError(
                f"Segmentation must start at 0, got start={segments[0].start}"
            )

        for index in range(len(segments) - 1):
            current, following = segments[index], segments[index + 1]
            if current.end != following.start:
                relation = "a gap" if following.start > current.end else "an overlap"
                raise ValueError(
                    f"Segmentation has {relation} between segments {index} and "
                    f"{index + 1}: segments[{index}].end={current.end} != "
                    f"segments[{index + 1}].start={following.start}"
                )
            # Contiguity plus end > start already forces this, but assert it so
            # the invariant is checked rather than merely implied.
            if following.end <= current.end:
                raise ValueError(
                    f"Segmentation end positions must strictly increase, got "
                    f"{current.end} then {following.end}"
                )

    @property
    def length(self) -> int:
        """Total length of the trace covered, i.e. the final end position."""
        return self.segments[-1].end

    def __len__(self) -> int:
        return len(self.segments)


@dataclass(frozen=True, eq=False)
class SkillTemplate:
    """One skill: a set of roles plus their latent product-order embedding.

    Attributes:
        cpa_labels: The CPA label of each role, one per row of ``u``. Labels are
            required to be unique in this first static model: a role is
            identified with its CPA, so recurrence is not yet representable.
        u: Latent positions ``U_k`` of shape ``(m_k, d)``. The strict partial
            order over roles is induced from this by coordinate-wise dominance
            (see :func:`hpop.mcmc_original.latent_poset.precedence_from_u`); it
            is never stored as edges.
        beta: Non-negative order-violation penalty / inverse temperature.
        epsilon: Noise probability, in ``[0, 1)``.

    ``eq=False`` because ``u`` is a numpy array: the generated ``__eq__`` would
    return an array and the generated ``__hash__`` would fail on it. Instances
    therefore compare and hash by identity.
    """

    cpa_labels: tuple[int, ...]
    u: np.ndarray
    beta: float
    epsilon: float

    def __post_init__(self) -> None:
        labels = self.cpa_labels
        if isinstance(labels, (str, bytes)) or not hasattr(labels, "__iter__"):
            raise ValueError("SkillTemplate.cpa_labels must be a sequence of ints")
        if not isinstance(labels, tuple):
            labels = tuple(labels)
        normalised: list[int] = []
        for index, label in enumerate(labels):
            if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
                raise ValueError(
                    f"SkillTemplate.cpa_labels[{index}] must be an integer, "
                    f"got {type(label).__name__}"
                )
            normalised.append(int(label))
        labels = tuple(normalised)
        object.__setattr__(self, "cpa_labels", labels)

        if len(labels) == 0:
            raise ValueError("SkillTemplate must have at least one role")
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"SkillTemplate.cpa_labels must be unique in this static model, "
                f"got {labels}"
            )

        # Copy, so freezing the array below never mutates the caller's input.
        u = np.array(self.u, dtype=float, copy=True)
        if u.ndim != 2:
            raise ValueError(
                f"SkillTemplate.u must be 2-D of shape (m, d), got ndim={u.ndim}"
            )
        if u.shape[0] != len(labels):
            raise ValueError(
                f"SkillTemplate.u must have one row per CPA label, got "
                f"u.shape[0]={u.shape[0]} and {len(labels)} labels"
            )
        if u.shape[1] < 1:
            raise ValueError(
                f"SkillTemplate latent dimension must be >= 1, got d={u.shape[1]}"
            )
        if not np.all(np.isfinite(u)):
            raise ValueError("SkillTemplate.u must not contain NaN or inf values")
        u.flags.writeable = False
        object.__setattr__(self, "u", u)

        beta = self.beta
        if isinstance(beta, bool) or not isinstance(beta, (int, float, np.number)):
            raise ValueError(
                f"SkillTemplate.beta must be a real number, got {type(beta).__name__}"
            )
        beta = float(beta)
        if not np.isfinite(beta):
            raise ValueError(f"SkillTemplate.beta must be finite, got {beta}")
        if beta < 0.0:
            raise ValueError(f"SkillTemplate.beta must be >= 0, got {beta}")
        object.__setattr__(self, "beta", beta)

        epsilon = self.epsilon
        if isinstance(epsilon, bool) or not isinstance(
            epsilon, (int, float, np.number)
        ):
            raise ValueError(
                f"SkillTemplate.epsilon must be a real number, "
                f"got {type(epsilon).__name__}"
            )
        epsilon = float(epsilon)
        if not np.isfinite(epsilon):
            raise ValueError(f"SkillTemplate.epsilon must be finite, got {epsilon}")
        if not (0.0 <= epsilon < 1.0):
            raise ValueError(
                f"SkillTemplate.epsilon must satisfy 0 <= epsilon < 1, got {epsilon}"
            )
        object.__setattr__(self, "epsilon", epsilon)

    @property
    def n_roles(self) -> int:
        """Number of roles ``m_k`` in this skill."""
        return self.u.shape[0]

    @property
    def latent_dim(self) -> int:
        """Product-order dimension ``d``."""
        return self.u.shape[1]
