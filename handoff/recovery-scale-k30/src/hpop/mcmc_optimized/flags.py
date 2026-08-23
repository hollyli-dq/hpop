"""Which optimisations are live, and evidence that each one fired.

Defaults are all-on: this backend exists to be fast, and a caller who wanted the reference
behaviour would import the reference. The flags exist so the four optimisations can be
measured independently and cumulatively *inside one process*, which is the only sound way
to compare them on a machine that is also running a formal chain.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields


@dataclass
class OptimizationFlags:
    inline_logsumexp: bool = True       # O1
    emission_hash_cache: bool = True    # O2
    factorised_forward: bool = True     # O3
    batched_forward: bool = True        # O4

    def reset(self) -> None:
        for field in fields(self):
            setattr(self, field.name, True)

    def all_off(self) -> None:
        """Reduce this backend to the reference algorithm, for A/B measurement."""
        for field in fields(self):
            setattr(self, field.name, False)

    def apply(self, **kwargs) -> None:
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise AttributeError(f"unknown optimisation flag {name!r}")
            setattr(self, name, bool(value))

    def only(self, *names) -> None:
        self.all_off()
        self.apply(**{name: True for name in names})

    def snapshot(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        live = [f.name for f in fields(self) if getattr(self, f.name)]
        return "+".join(live) if live else "reference_algorithm"


@dataclass
class Counters:
    emission_rebuilds: int = 0
    emission_cache_hits: int = 0
    forward_reference_calls: int = 0
    forward_inline_calls: int = 0
    forward_factorised_calls: int = 0
    forward_batched_groups: int = 0
    forward_batched_traces: int = 0

    def reset(self) -> None:
        for field in fields(self):
            setattr(self, field.name, 0)

    def snapshot(self) -> dict:
        return asdict(self)


FLAGS = OptimizationFlags()
COUNTERS = Counters()


def _from_environment() -> None:
    """`HPOP_PERF_FLAGS=inline_logsumexp,batched_forward` selects exactly those.

    Set to `none` to run the reference algorithm through this backend. Unset leaves every
    optimisation on. This lets the existing audit suite be pointed at any configuration
    without a test knowing the flags exist.
    """
    raw = os.environ.get("HPOP_PERF_FLAGS")
    if raw is None:
        return
    raw = raw.strip()
    if raw.lower() in ("", "none", "off"):
        FLAGS.all_off()
        return
    FLAGS.only(*[name.strip() for name in raw.split(",") if name.strip()])


_from_environment()
