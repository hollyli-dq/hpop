"""CPA-vocabulary extension: per-skill role supports over a larger action alphabet."""

from .block_tables import CPABlockScoreTable, assert_matches_sealed_scorer
from .role_maps import NOT_IN_SUPPORT, RoleMaps, sample_role_maps

__all__ = ["CPABlockScoreTable", "assert_matches_sealed_scorer", "RoleMaps",
           "sample_role_maps", "NOT_IN_SUPPORT"]
