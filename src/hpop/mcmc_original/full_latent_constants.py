"""Truth-free fixed specification for the sealed FULL-LATENT experiment.

This module deliberately contains only the numerical model declaration required by
``matched_full_latent``.  It does not import a legacy frozen-model module, a synthetic
generator, a corpus, or terminal recovery truth.  The values are the registered
finite-Markov specialization, copied as literals so that a live formal sampler does
not need to consult a generating configuration to establish its fixed coordinates.
"""

from __future__ import annotations

import math

__all__ = [
    "N_SKILLS", "N_ROLES", "LATENT_DIM", "EPSILON", "DELTA_B",
    "MIN_BLOCK_WIDTH", "MAX_BLOCK_WIDTH", "ETA_INITIAL", "ETA_TRANSITION",
    "FIXED_RHO_0", "FIXED_BETA", "FIXED_OMEGA", "FIXED_LAMBDA_REP",
    "FIXED_LAMBDA_BACK",
]


# Finite-Markov structural model.
N_SKILLS = 3
N_ROLES = 5
LATENT_DIM = 2
EPSILON = 0.02
DELTA_B = 0.15
MIN_BLOCK_WIDTH = 3
MAX_BLOCK_WIDTH = 12
ETA_INITIAL = 1.0
ETA_TRANSITION = 1.0

# Nuisance coordinates held fixed by the preregistered isolation experiment.
FIXED_RHO_0 = 0.5
FIXED_BETA = 1.5
FIXED_OMEGA = math.log(0.85 / 0.15)
FIXED_LAMBDA_REP = 0.8
FIXED_LAMBDA_BACK = 0.25
