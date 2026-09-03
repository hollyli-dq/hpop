"""The registered regime of the recovery-at-scale experiment. One module, all constants.

Every value here was either measured into place by the corpus calibration
(CORPUS_CALIBRATION_MEMO.md) or set by the registered design (RECOVERY_AT_SCALE_DESIGN.md).
Nothing downstream may restate these as its own defaults: generators, runners and gates
import them, so the experiment cannot drift apart from its registration.
"""

from __future__ import annotations

import numpy as np

from hpop.mcmc_original.recurrent_rfs import RecurrentRFSParameters
from hpop.mcmc_original.recurrent_scalar_posterior import TRUE_VALUES

__all__ = ["REGIME", "generation_params", "regime_dict"]


class REGIME:
    """Frozen at registration. Change = new experiment version, never an edit."""

    # ---- generation (calibrated: beta=0 dominates beta=1 at every corpus tier;
    #      widths/J/multiplier are the cheapest cell meeting the evidence standard)
    BETA = 0.0                       # traces = constrained linear extensions (BPOP regime)
    EPSILON = 0.02                   # feasibility channel stays sharp
    MIN_WIDTH = 8
    MAX_WIDTH = 20
    TRACE_LENGTH = 128
    TRAIN_PER_SKILL = 15             # 3x the old 5; edge witnessing scales linearly
    TEST_PER_SKILL = 4
    DELTA_B = 0.15

    # ---- evidence standard (measured with the vendored BPOP machinery)
    IP_COV_TARGET = 0.90             # incomparable pairs witnessed in BOTH orders
    RESOLVED_TARGET = 0.90           # all-pairs resolved fraction
    EDGE_COOC_MIN = 5

    # ---- library admissibility (invocation exposure; floor set from measurement)
    EXPOSURE_PROBES = 400            # prior-predictive invocations per skill
    EXPOSURE_FLOOR = None            # set after the eta measurement; None = not yet frozen

    # ---- ladder
    K_LADDER = (3, 5, 10, 20, 30)
    REPLICATES = (0, 1)
    CHAINS = 4

    # ---- inference (same engine at every K; scores at the SAME beta as generation)
    U_EVERY = 10                     # path-marginal collapsed-U cadence
    # Effort pacing: proposals/(role*sweep), flat in K. PRICED on the frozen dataset
    # (2026-09-03, M4, single thread): at K=30 one U move costs ~9.7 s and one FFBS
    # sweep ~4.5 s, so 0.002 balances U work against FFBS work at the top rung
    # (~10 s/sweep total; ~62 s at the 0.02 an earlier draft shipped unpriced). The cap
    # then implies at most 0.002 * 100,000 = 200 proposals per role -- the structural-
    # epoch bound arrives as a DERIVED consequence of (rate, cap), not as an input.
    U_RATE_PER_ROLE_PER_SWEEP = 0.002
    SEGMENT_SWEEPS = 2_000           # one resumable work unit
    THIN = 5                         # segmentation draw thinning (U is on the event axis)
    CHECK_EVERY_SEGMENTS = 1         # coordinator evaluates gates after every segment

    # ---- run-to-convergence (effort is an OUTCOME; only the gates and cap are inputs)
    RHAT_MAX = 1.05
    ESS_MIN = 100.0
    ACCEPT_WINDOW = (0.15, 0.60)
    CAP_SWEEPS = 100_000             # hard cap; hitting it = inference FAIL at that K
    CAP_IS_RESOURCE_STATEMENT = ("the cap is the fleet budget, not a tuning knob; "
                                 "a K that needs more is reported as inference FAIL")

    # ---- streams
    ROOT_ENTROPY = 6_900_000         # fresh root; pilot used 6.7e6, production seeds 6.5e6


def generation_params() -> RecurrentRFSParameters:
    return RecurrentRFSParameters(
        beta=float(REGIME.BETA), epsilon=float(REGIME.EPSILON),
        shared_omega=float(TRUE_VALUES["omega"]),
        lambda_rep=float(TRUE_VALUES["lambda_rep"]),
        lambda_back=float(TRUE_VALUES["lambda_back"]))


def regime_dict() -> dict:
    """The registration record embedded in every artifact this experiment writes."""
    out = {name: getattr(REGIME, name) for name in dir(REGIME)
           if name.isupper()}
    out["omega"] = float(TRUE_VALUES["omega"])
    out["lambda_rep"] = float(TRUE_VALUES["lambda_rep"])
    out["lambda_back"] = float(TRUE_VALUES["lambda_back"])
    out["schema"] = "recovery-at-scale-regime/1.0.0"
    return out
