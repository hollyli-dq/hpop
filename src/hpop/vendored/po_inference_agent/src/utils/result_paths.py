from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"
RESULTS_ROOT = PROJECT_ROOT / "results"

CLOUD_IAC_RESULTS_DIR = RESULTS_ROOT / "cloud_iac" / "systematic_experiments"
LEGACY_CLOUD_IAC_RESULTS_DIR = NOTEBOOKS_ROOT / "systematic_experiment_results"
CLOUD_IAC_ANALYSIS_OUTPUT_DIR = RESULTS_ROOT / "cloud_iac" / "agent_trace_analysis"

WFINSTANCES_RESULTS_ROOT = RESULTS_ROOT / "wfinstances"
WFINSTANCES_SRASEARCH_RESULTS_DIR = WFINSTANCES_RESULTS_ROOT / "srasearch"
LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR = NOTEBOOKS_ROOT / "wfinstances_srasearch_results"
WFINSTANCES_EPIGENOMICS_RESULTS_DIR = WFINSTANCES_RESULTS_ROOT / "epigenomics"
LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR = NOTEBOOKS_ROOT / "wfinstances_epigenomics_results"
WFINSTANCES_EPIGENOMICS_FIXED_K_RESULTS_DIR = WFINSTANCES_RESULTS_ROOT / "epigenomics_fixed_k"
LEGACY_WFINSTANCES_EPIGENOMICS_FIXED_K_RESULTS_DIR = NOTEBOOKS_ROOT / "wfinstances_epigenomics_results_fixed_k"
WFINSTANCES_MONTAGE_RESULTS_DIR = WFINSTANCES_RESULTS_ROOT / "montage"
WFINSTANCES_PLOTS_DIR = WFINSTANCES_RESULTS_ROOT / "plots"
LEGACY_WFINSTANCES_PLOTS_DIR = NOTEBOOKS_ROOT / "wfinstances_plots"

WFCOMMONS_RESULTS_ROOT = RESULTS_ROOT / "wfcommons"
WFCOMMONS_MONTAGE_RECOVERY_RESULTS_DIR = WFCOMMONS_RESULTS_ROOT / "montage_recovery"
LEGACY_WFCOMMONS_MONTAGE_RECOVERY_RESULTS_DIR = NOTEBOOKS_ROOT / "systematic_experiment_results_wfcommons"

WORKARENA_RESULTS_DIR = RESULTS_ROOT / "workarena_bhpop"


def prefer_existing_path(preferred: Path, legacy: Path | None = None) -> Path:
    """Return the preferred path when present, otherwise fall back to a legacy path if available."""
    if preferred.exists():
        return preferred
    if legacy is not None and legacy.exists():
        return legacy
    return preferred
