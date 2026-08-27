from __future__ import annotations

from pathlib import Path


CANONICAL_CLOUD_IAC_RELATIVE_PATH = Path("data") / "cloud_iac_dataset"
CANONICAL_CLOUD_IAC_DIRNAME = "cloud_iac_dataset"
LEGACY_CLOUD_IAC_DIRNAME = "aliyun_data"
GROUND_TRUTH_DIRNAMES = ("ground_truth", "manual_scenarios")


def resolve_cloud_iac_data_root(project_root: Path) -> Path:
    """Return the canonical Cloud-IaC dataset root, falling back to the legacy name."""
    candidates = [
        project_root / CANONICAL_CLOUD_IAC_RELATIVE_PATH,
        project_root / CANONICAL_CLOUD_IAC_DIRNAME,
        project_root / LEGACY_CLOUD_IAC_DIRNAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cloud-IaC dataset directory not found. Checked: {joined}")


def resolve_cloud_iac_ground_truth_dir(data_root: Path) -> Path:
    """Support both the public `ground_truth/` name and the legacy `manual_scenarios/` name."""
    for dirname in GROUND_TRUTH_DIRNAMES:
        candidate = data_root / dirname
        if candidate.exists():
            return candidate
    checked = ", ".join(str(data_root / dirname) for dirname in GROUND_TRUTH_DIRNAMES)
    raise FileNotFoundError(f"Cloud-IaC ground-truth directory not found. Checked: {checked}")
