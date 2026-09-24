"""OOD gates and threshold grids for offline sweeps."""

from __future__ import annotations

from .models import RouterFeatures


OOD_THRESHOLDS = (0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def passes_ood_gate(features: RouterFeatures, threshold: float | None) -> bool:
    return threshold is None or features.top1_similarity >= threshold
