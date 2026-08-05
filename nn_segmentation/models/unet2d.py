"""
Optional 2D (time x distance) U-Net ablation — documented stretch goal, NOT
part of the primary design.

DESIGN.md section 3: since every label spans the site's full distance
window (no per-channel ground truth exists anywhere in the repo), a 2D
U-Net's target would just be the same time-only label broadcast as a
row-constant image. It can only meaningfully outperform the 1D TCN
(models/tcn.py) if combined with trajectory-line pseudo-masks
(DESIGN.md section 1's documented future option, from trajectory_detection.
ipynb's Hough/LSD/RANSAC line fits) as weak per-pixel targets. Build this
only if that stretch goal is actually pursued.
"""

from __future__ import annotations

from torch import nn

from nn_segmentation.config.model_tcn import TCNConfig


class DASUNet2D(nn.Module):
    """
    Placeholder for a time x distance U-Net over DAS waterfall patches.

    Not implemented: requires per-channel supervision (e.g. trajectory
    pseudo-masks) to justify the added complexity over DASMultiStageTCN
    (models/tcn.py). See module docstring and DESIGN.md section 3.
    """

    def __init__(self, config: TCNConfig) -> None:
        super().__init__()
        raise NotImplementedError(
            "2D U-Net ablation is a documented future option (DESIGN.md section 3), "
            "contingent on building trajectory-line pseudo-masks first."
        )
