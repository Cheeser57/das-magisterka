"""
Hyperparameters for the multi-stage dilated Temporal Convolutional Network.

Defaults and their justification are documented in DESIGN.md section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

from nn_segmentation.config.base import FS


@dataclass
class TCNConfig:
    adapter_channels: int = 64  # per-site Conv1d(C_site -> adapter_channels, k=1)
    hidden_channels: int = 64  # width inside every TemporalBlock

    kernel_size: int = 3
    dropout: float = 0.3

    # Temporal downsampling applied right after the site adapter.
    # ~17 samples @ FS=67 Hz => ~0.25s / ~4 predictions per second.
    # See DESIGN.md section 3 for why per-sample output resolution is not
    # useful given Label Studio's pixel-resolution time boundaries.
    out_hop: int = 17

    # Stage 1 ("prediction generation"): L1 dilated residual layers,
    # dilation 2**i for i in range(n_layers_stage1). RF = 1 + 2*(2**L - 1)
    # reduced-resolution frames ~= 127 frames ~= 31.8s of real-world context.
    n_layers_stage1: int = 6

    # Refinement stages: each consumes softmax(prev stage) instead of raw
    # features again (mirrors MS-TCN's SingleStageModel refinement design).
    n_layers_refine: int = 4
    n_refine_stages: int = 3

    n_classes: int = 5  # background, tram, bus, truck, traffic

    @property
    def out_fs(self) -> float:
        """Effective output sampling rate (Hz) after out_hop downsampling."""
        return FS / self.out_hop
