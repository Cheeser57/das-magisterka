"""
Multi-stage dilated Temporal Convolutional Network (MS-TCN-style) for
framewise DAS event segmentation.

See DESIGN.md section 3 for the full architecture rationale, sizing, and
receptive-field justification. Unlike the rest of the nn_segmentation
skeleton, this module is fully implemented: the model architecture is the
central deliverable of this design task, whereas the surrounding
data/training pipeline is left as a documented skeleton pending real data
and experimentation.

Shape convention throughout: ``(B, C, T)`` — batch, channels, time —
matching torch.nn.Conv1d's expected input layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nn_segmentation.config.model_tcn import TCNConfig


class TemporalBlock(nn.Module):
    """
    One dilated residual convolution unit.

    Conv1d(k=3, dilation=d) -> ReLU -> Dropout -> Conv1d(1x1) -> + residual.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class TCNStage(nn.Module):
    """
    A single MS-TCN stage: 1x1 input projection, a stack of dilated
    TemporalBlocks with dilation ``2**i``, and a 1x1 output projection to
    class logits.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_classes: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(hidden_channels, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(n_layers)
            ]
        )
        self.output_proj = nn.Conv1d(hidden_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, T)
            Raw features (stage 1) or the softmax output of the previous
            stage (refinement stages).

        Returns
        -------
        torch.Tensor, shape (B, n_classes, T)
        """
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        return self.output_proj(out)

    @staticmethod
    def receptive_field(n_layers: int, kernel_size: int = 3) -> int:
        """Receptive field in frames for a stack of ``n_layers`` dilated blocks."""
        return 1 + (kernel_size - 1) * (2**n_layers - 1)


class DASMultiStageTCN(nn.Module):
    """
    Full model: per-site input adapter -> temporal downsampling ->
    Stage 1 (prediction generation) -> N refinement stages.

    Site adapters are necessary because sites span different numbers of DAS
    channels (locations.csv distance windows / DX); all adapters project to
    the same ``config.adapter_channels`` width so a single shared backbone
    can train on pooled data from every site (DESIGN.md section 2.5).
    """

    def __init__(self, site_channel_counts: dict[str, int], config: TCNConfig) -> None:
        super().__init__()
        self.config = config
        self.site_adapters = nn.ModuleDict(
            {
                site_name: nn.Conv1d(n_channels, config.adapter_channels, kernel_size=1)
                for site_name, n_channels in site_channel_counts.items()
            }
        )
        self.downsample = nn.AvgPool1d(kernel_size=config.out_hop, stride=config.out_hop)

        self.stage1 = TCNStage(
            in_channels=config.adapter_channels,
            hidden_channels=config.hidden_channels,
            n_classes=config.n_classes,
            n_layers=config.n_layers_stage1,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.refine_stages = nn.ModuleList(
            [
                TCNStage(
                    in_channels=config.n_classes,
                    hidden_channels=config.hidden_channels,
                    n_classes=config.n_classes,
                    n_layers=config.n_layers_refine,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_refine_stages)
            ]
        )

    def forward(self, x: torch.Tensor, site_name: str) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C_site, T)
            Normalized, boundary-clean signal for one site
            (data/normalization.py, data/anomaly.py).
        site_name : str
            Selects which adapter to apply; must be a key of
            ``site_channel_counts`` passed to __init__.

        Returns
        -------
        list[torch.Tensor]
            Per-stage logits, each shape ``(B, n_classes, T // out_hop)``.
            ``outputs[-1]`` is the final refined prediction; every stage is
            supervised during training (training/losses.py), matching
            MS-TCN's multi-stage supervision.
        """
        if site_name not in self.site_adapters:
            raise KeyError(
                f"No adapter for site {site_name!r}; known sites: "
                f"{sorted(self.site_adapters.keys())}"
            )
        h = self.site_adapters[site_name](x)
        h = self.downsample(h)

        outputs = [self.stage1(h)]
        for stage in self.refine_stages:
            outputs.append(stage(F.softmax(outputs[-1], dim=1)))
        return outputs

    def receptive_field_seconds(self, fs: float) -> float:
        """Stage-1 receptive field in real-world seconds, given the input sampling rate."""
        rf_reduced_frames = TCNStage.receptive_field(self.config.n_layers_stage1, self.config.kernel_size)
        return rf_reduced_frames * self.config.out_hop / fs
