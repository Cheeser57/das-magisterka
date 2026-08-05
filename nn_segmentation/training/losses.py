"""
Loss functions for multi-stage TCN training.

DESIGN.md section 3 "Loss": per-stage masked weighted cross-entropy summed
across all stages (standard MS-TCN multi-stage supervision), plus a T-MSE
smoothing term to discourage single-frame flicker. These are pure functions
(no dataset/optimizer dependency) so, unlike the training loops in this
package, they are fully implemented here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_weighted_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    labeled_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-entropy over timesteps where ``labeled_mask`` is True, ignoring
    unlabeled/unconfirmed timesteps entirely (DESIGN.md section 2.2's
    per-timestep labeled_mask, not just a chunk-level flag).

    Parameters
    ----------
    logits : torch.Tensor, shape (B, n_classes, T)
    target : torch.Tensor, shape (B, T), int64 class ids
    labeled_mask : torch.Tensor, shape (B, T), bool
    class_weights : torch.Tensor, shape (n_classes,), optional
        Inverse-frequency weights computed on the training range only.

    Returns
    -------
    torch.Tensor
        Scalar mean cross-entropy over labeled timesteps. Returns 0 (with
        grad) if no timestep in the batch is labeled.
    """
    if not labeled_mask.any():
        return logits.sum() * 0.0

    per_frame = F.cross_entropy(
        logits, target, weight=class_weights, reduction="none"
    )  # (B, T)
    return per_frame[labeled_mask].mean()


def pool_targets(
    label: torch.Tensor,
    labeled_mask: torch.Tensor,
    out_hop: int,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample per-sample label/labeled_mask from input resolution to the
    model's output resolution (the same AvgPool1d(out_hop) applied to the
    signal by DASMultiStageTCN — models/tcn.py).

    A pooled frame's label is the majority (mode) class among its out_hop
    input samples. A pooled frame counts as labeled only if every input
    sample in its window is labeled_mask=True (DESIGN.md section 3 "Loss":
    "a reduced frame counts as labeled only if fully covered").

    Parameters
    ----------
    label, labeled_mask : torch.Tensor, shape (B, T)
        Input-resolution targets, e.g. from data/dataset.py's
        SegmentationSample.

    Returns
    -------
    pooled_label : torch.Tensor, shape (B, T // out_hop)
    pooled_mask : torch.Tensor, shape (B, T // out_hop), bool
    """
    batch_size, n_samples = label.shape
    n_frames = n_samples // out_hop
    n_trunc = n_frames * out_hop

    label = label[:, :n_trunc].reshape(batch_size, n_frames, out_hop)
    mask = labeled_mask[:, :n_trunc].reshape(batch_size, n_frames, out_hop)

    onehot = F.one_hot(label, num_classes=n_classes)  # (B, n_frames, out_hop, n_classes)
    counts = onehot.sum(dim=2)  # (B, n_frames, n_classes)
    pooled_label = counts.argmax(dim=-1)
    pooled_mask = mask.all(dim=2)
    return pooled_label, pooled_mask


def tmse_loss(logits: torch.Tensor, clip: float = 4.0) -> torch.Tensor:
    """
    Truncated MSE between adjacent-frame log-probabilities (MS-TCN paper),
    discouraging flicker/over-segmentation in the framewise output.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, n_classes, T)
    clip : float
        Per-element squared-difference clip threshold.
    """
    log_probs = F.log_softmax(logits, dim=1)
    diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]
    diff_sq = torch.clamp(diff**2, max=clip**2)
    return diff_sq.mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    labeled_mask: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Optional alternative to masked_weighted_ce (DESIGN.md section 3's
    ablation note), for severe background-class dominance.
    """
    raise NotImplementedError


def multistage_loss(
    stage_logits: list[torch.Tensor],
    target: torch.Tensor,
    labeled_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    tmse_weight: float = 0.15,
    tmse_clip: float = 4.0,
) -> torch.Tensor:
    """
    Sum masked_weighted_ce + tmse_weight * tmse_loss across every stage's
    output (every stage is supervised against the same target — MS-TCN
    multi-stage supervision, DESIGN.md section 3).

    Parameters
    ----------
    stage_logits : list[torch.Tensor]
        DASMultiStageTCN.forward()'s return value.
    target, labeled_mask : see masked_weighted_ce. Must already be pooled to
        the model's output resolution (config.out_hop).
    """
    total = stage_logits[0].sum() * 0.0
    for logits in stage_logits:
        total = total + masked_weighted_ce(logits, target, labeled_mask, class_weights)
        total = total + tmse_weight * tmse_loss(logits, clip=tmse_clip)
    return total
