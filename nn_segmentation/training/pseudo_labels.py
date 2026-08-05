"""
Shared confident-pseudo-label selection, used by both training/fixmatch.py
(weak-view predictions supervising the strong view) and
training/cotraining.py (one branch's predictions cross-supervising the
other) — DESIGN.md sections 4.1/4.2 deliberately reuse the same asymmetric
per-class threshold + margin gate + vehicle-only design in both.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_pseudo_labels(
    logits: torch.Tensor,
    tau_vehicle: float,
    tau_background: float,
    confidence_margin: float,
    vehicle_class_ids: set[int],
    background_class_id: int,
    pseudo_label_background: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn one model's predictions into (pseudo_label, trust_mask) pairs.

    A frame's pseudo-label is trusted only if:
      - its predicted class is a vehicle class (tram/bus/truck) with
        top-1 probability >= tau_vehicle and a decisiveness margin
        (top1 - top2) >= confidence_margin, matching co_training.ipynb's
        CT_CONF_VEH/CT_CONF_MARGIN; or
      - (only if pseudo_label_background=True) its predicted class is
        background, gated by the much stricter tau_background — off by
        default, since background is trivially easy/dominant early in
        training and co_training.ipynb's own vehicle-only pseudo-labeling
        found background pseudo-labels add no useful signal (DESIGN.md
        section 4.1).
    "traffic" (neither vehicle nor background) is never pseudo-labeled —
    it is an ill-defined aggregate class (DESIGN.md section 3).

    Parameters
    ----------
    logits : torch.Tensor, shape (B, n_classes, T)

    Returns
    -------
    pseudo_label : torch.Tensor, shape (B, T), int64
    trust_mask : torch.Tensor, shape (B, T), bool
    """
    probs = F.softmax(logits, dim=1)  # (B, n_classes, T)
    top2 = probs.topk(2, dim=1).values  # (B, 2, T)
    top1_prob = top2[:, 0, :]
    margin = top2[:, 0, :] - top2[:, 1, :]
    pseudo_label = probs.argmax(dim=1)  # (B, T)

    is_vehicle = torch.zeros_like(pseudo_label, dtype=torch.bool)
    for cls_id in vehicle_class_ids:
        is_vehicle |= pseudo_label == cls_id
    is_background = pseudo_label == background_class_id

    decisive = margin >= confidence_margin
    trust_mask = is_vehicle & (top1_prob >= tau_vehicle) & decisive
    if pseudo_label_background:
        trust_mask = trust_mask | (is_background & (top1_prob >= tau_background) & decisive)

    return pseudo_label, trust_mask


def cap_trust_mask(
    trust_mask: torch.Tensor,
    confidence: torch.Tensor,
    max_fraction: float,
) -> torch.Tensor:
    """
    If more than ``max_fraction`` of all frames are trusted, keep only the
    highest-confidence ones up to that cap — continuous analogue of
    co_training.ipynb's CT_MAX_VEH_PER_ITER, guarding against
    confirmation-bias runaway from flooding the unsupervised loss early in
    training (DESIGN.md section 4.1).

    Parameters
    ----------
    trust_mask : torch.Tensor, shape (B, T), bool
    confidence : torch.Tensor, shape (B, T)
        Per-frame top-1 probability (or any monotonic confidence score).
    """
    max_allowed = int(trust_mask.numel() * max_fraction)
    if max_allowed <= 0:
        return torch.zeros_like(trust_mask)

    n_trusted = int(trust_mask.sum().item())
    if n_trusted <= max_allowed:
        return trust_mask

    masked_confidence = torch.where(trust_mask, confidence, torch.full_like(confidence, -1.0))
    cutoff = torch.topk(masked_confidence.flatten(), max_allowed).values.min()
    return trust_mask & (confidence >= cutoff)
