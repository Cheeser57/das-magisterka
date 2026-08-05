"""
Weights & Biases integration.

main.py's original argparse stub already imported wandb without using it —
this is the intended integration point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import wandb


def get_logger(name: str) -> logging.Logger:
    """Standard library logger, configured once with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def wandb_init(
    project: str,
    run_name: str,
    config: dict[str, Any],
    run_dir: str = "runs/nn_segmentation",
    mode: str = "offline",
):
    """
    Thin wrapper around ``wandb.init()``. Kept separate from
    training/loop_utils.py's init_wandb_run() so the W&B dependency stays
    isolated to this module.

    Parameters
    ----------
    mode : {"online", "offline"}
        "offline" (the default, see config/supervised.py) writes local run
        files under ``run_dir`` without requiring network access or a
        logged-in account; sync later with ``wandb sync <run dir>``.
    run_dir : str
        Local directory wandb writes its run files under
        (``{run_dir}/{run_name}/wandb/``) — DESIGN.md section 5's
        run-artifact location, kept separate from
        nn_segmentation/models/training/'s checkpoint-only directory.

    Returns
    -------
    wandb.sdk.wandb_run.Run
    """
    run_path = Path(run_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=project,
        name=run_name,
        config=config,
        dir=str(run_path),
        mode=mode,
        reinit="finish_previous",
    )
