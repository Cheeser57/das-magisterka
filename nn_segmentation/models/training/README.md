# Checkpoint output directory

This directory holds **model weight checkpoints only**
(`<run_name>/checkpoint_best.pt`, `<run_name>/checkpoint_last.pt`, optimizer
state) written by `training/loop_utils.py::save_checkpoint()`.

It does **not** contain source code. Training-loop source code lives at
`nn_segmentation/training/` (a distinct top-level folder) — see
`nn_segmentation/DESIGN.md` section 5 for why these two are deliberately
separated.

Contents of this directory are gitignored, mirroring the existing `cache/`
and `runs/` conventions elsewhere in the repo.
