"""
CLI entrypoint for DAS temporal-CNN segmentation.

Usage
-----
    python -m nn_segmentation.main build-cache --data_path cache/nn_segmentation --site most_mieszka
    python -m nn_segmentation.main supervised  --data_path cache/nn_segmentation --epochs 100 --batch_size 16
    python -m nn_segmentation.main fixmatch    --data_path cache/nn_segmentation --epochs 100
    python -m nn_segmentation.main cotrain     --data_path cache/nn_segmentation --epochs 100
    python -m nn_segmentation.main predict     --data_path cache/nn_segmentation --run_name <run> --site pcss

See DESIGN.md for the full architecture/training design this CLI drives.
Subcommand bodies are thin dispatches into training/*.py. `supervised`,
`fixmatch`, and `cotrain` are all fully implemented; `predict` remains a
skeleton pending a later implementation pass.
"""

import argparse

from nn_segmentation.config.base import BaseConfig
from nn_segmentation.config.cotraining import CoTrainingConfig
from nn_segmentation.config.fixmatch import FixMatchConfig
from nn_segmentation.config.supervised import SupervisedConfig
from nn_segmentation.utils.seeding import set_seed


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_path", type=str, required=True, help="Path to cache/nn_segmentation root")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--run_name", type=str, default="default_run", help="Run name for checkpoints/logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument(
        "--wandb_mode", type=str, default="online", choices=["online", "offline"],
        help="'offline' (default) writes local run files without requiring login",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAS Temporal CNN Segmentation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build_cache = subparsers.add_parser("build-cache", help="Build per-site preprocessed caches")
    p_build_cache.add_argument("--data_path", type=str, required=True, help="Output cache root")
    p_build_cache.add_argument("--site", type=str, default=None, help="Site name; omit to build all sites")
    p_build_cache.add_argument("--force_rebuild", action="store_true")

    p_supervised = subparsers.add_parser("supervised", help="Plain supervised training")
    _add_common_args(p_supervised)

    p_fixmatch = subparsers.add_parser("fixmatch", help="FixMatch semi-supervised training (primary)")
    _add_common_args(p_fixmatch)

    p_cotrain = subparsers.add_parser("cotrain", help="Cross-Pseudo-Supervision co-training (alternative)")
    _add_common_args(p_cotrain)

    p_predict = subparsers.add_parser("predict", help="Run a trained model over a full recording")
    p_predict.add_argument("--data_path", type=str, required=True)
    p_predict.add_argument("--run_name", type=str, required=True)
    p_predict.add_argument("--site", type=str, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(getattr(args, "seed", 42))

    base_config = BaseConfig(cache_dir=args.data_path)

    if args.command == "build-cache":
        from nn_segmentation.data.cache_builder import build_site_cache
        from nn_segmentation.data.raster_labels import load_site_configs

        sites = load_site_configs(base_config)
        if args.site:
            sites = [s for s in sites if s.name == args.site]
        for site in sites:
            build_site_cache(site, base_config, force_rebuild=args.force_rebuild)

    elif args.command == "supervised":
        from nn_segmentation.training.supervised import train_supervised

        train_config = SupervisedConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            use_wandb=args.use_wandb, wandb_mode=args.wandb_mode,
        )
        train_supervised(args.run_name, base_config, train_config)

    elif args.command == "fixmatch":
        from nn_segmentation.training.fixmatch import train_fixmatch

        fixmatch_config = FixMatchConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            use_wandb=args.use_wandb, wandb_mode=args.wandb_mode,
        )
        train_fixmatch(args.run_name, base_config, fixmatch_config)

    elif args.command == "cotrain":
        from nn_segmentation.training.cotraining import train_cross_pseudo_supervision

        cotraining_config = CoTrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            use_wandb=args.use_wandb, wandb_mode=args.wandb_mode,
        )
        train_cross_pseudo_supervision(args.run_name, base_config, cotraining_config)

    elif args.command == "predict":
        raise NotImplementedError("predict subcommand: run DASMultiStageTCN over a full site recording")

    else:  # pragma: no cover - argparse enforces valid choices
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
