from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from argparse import Namespace
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from config_loader import load_training_settings
from config_schema import TrainingSettings


REFACT_ROOT = Path(__file__).resolve().parent
LEGACY_TRAINING_BACKEND = REFACT_ROOT.parent / "elec-Li-new" / "enhance-V1" / "train.py"


def load_training_backend() -> ModuleType:
    """Load the numerically validated electrolyte training implementation."""

    module_name = "mattertune_electrolyte_training_backend"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, LEGACY_TRAINING_BACKEND)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training backend: {LEGACY_TRAINING_BACKEND}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _path_or_none(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _safe_label(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "_.-" else "-"
        for character in value
    )


def _default_run_name(settings: TrainingSettings, training_mode: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model = _safe_label(settings.model.name.replace(".", "p"))
    return (
        f"{timestamp}-{_safe_label(settings.experiment.name)}-{model}-"
        f"{settings.optimizer.name}-{training_mode}"
    )


def build_training_args(
    settings: TrainingSettings,
    *,
    validate_files: bool = True,
) -> Namespace:
    backend = load_training_backend()
    args = Namespace(
        model_type=backend.normalize_model_type(settings.model.type),
        model_name=backend.normalize_model_name(
            settings.model.type, settings.model.name
        ),
        force_mode=backend.normalize_force_mode(settings.model.force_mode),
        task_name=settings.model.task_name,
        graph_radius=settings.model.graph_radius,
        max_num_neighbors=settings.model.max_num_neighbors,
        orb_edge_method=settings.model.orb_edge_method,
        reset_output_heads=settings.model.reset_output_heads,
        freeze_backbone=settings.model.freeze_backbone,
        train_file=Path(settings.data.train_file).expanduser(),
        pair_train_file=_path_or_none(settings.data.pair_train_file),
        test_file=_path_or_none(settings.data.test_file),
        energy_reference=Path(settings.data.energy_reference).expanduser(),
        train_split=settings.data.train_split,
        shuffle=settings.data.shuffle,
        shuffle_seed=settings.data.shuffle_seed,
        batch_size=settings.data.batch_size,
        num_workers=settings.data.num_workers,
        pin_memory=settings.data.pin_memory,
        max_parent_frame=settings.data.max_parent_frame,
        per_atom_energy_normalize=settings.objective.per_atom_energy_normalize,
        e_loss_weight=settings.objective.energy_weight,
        f_loss_weight=settings.objective.force_weight,
        delta_e_loss_weight=settings.objective.delta_energy_weight,
        force_training_strategy=settings.objective.force_training_strategy,
        force_every_n_steps=settings.objective.force_every_n_steps,
        force_subset_key=settings.objective.force_subset_key,
        validation_force_mode=settings.objective.validation_force_mode,
        optimizer=settings.optimizer.name,
        lr=settings.optimizer.lr,
        weight_decay=settings.optimizer.weight_decay,
        adam_betas=tuple(settings.optimizer.betas),
        adam_eps=settings.optimizer.eps,
        adam_amsgrad=settings.optimizer.amsgrad,
        muon_momentum=settings.optimizer.muon_momentum,
        muon_no_nesterov=not settings.optimizer.muon_nesterov,
        muon_ns_steps=settings.optimizer.muon_ns_steps,
        muon_adjust_lr_fn=settings.optimizer.muon_adjust_lr_fn,
        muon_aux_lr=settings.optimizer.muon_aux_lr,
        muon_aux_weight_decay=settings.optimizer.muon_aux_weight_decay,
        muon_exclude_pattern=settings.optimizer.muon_exclude_patterns,
        monitor=settings.scheduler.monitor,
        lr_mode=settings.scheduler.mode,
        lr_factor=settings.scheduler.factor,
        lr_patience=settings.scheduler.patience,
        min_lr=settings.scheduler.min_lr,
        accelerator=settings.trainer.accelerator,
        devices=backend.normalize_devices(settings.trainer.devices),
        strategy=settings.trainer.strategy,
        precision=settings.trainer.precision,
        max_epochs=settings.trainer.max_epochs,
        gradient_clip_val=settings.trainer.gradient_clip_val,
        ema_decay=settings.trainer.ema_decay,
        patience=settings.trainer.early_stopping_patience,
        early_stopping_min_delta=settings.trainer.early_stopping_min_delta,
        limit_train_batches=settings.trainer.limit_train_batches,
        limit_val_batches=settings.trainer.limit_val_batches,
        init_checkpoint=_path_or_none(settings.checkpoint.init_checkpoint),
        resume_checkpoint=_path_or_none(settings.checkpoint.resume_checkpoint),
        checkpoint_save_last=settings.checkpoint.save_last,
        checkpoint_save_top_k=settings.checkpoint.save_top_k,
        logger=settings.logging.backend,
        wandb_project=settings.logging.project,
        wandb_offline=settings.logging.offline,
        log_loss_grad_norms=settings.logging.log_loss_grad_norms,
        grad_norm_log_every_n_steps=settings.logging.grad_norm_log_every_n_steps,
        skip_eval=settings.evaluation.skip,
        eval_device=settings.evaluation.device,
        eval_seed=settings.evaluation.seed,
        max_eval_structures=settings.evaluation.max_structures,
        max_force_plot_points=settings.evaluation.max_force_plot_points,
    )

    backend.validate_supervision_args(args)
    run_name = settings.logging.run_name or _default_run_name(
        settings, args.training_mode
    )
    args.wandb_name = run_name
    if settings.experiment.output_dir:
        output_dir = Path(settings.experiment.output_dir).expanduser()
    else:
        output_dir = (
            Path(settings.experiment.output_root).expanduser()
            / args.training_mode
            / run_name
        )
    args.output_dir = output_dir
    args.checkpoint_dir = output_dir / "checkpoints"
    args.log_dir = output_dir / "logs"

    if validate_files:
        required: list[Path | None] = [args.train_file, args.energy_reference]
        if not args.skip_eval:
            required.append(args.test_file)
        if args.training_mode == backend.TRAIN_WITH_DELTA_E:
            required.append(args.pair_train_file)
        required.extend((args.init_checkpoint, args.resume_checkpoint))
        for path in required:
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        if (
            args.training_mode == backend.TRAIN_WITH_DELTA_E
            and args.pair_train_file is None
        ):
            raise ValueError("Delta-E training requires data.pair_train_file.")
        if not args.skip_eval and args.test_file is None:
            raise ValueError("Evaluation requires data.test_file.")
    return args


def _jsonable_args(args: Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resolved_snapshot(settings: TrainingSettings, args: Namespace) -> dict[str, Any]:
    return {
        "settings": asdict(settings),
        "derived": _jsonable_args(args),
    }


def run_training(settings: TrainingSettings, *, dry_run: bool = False) -> Namespace:
    args = build_training_args(settings, validate_files=not dry_run)
    snapshot = resolved_snapshot(settings, args)
    if dry_run:
        print(yaml.safe_dump(snapshot, sort_keys=False))
        return args

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output_dir / "resolved-config.yml"
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)

    print(json.dumps(_jsonable_args(args), indent=2, sort_keys=True))
    load_training_backend().main(args)
    return args


def run_from_cli(default_config: str | Path, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch electrolyte fine-tuning from composed YAML files."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML overlay. May be repeated; later files win.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted configuration key. May be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the configuration without checking files or training.",
    )
    cli = parser.parse_args(argv)
    settings = load_training_settings(
        [default_config, *cli.config],
        overrides=cli.overrides,
    )
    run_training(settings, dry_run=cli.dry_run)
