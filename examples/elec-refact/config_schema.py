from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, TypeVar


OptimizerName = Literal["adam", "adamw", "muon"]
ForceStrategy = Literal["all", "none", "every_n", "subset"]
ValidationForceMode = Literal["all", "energy_only", "match_train"]


@dataclass
class ExperimentSettings:
    name: str
    output_root: str
    output_dir: str | None = None


@dataclass
class ModelSettings:
    type: str = "mattersim"
    name: str = "MatterSim-v1.0.0-1M"
    task_name: str = "omol"
    force_mode: str = "conservative"
    graph_radius: float = 6.0
    max_num_neighbors: int = 120
    orb_edge_method: str | None = None
    reset_output_heads: bool = False
    freeze_backbone: bool = False


@dataclass
class DataSettings:
    train_file: str | list[str]
    energy_reference: str
    pair_train_file: str | None = None
    test_file: str | None = None
    train_split: float = 0.9
    shuffle: bool = True
    shuffle_seed: int = 42
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    max_parent_frame: int | None = None


@dataclass
class ReferenceSettings:
    auto_generate: bool = True
    refit: bool = False
    method: Literal["residual"] = "residual"
    baseline: Literal["ase_pretrained"] = "ase_pretrained"
    regression: Literal["ridge", "linear"] = "ridge"
    ridge_alpha: float = 1.0
    device: str = "cuda:0"


@dataclass
class ObjectiveSettings:
    energy_weight: float = 200.0
    force_weight: float = 20.0
    delta_energy_weight: float = 0.0
    force_training_strategy: ForceStrategy = "all"
    force_every_n_steps: int = 1
    force_subset_key: str = "force_train_mask"
    validation_force_mode: ValidationForceMode = "all"
    per_atom_energy_normalize: bool = True


@dataclass
class OptimizerSettings:
    name: OptimizerName = "adamw"
    lr: float = 8e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    amsgrad: bool = False
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_aux_lr: float | None = None
    muon_aux_weight_decay: float | None = None
    muon_exclude_patterns: list[str] | None = None


@dataclass
class SchedulerSettings:
    monitor: str = "val/total_loss"
    mode: str = "min"
    factor: float = 0.8
    patience: int = 5
    min_lr: float = 1e-8


@dataclass
class TrainerSettings:
    accelerator: str = "gpu"
    devices: list[int] = field(default_factory=lambda: [0])
    strategy: str | None = None
    precision: str = "32"
    max_epochs: int = 5000
    gradient_clip_val: float = 2.0
    ema_decay: float = 0.99
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 1e-5
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None


@dataclass
class CheckpointSettings:
    init_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    save_last: bool = True
    save_top_k: int = 1


@dataclass
class LoggingSettings:
    backend: Literal["wandb", "csv"] = "wandb"
    project: str = "MatterTune-Electrolyte"
    run_name: str | None = None
    offline: bool = False
    log_loss_grad_norms: bool = True
    grad_norm_log_every_n_steps: int = 50


@dataclass
class EvaluationSettings:
    skip: bool = True
    device: str = ""
    seed: int = 42
    max_structures: int | None = None
    max_force_plot_points: int = 200_000


T = TypeVar("T")


def _build_section(section_type: type[T], value: Any, section_name: str) -> T:
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section {section_name!r} must be a mapping.")
    allowed = {item.name for item in fields(section_type)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys in configuration section {section_name!r}: {unknown}"
        )
    return section_type(**value)


@dataclass
class TrainingSettings:
    experiment: ExperimentSettings
    model: ModelSettings
    data: DataSettings
    reference: ReferenceSettings
    objective: ObjectiveSettings
    optimizer: OptimizerSettings
    scheduler: SchedulerSettings
    trainer: TrainerSettings
    checkpoint: CheckpointSettings
    logging: LoggingSettings
    evaluation: EvaluationSettings
    variables: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingSettings:
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown top-level configuration sections: {unknown}")

        required = allowed - {"variables"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Missing configuration sections: {missing}")

        settings = cls(
            experiment=_build_section(
                ExperimentSettings, value["experiment"], "experiment"
            ),
            model=_build_section(ModelSettings, value["model"], "model"),
            data=_build_section(DataSettings, value["data"], "data"),
            reference=_build_section(
                ReferenceSettings, value["reference"], "reference"
            ),
            objective=_build_section(
                ObjectiveSettings, value["objective"], "objective"
            ),
            optimizer=_build_section(
                OptimizerSettings, value["optimizer"], "optimizer"
            ),
            scheduler=_build_section(
                SchedulerSettings, value["scheduler"], "scheduler"
            ),
            trainer=_build_section(TrainerSettings, value["trainer"], "trainer"),
            checkpoint=_build_section(
                CheckpointSettings, value["checkpoint"], "checkpoint"
            ),
            logging=_build_section(LoggingSettings, value["logging"], "logging"),
            evaluation=_build_section(
                EvaluationSettings, value["evaluation"], "evaluation"
            ),
            variables=dict(value.get("variables", {})),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.optimizer.name not in {"adam", "adamw", "muon"}:
            raise ValueError(f"Unsupported optimizer: {self.optimizer.name}")
        if self.objective.force_training_strategy not in {
            "all",
            "none",
            "every_n",
            "subset",
        }:
            raise ValueError(
                "Unsupported objective.force_training_strategy: "
                f"{self.objective.force_training_strategy}"
            )
        if self.objective.validation_force_mode not in {
            "all",
            "energy_only",
            "match_train",
        }:
            raise ValueError(
                "Unsupported objective.validation_force_mode: "
                f"{self.objective.validation_force_mode}"
            )
        if self.scheduler.mode not in {"min", "max"}:
            raise ValueError("scheduler.mode must be 'min' or 'max'.")
        if self.logging.backend not in {"wandb", "csv"}:
            raise ValueError("logging.backend must be 'wandb' or 'csv'.")
        if self.optimizer.name == "muon" and self.model.type != "mattersim":
            raise ValueError("Muon is currently validated only with MatterSim.")
        if self.checkpoint.init_checkpoint and self.checkpoint.resume_checkpoint:
            raise ValueError(
                "checkpoint.init_checkpoint and checkpoint.resume_checkpoint "
                "are mutually exclusive."
            )
        if not 0.0 < self.data.train_split < 1.0:
            raise ValueError("data.train_split must be between zero and one.")
        if self.data.batch_size < 1:
            raise ValueError("data.batch_size must be positive.")
        if isinstance(self.data.train_file, list) and not self.data.train_file:
            raise ValueError("data.train_file must contain at least one path.")
        if self.reference.ridge_alpha < 0.0:
            raise ValueError("reference.ridge_alpha must be non-negative.")
        if self.trainer.max_epochs < 1:
            raise ValueError("trainer.max_epochs must be positive.")
        if not self.trainer.devices:
            raise ValueError("trainer.devices must not be empty.")
        if self.optimizer.lr <= 0.0:
            raise ValueError("optimizer.lr must be positive.")
        if self.objective.force_every_n_steps < 1:
            raise ValueError("objective.force_every_n_steps must be positive.")
