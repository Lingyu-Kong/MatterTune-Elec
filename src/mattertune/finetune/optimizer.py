from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable, Sequence
from typing import Annotated, Any, Literal, cast

import nshconfig as C
import torch
import torch.nn as nn
from typing_extensions import NotRequired, TypeAliasType, TypedDict, assert_never


log = logging.getLogger(__name__)


class PerParamHparamsDict(TypedDict):
    patterns: Sequence[str]
    """Patterns to match parameter names."""

    hparams: dict[str, Any]
    """Hyperparameters for the matched parameters."""

    optimize: NotRequired[bool]
    """Whether to optimize this parameter. Default is True."""


class OptimizerConfigBase(C.Config):
    per_parameter_hparams: Sequence[PerParamHparamsDict] | None = None
    """Per parameter hyperparameters.

    This should be a list of dictionaries, each of which has the following keys:

    - `patterns`: a list of patterns to match parameter names.
    - `hparams`: a dictionary of hyperparameters for the matched parameters.
    - `optimize`: whether to optimize this parameter. Default is True.

    This allows you to, for example, set different learning rates for
    different parameters."""


class AdamConfig(OptimizerConfigBase):
    name: Literal["Adam"] = "Adam"
    """name of the optimizer."""
    lr: C.PositiveFloat
    """Learning rate."""
    eps: C.NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[C.PositiveFloat, C.PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: C.NonNegativeFloat = 0.0
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""


class AdamWConfig(OptimizerConfigBase):
    name: Literal["AdamW"] = "AdamW"
    """name of the optimizer."""
    lr: C.PositiveFloat
    """Learning rate."""
    eps: C.NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[C.PositiveFloat, C.PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: C.NonNegativeFloat = 0.01
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""


class MuonConfig(OptimizerConfigBase):
    name: Literal["Muon"] = "Muon"
    """Name of the optimizer."""
    lr: C.PositiveFloat
    """Learning rate for hidden matrix parameters optimized by Muon."""
    weight_decay: C.NonNegativeFloat = 0.1
    """Decoupled weight decay for Muon parameters."""
    momentum: C.NonNegativeFloat = 0.95
    """Momentum factor for Muon."""
    nesterov: bool = True
    """Whether to use Nesterov momentum in Muon."""
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    """Newton-Schulz polynomial coefficients."""
    eps: C.PositiveFloat = 1e-7
    """Numerical stability epsilon for Newton-Schulz orthogonalization."""
    ns_steps: C.PositiveInt = 5
    """Number of Newton-Schulz iterations."""
    adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = "match_rms_adamw"
    """Shape-dependent Muon learning-rate adjustment."""
    min_matrix_dim: C.PositiveInt = 2
    """Minimum size of both matrix dimensions for a parameter to use Muon."""
    exclude_patterns: Sequence[str] = (
        "*embed*",
        "*lm_head*",
        "*output_head*",
    )
    """Parameter-name patterns that must use the auxiliary AdamW optimizer."""
    adamw_lr: C.PositiveFloat | None = None
    """Auxiliary AdamW learning rate. Defaults to ``lr``."""
    adamw_betas: tuple[C.PositiveFloat, C.PositiveFloat] = (0.9, 0.95)
    """Auxiliary AdamW beta coefficients."""
    adamw_eps: C.PositiveFloat = 1e-8
    """Auxiliary AdamW epsilon."""
    adamw_weight_decay: C.NonNegativeFloat | None = None
    """Auxiliary AdamW weight decay. Defaults to ``weight_decay``."""
    adamw_amsgrad: bool = False
    """Whether the auxiliary AdamW optimizer uses AMSGrad."""


class SGDConfig(OptimizerConfigBase):
    name: Literal["SGD"] = "SGD"
    """name of the optimizer."""
    lr: C.PositiveFloat
    """Learning rate."""
    momentum: C.NonNegativeFloat = 0.0
    """Momentum."""
    weight_decay: C.NonNegativeFloat = 0.0
    """Weight decay."""
    nestrov: bool = False
    """Whether to use nestrov."""


OptimizerConfig = TypeAliasType(
    "OptimizerConfig",
    Annotated[
        AdamConfig | AdamWConfig | MuonConfig | SGDConfig,
        C.Field(discriminator="name"),
    ],
)


class _CompositeOptimizer(torch.optim.Optimizer):
    """Expose multiple disjoint optimizers as one optimizer to Lightning."""

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("At least one optimizer is required.")

        self.optimizers = tuple(optimizers)
        all_parameters = [
            parameter
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
            raise ValueError("Composite optimizer parameter groups must be disjoint.")

        # Initialize Optimizer's hook machinery, then expose the child parameter
        # group dictionaries themselves so schedulers update the child optimizers.
        super().__init__(all_parameters, defaults={})
        self._refresh_param_groups()

    def _refresh_param_groups(self) -> None:
        """Rebind the wrapper to the parameter groups owned by its children."""
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Lightning's closure performs forward, backward, and gradient setup. It
        # must run exactly once before either child optimizer consumes gradients.
        loss = self.optimizers[0].step(closure)
        for optimizer in self.optimizers[1:]:
            optimizer.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "composite_optimizer_version": 1,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers")
        if not isinstance(optimizer_states, list):
            raise ValueError("Invalid composite optimizer state dict.")
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "Composite optimizer state has a different number of child "
                f"optimizers: expected {len(self.optimizers)}, got "
                f"{len(optimizer_states)}."
            )
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)

        # Optimizer.load_state_dict() replaces the child's param-group dicts.
        # Rebind the wrapper so schedulers keep updating the groups that the
        # child optimizers actually consume after checkpoint restoration.
        self._refresh_param_groups()


def _parameter_uses_muon(
    name: str,
    parameter: nn.Parameter,
    config: MuonConfig,
) -> bool:
    if parameter.ndim != 2 or min(parameter.shape) < config.min_matrix_dim:
        return False
    return not any(
        fnmatch.fnmatch(name, pattern) for pattern in config.exclude_patterns
    )


def _create_muon_optimizer(
    config: MuonConfig,
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> torch.optim.Optimizer:
    if config.per_parameter_hparams is not None:
        raise ValueError(
            "MuonConfig does not support per_parameter_hparams; use "
            "exclude_patterns and the Muon/AdamW-specific fields instead."
        )

    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError(
            "Muon requires a PyTorch build that provides torch.optim.Muon "
            "(PyTorch 2.9 or newer)."
        )

    muon_parameters: list[nn.Parameter] = []
    adamw_parameters: list[nn.Parameter] = []
    for name, parameter in named_parameters:
        if _parameter_uses_muon(name, parameter, config):
            muon_parameters.append(parameter)
        else:
            adamw_parameters.append(parameter)

    if not muon_parameters:
        raise ValueError(
            "No eligible 2D hidden matrix parameters were found for Muon. "
            "Check min_matrix_dim and exclude_patterns."
        )

    muon_optimizer = muon_cls(
        muon_parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
        nesterov=config.nesterov,
        ns_coefficients=config.ns_coefficients,
        eps=config.eps,
        ns_steps=config.ns_steps,
        adjust_lr_fn=config.adjust_lr_fn,
    )
    for group in muon_optimizer.param_groups:
        group["optimizer_name"] = "muon"

    optimizers: list[torch.optim.Optimizer] = [
        cast(torch.optim.Optimizer, muon_optimizer)
    ]
    if adamw_parameters:
        adamw_optimizer = torch.optim.AdamW(
            adamw_parameters,
            lr=config.adamw_lr if config.adamw_lr is not None else config.lr,
            betas=config.adamw_betas,
            eps=config.adamw_eps,
            weight_decay=(
                config.adamw_weight_decay
                if config.adamw_weight_decay is not None
                else config.weight_decay
            ),
            amsgrad=config.adamw_amsgrad,
        )
        for group in adamw_optimizer.param_groups:
            group["optimizer_name"] = "adamw"
        optimizers.append(adamw_optimizer)

    muon_numel = sum(parameter.numel() for parameter in muon_parameters)
    adamw_numel = sum(parameter.numel() for parameter in adamw_parameters)
    log.info(
        "Muon parameter split: %d tensors / %d parameters use Muon; "
        "%d tensors / %d parameters use AdamW.",
        len(muon_parameters),
        muon_numel,
        len(adamw_parameters),
        adamw_numel,
    )
    return _CompositeOptimizer(optimizers)


def _named_parameters_matching_patterns(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    patterns: Iterable[str],
):
    for name, param in named_parameters:
        if (
            matching_pattern := next(
                (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                None,
            )
        ) is None:
            continue

        yield name, param, matching_pattern


def _split_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    pattern_lists: Iterable[Iterable[str]],
):
    named_parameters_list = list(named_parameters)
    all_parameters = [p for _, p in named_parameters_list]

    parameters: list[list[torch.nn.Parameter]] = []
    for patterns in pattern_lists:
        matching = [
            p
            for _, p, _ in _named_parameters_matching_patterns(
                named_parameters_list, patterns
            )
        ]
        parameters.append(matching)

        # Remove matching parameters from all_parameters
        all_parameters = [
            p for p in all_parameters if all(p is not m for m in matching)
        ]

    return parameters, all_parameters


def create_optimizer(
    config: OptimizerConfig,
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> torch.optim.Optimizer:
    default_kwargs: dict[str, Any]
    match config:
        case AdamConfig():
            default_kwargs = dict(
                lr=config.lr,
                eps=config.eps,
                betas=config.betas,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
            )
            cls = torch.optim.Adam
        case AdamWConfig():
            default_kwargs = dict(
                lr=config.lr,
                eps=config.eps,
                betas=config.betas,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
            )
            cls = torch.optim.AdamW
        case MuonConfig():
            return _create_muon_optimizer(config, named_parameters)
        case SGDConfig():
            default_kwargs = dict(
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nestrov,
            )
            cls = torch.optim.SGD
        case _:
            assert_never(config)

    # If per_parameter_hparams is not specified, return the optimizer
    if config.per_parameter_hparams is None:
        return cls((p for _, p in named_parameters), **default_kwargs)

    # Otherwise, split parameters
    parameters, all_parameters = _split_parameters(
        named_parameters, [d["patterns"] for d in config.per_parameter_hparams]
    )

    params_list: list[dict[str, Any]] = []
    for p, d in zip(parameters, config.per_parameter_hparams):
        if not d.get("optimize", True):
            continue

        param_dict = {}
        param_dict.update(default_kwargs)
        param_dict.update(d["hparams"])
        param_dict["params"] = p
        params_list.append(param_dict)

    if all_parameters:
        params_list.append({"params": all_parameters, **default_kwargs})

    return cls(params_list, **default_kwargs)
