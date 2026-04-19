from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _strip_backbone_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefix = "backbone.model."
    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        stripped[key.removeprefix(prefix)] = value
    if not stripped:
        raise ValueError(
            "Could not find any `backbone.model.*` weights in the MatterTune checkpoint."
        )
    return stripped


def _filter_message_passing_layers(
    model_state: dict[str, Any],
    *,
    num_blocks: int | None,
) -> dict[str, Any]:
    if num_blocks is None:
        return dict(model_state)

    filtered: dict[str, Any] = {}
    graph_conv_prefix = "graph_conv."
    for key, value in model_state.items():
        if key.startswith(graph_conv_prefix):
            remainder = key.removeprefix(graph_conv_prefix)
            layer_idx_str, _, _ = remainder.partition(".")
            try:
                layer_idx = int(layer_idx_str)
            except ValueError:
                filtered[key] = value
                continue
            if layer_idx >= num_blocks:
                continue
        filtered[key] = value
    return filtered


def export_mattertune_mattersim_checkpoint(
    mattertune_checkpoint: str | Path,
    *,
    base_native_checkpoint: str | Path,
    output_checkpoint: str | Path,
) -> dict[str, Any]:
    """
    Convert a MatterTune MatterSim checkpoint into a native MatterSim checkpoint.

    The exported checkpoint can be loaded directly by MatterSim's
    `Potential.from_checkpoint(...)` or `MatterSimCalculator(load_path=...)`.
    """

    mattertune_checkpoint = Path(mattertune_checkpoint)
    base_native_checkpoint = Path(base_native_checkpoint)
    output_checkpoint = Path(output_checkpoint)

    mattertune_payload = torch.load(
        mattertune_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    hparams = mattertune_payload.get("hyper_parameters", {})
    checkpoint_name = hparams.get("name")
    if checkpoint_name != "mattersim":
        raise ValueError(
            f"Expected MatterTune checkpoint with backbone `mattersim`, got `{checkpoint_name}`."
        )

    native_payload = torch.load(base_native_checkpoint, map_location="cpu")
    if native_payload.get("model_name") != "m3gnet":
        raise ValueError(
            f"Expected native MatterSim checkpoint with `model_name='m3gnet'`, got `{native_payload.get('model_name')}`."
        )

    pruning_message_passing = hparams.get("pruning_message_passing")
    stripped_state = _strip_backbone_prefix(mattertune_payload["state_dict"])
    exported_state = _filter_message_passing_layers(
        stripped_state,
        num_blocks=pruning_message_passing,
    )

    model_args = dict(native_payload["model_args"])
    if pruning_message_passing is not None:
        model_args["num_blocks"] = int(pruning_message_passing)

    exported_payload = {
        "model_name": native_payload["model_name"],
        "model": exported_state,
        "model_args": model_args,
        "description": (
            "Exported from MatterTune MatterSim checkpoint "
            f"`{mattertune_checkpoint.name}`."
        ),
        "source": {
            "mattertune_checkpoint": str(mattertune_checkpoint.resolve()),
            "base_native_checkpoint": str(base_native_checkpoint.resolve()),
            "pruning_message_passing": pruning_message_passing,
        },
    }

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported_payload, output_checkpoint)
    return {
        "output_checkpoint": str(output_checkpoint.resolve()),
        "num_exported_tensors": len(exported_state),
        "num_blocks": model_args["num_blocks"],
        "pruning_message_passing": pruning_message_passing,
    }
