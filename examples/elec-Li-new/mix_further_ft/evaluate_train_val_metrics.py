"""Evaluate a MatterTune checkpoint on its original train/validation split.

By default this script evaluates the bulk/interface checkpoint requested for the
Li-electrolyte-Mix run.  For another checkpoint, pass ``--ckpt``; the dataset
path and split settings are then read from that checkpoint's saved datamodule
hyperparameters unless explicitly overridden on the command line.

Energy errors are reported both per atom and per structure.  Force errors are
computed over all Cartesian force components, matching MatterTune's standard
MAE/RMSE convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.io import iread
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mattertune.main import load_finetuned_checkpoint  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix/local_runs/"
    "mix_further_ft/from_with_enhance/bulk_interface/train_without_delta_e/"
    "20260714-184039-mattersim-1m-MatterSim-v1p0p0-1M-conservative-"
    "train_without_delta_e-ew200p0-fw20p0-dew0/checkpoints/"
    "mattersim-MatterSim-v1.0.0-1M-conservative-train_without_delta_e-best.ckpt"
)

TRAIN = 1
VALIDATION = 2
SPLIT_NAMES = {TRAIN: "train", VALIDATION: "validation"}
DEFAULT_FORCE_PLOT_MAX_POINTS = 500_000
DEFAULT_PLOT_SEED = 20260803


@dataclass
class RunningMetrics:
    energy_abs_mev_per_atom: float = 0.0
    energy_sq_mev2_per_atom2: float = 0.0
    energy_abs_mev_per_structure: float = 0.0
    energy_sq_mev2_per_structure2: float = 0.0
    n_structures: int = 0
    forces_abs_mev_per_a: float = 0.0
    forces_sq_mev2_per_a2: float = 0.0
    n_force_components: int = 0

    def update(
        self,
        *,
        energy_error_ev: float,
        n_atoms: int,
        force_error_ev_per_a: np.ndarray,
    ) -> None:
        if n_atoms <= 0:
            raise ValueError(f"A structure has an invalid atom count: {n_atoms}")

        energy_error_mev_per_structure = energy_error_ev * 1000.0
        energy_error_mev_per_atom = energy_error_mev_per_structure / n_atoms
        self.energy_abs_mev_per_atom += abs(energy_error_mev_per_atom)
        self.energy_sq_mev2_per_atom2 += energy_error_mev_per_atom**2
        self.energy_abs_mev_per_structure += abs(energy_error_mev_per_structure)
        self.energy_sq_mev2_per_structure2 += energy_error_mev_per_structure**2
        self.n_structures += 1

        force_error_mev_per_a = np.asarray(
            force_error_ev_per_a, dtype=np.float64
        ).reshape(-1) * 1000.0
        self.forces_abs_mev_per_a += float(np.abs(force_error_mev_per_a).sum())
        self.forces_sq_mev2_per_a2 += float(
            np.square(force_error_mev_per_a).sum()
        )
        self.n_force_components += int(force_error_mev_per_a.size)

    def as_dict(self) -> dict[str, int | float]:
        if self.n_structures == 0:
            raise ValueError("Cannot compute metrics for an empty structure split.")
        if self.n_force_components == 0:
            raise ValueError("Cannot compute force metrics without force labels.")

        return {
            "n_structures": self.n_structures,
            "n_force_components": self.n_force_components,
            "energy_mae_meV_per_atom": (
                self.energy_abs_mev_per_atom / self.n_structures
            ),
            "energy_rmse_meV_per_atom": float(
                np.sqrt(self.energy_sq_mev2_per_atom2 / self.n_structures)
            ),
            "energy_mae_meV_per_structure": (
                self.energy_abs_mev_per_structure / self.n_structures
            ),
            "energy_rmse_meV_per_structure": float(
                np.sqrt(self.energy_sq_mev2_per_structure2 / self.n_structures)
            ),
            "forces_mae_meV_per_A": (
                self.forces_abs_mev_per_a / self.n_force_components
            ),
            "forces_rmse_meV_per_A": float(
                np.sqrt(self.forces_sq_mev2_per_a2 / self.n_force_components)
            ),
        }


class ForceReservoir:
    """Uniform priority sample of force components with chunked compaction."""

    def __init__(self, max_points: int, seed: int) -> None:
        if max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        self.max_points = max_points
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty(0, dtype=np.float64)
        self.ground_truth = np.empty(0, dtype=np.float32)
        self.prediction = np.empty(0, dtype=np.float32)
        self.pending_keys: list[np.ndarray] = []
        self.pending_ground_truth: list[np.ndarray] = []
        self.pending_prediction: list[np.ndarray] = []
        self.pending_count = 0
        self.n_seen = 0

    def update(self, ground_truth: np.ndarray, prediction: np.ndarray) -> None:
        ground_truth_flat = np.asarray(ground_truth, dtype=np.float32).reshape(-1)
        prediction_flat = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if ground_truth_flat.shape != prediction_flat.shape:
            raise ValueError(
                "Force sample shape mismatch: "
                f"{ground_truth_flat.shape} != {prediction_flat.shape}"
            )
        if ground_truth_flat.size == 0:
            return

        self.pending_keys.append(self.rng.random(ground_truth_flat.size))
        self.pending_ground_truth.append(ground_truth_flat)
        self.pending_prediction.append(prediction_flat)
        self.pending_count += int(ground_truth_flat.size)
        self.n_seen += int(ground_truth_flat.size)
        if self.pending_count >= self.max_points:
            self._compact()

    def _compact(self) -> None:
        if self.pending_count == 0:
            return
        keys = np.concatenate([self.keys, *self.pending_keys])
        ground_truth = np.concatenate(
            [self.ground_truth, *self.pending_ground_truth]
        )
        prediction = np.concatenate([self.prediction, *self.pending_prediction])
        if keys.size > self.max_points:
            keep = np.argpartition(keys, self.max_points - 1)[: self.max_points]
            keys = keys[keep]
            ground_truth = ground_truth[keep]
            prediction = prediction[keep]
        self.keys = keys
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.pending_keys.clear()
        self.pending_ground_truth.clear()
        self.pending_prediction.clear()
        self.pending_count = 0

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        self._compact()
        return self.ground_truth, self.prediction


class SplitParityData:
    def __init__(self, force_plot_max_points: int, seed: int) -> None:
        self.energy_ground_truth_ref_subtracted_ev_per_atom: list[float] = []
        self.energy_prediction_ref_subtracted_ev_per_atom: list[float] = []
        self.energy_ground_truth_ref_subtracted_ev_per_structure: list[float] = []
        self.energy_prediction_ref_subtracted_ev_per_structure: list[float] = []
        self.force_sample = ForceReservoir(force_plot_max_points, seed)

    def energy_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(
                self.energy_ground_truth_ref_subtracted_ev_per_atom,
                dtype=np.float64,
            ),
            np.asarray(
                self.energy_prediction_ref_subtracted_ev_per_atom,
                dtype=np.float64,
            ),
            np.asarray(
                self.energy_ground_truth_ref_subtracted_ev_per_structure,
                dtype=np.float64,
            ),
            np.asarray(
                self.energy_prediction_ref_subtracted_ev_per_structure,
                dtype=np.float64,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a MatterTune checkpoint, reconstruct its original train/validation "
            "split, and report energy/force MAE and RMSE."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"MatterTune checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the XYZ path saved in the checkpoint.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Override the training fraction saved in the checkpoint.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help=(
            "Override the validation fraction. By default, use the checkpoint value; "
            "the saved value 'auto' means 1 - train_split."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Override the split shuffle seed saved in the checkpoint.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether indices are shuffled before splitting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Prediction batch size (default: 1, safest for large structures).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <checkpoint run>/train_val_eval).",
    )
    parser.add_argument(
        "--force-plot-max-points",
        type=int,
        default=DEFAULT_FORCE_PLOT_MAX_POINTS,
        help=(
            "Maximum uniformly sampled force components per split in the parity plot "
            f"(default: {DEFAULT_FORCE_PLOT_MAX_POINTS:,})."
        ),
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=DEFAULT_PLOT_SEED,
        help=f"Random seed for force-component sampling (default: {DEFAULT_PLOT_SEED}).",
    )
    return parser.parse_args()


def checkpoint_datamodule_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("datamodule_hyper_parameters")
    if not isinstance(config, dict):
        raise KeyError(
            "Checkpoint does not contain dict-valued datamodule_hyper_parameters; "
            "pass the dataset and split overrides explicitly."
        )
    return config


def resolve_dataset_and_split(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> tuple[Path, float, float, bool, int]:
    config = checkpoint_datamodule_config(checkpoint)
    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise KeyError("Checkpoint datamodule config has no dataset mapping.")
    if dataset_config.get("down_sample") is not None:
        raise ValueError(
            "This checkpoint used random dataset down-sampling, whose sampled indices "
            "were not saved; the exact original split cannot be reconstructed."
        )

    saved_data_path = dataset_config.get("src")
    data_path_value = args.data_path if args.data_path is not None else saved_data_path
    if data_path_value is None:
        raise ValueError("No dataset path was provided or saved in the checkpoint.")
    data_path = Path(data_path_value).expanduser().resolve()

    train_split = float(
        args.train_split
        if args.train_split is not None
        else config.get("train_split")
    )
    saved_validation_split = config.get("validation_split", "auto")
    if args.validation_split is not None:
        validation_split = float(args.validation_split)
    elif saved_validation_split == "auto":
        validation_split = 1.0 - train_split
    elif saved_validation_split == "disable":
        raise ValueError("The checkpoint disabled validation and has no validation split.")
    else:
        validation_split = float(saved_validation_split)

    shuffle = bool(config.get("shuffle", True) if args.shuffle is None else args.shuffle)
    shuffle_seed = int(
        config.get("shuffle_seed", 42)
        if args.shuffle_seed is None
        else args.shuffle_seed
    )

    if not 0.0 < train_split <= 1.0:
        raise ValueError(f"train_split must be in (0, 1], got {train_split}")
    if not 0.0 < validation_split <= 1.0:
        raise ValueError(
            f"validation_split must be in (0, 1], got {validation_split}"
        )
    if train_split + validation_split > 1.0 + 1.0e-12:
        raise ValueError(
            "train_split + validation_split must not exceed 1, got "
            f"{train_split + validation_split}"
        )

    return data_path, train_split, validation_split, shuffle, shuffle_seed


def count_extxyz_structures(path: Path) -> int:
    """Count extxyz frames without constructing all ASE Atoms objects."""
    count = 0
    line_number = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            atom_count_line = handle.readline()
            if not atom_count_line:
                break
            line_number += 1
            if not atom_count_line.strip():
                continue
            try:
                n_atoms = int(atom_count_line.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid atom count at {path}:{line_number}: "
                    f"{atom_count_line.rstrip()!r}"
                ) from exc
            if n_atoms < 0:
                raise ValueError(
                    f"Negative atom count at {path}:{line_number}: {n_atoms}"
                )

            comment = handle.readline()
            line_number += 1
            if not comment:
                raise ValueError(f"Truncated extxyz comment line after frame {count}.")
            for _ in range(n_atoms):
                atom_line = handle.readline()
                line_number += 1
                if not atom_line:
                    raise ValueError(f"Truncated extxyz atom block in frame {count}.")
            count += 1
    return count


def build_split_assignments(
    n_structures: int,
    *,
    train_split: float,
    validation_split: float,
    shuffle: bool,
    shuffle_seed: int,
) -> tuple[np.ndarray, int, int, int]:
    indices = np.arange(n_structures)
    if shuffle:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(indices)

    train_len = int(train_split * n_structures)
    validation_len = int(validation_split * n_structures)
    assignments = np.zeros(n_structures, dtype=np.int8)
    assignments[indices[:train_len]] = TRAIN
    assignments[indices[train_len : train_len + validation_len]] = VALIDATION
    unused_len = n_structures - train_len - validation_len
    return assignments, train_len, validation_len, unused_len


def atoms_energy(atoms: Atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception as calculator_error:
        for key in ("energy", "Energy", "dft_energy", "DFT_energy"):
            if key in atoms.info:
                return float(atoms.info[key])
        raise ValueError("Structure has no readable energy label.") from calculator_error


def atoms_forces(atoms: Atoms) -> np.ndarray:
    try:
        return np.asarray(atoms.get_forces(), dtype=np.float64)
    except Exception as calculator_error:
        for key in ("forces", "force"):
            if key in atoms.arrays:
                return np.asarray(atoms.arrays[key], dtype=np.float64)
        raise ValueError("Structure has no readable force labels.") from calculator_error


def load_per_atom_references(checkpoint: dict[str, Any]) -> dict[int, float]:
    normalizers = checkpoint.get("hyper_parameters", {}).get("normalizers", {})
    for normalizer in normalizers.get("energy", []):
        references = normalizer.get("per_atom_references")
        if references:
            return {
                int(atomic_number): float(value)
                for atomic_number, value in references.items()
            }
    return {}


def reference_energy(
    atoms: Atoms, per_atom_references: dict[int, float]
) -> float:
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    return float(
        sum(per_atom_references.get(int(number), 0.0) for number in atomic_numbers)
    )


def predict_batch(
    model: Any, atoms_batch: list[Atoms], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    data_list = [model.atoms_to_data(atoms, has_labels=False) for atoms in atoms_batch]
    batch = model.collate_fn(data_list)
    batch = model.batch_to_device(batch, device)
    model.zero_grad(set_to_none=True)
    # Conservative forces are energy gradients, so inference must retain autograd.
    with torch.enable_grad():
        predictions = model.predict_step(batch, batch_idx=0)
    return predictions


def evaluate_batch(
    model: Any,
    device: torch.device,
    records: list[tuple[int, int, Atoms]],
    metrics_by_split: dict[str, RunningMetrics],
    parity_by_split: dict[str, SplitParityData],
    per_atom_references: dict[int, float],
    writer: csv.writer,
) -> None:
    predictions = predict_batch(model, [record[2] for record in records], device)
    if len(predictions) != len(records):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(predictions)} for {len(records)} inputs."
        )

    for (structure_index, split_id, atoms), prediction in zip(
        records, predictions, strict=True
    ):
        split_name = SPLIT_NAMES[split_id]
        try:
            energy_ground_truth = atoms_energy(atoms)
            forces_ground_truth = atoms_forces(atoms)
        except ValueError as exc:
            raise ValueError(
                f"Failed to read labels for structure index {structure_index}."
            ) from exc

        if "energy" not in prediction or "forces" not in prediction:
            raise KeyError(
                "Checkpoint prediction does not contain both 'energy' and 'forces': "
                f"{sorted(prediction)}"
            )
        energy_prediction = float(prediction["energy"].detach().cpu().item())
        forces_prediction = np.asarray(
            prediction["forces"].detach().cpu(), dtype=np.float64
        )
        if forces_prediction.shape != forces_ground_truth.shape:
            raise ValueError(
                f"Force shape mismatch at structure {structure_index}: ground truth "
                f"{forces_ground_truth.shape}, prediction {forces_prediction.shape}."
            )

        energy_error_ev = energy_prediction - energy_ground_truth
        force_error_ev_per_a = forces_prediction - forces_ground_truth
        metrics_by_split[split_name].update(
            energy_error_ev=energy_error_ev,
            n_atoms=len(atoms),
            force_error_ev_per_a=force_error_ev_per_a,
        )

        energy_reference = reference_energy(atoms, per_atom_references)
        energy_ground_truth_ref_subtracted = (
            energy_ground_truth - energy_reference
        )
        energy_prediction_ref_subtracted = energy_prediction - energy_reference
        split_parity = parity_by_split[split_name]
        split_parity.energy_ground_truth_ref_subtracted_ev_per_atom.append(
            energy_ground_truth_ref_subtracted / len(atoms)
        )
        split_parity.energy_prediction_ref_subtracted_ev_per_atom.append(
            energy_prediction_ref_subtracted / len(atoms)
        )
        split_parity.energy_ground_truth_ref_subtracted_ev_per_structure.append(
            energy_ground_truth_ref_subtracted
        )
        split_parity.energy_prediction_ref_subtracted_ev_per_structure.append(
            energy_prediction_ref_subtracted
        )
        split_parity.force_sample.update(forces_ground_truth, forces_prediction)

        force_error_mev_per_a = force_error_ev_per_a.reshape(-1) * 1000.0
        writer.writerow(
            [
                structure_index,
                split_name,
                len(atoms),
                f"{energy_ground_truth:.12g}",
                f"{energy_prediction:.12g}",
                f"{energy_reference:.12g}",
                f"{energy_ground_truth_ref_subtracted:.12g}",
                f"{energy_prediction_ref_subtracted:.12g}",
                f"{energy_ground_truth_ref_subtracted / len(atoms):.12g}",
                f"{energy_prediction_ref_subtracted / len(atoms):.12g}",
                f"{energy_error_ev * 1000.0:.12g}",
                f"{energy_error_ev * 1000.0 / len(atoms):.12g}",
                force_error_mev_per_a.size,
                f"{np.abs(force_error_mev_per_a).mean():.12g}",
                f"{np.sqrt(np.square(force_error_mev_per_a).mean()):.12g}",
            ]
        )


def set_parity_limits(
    axis: Any,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    quantile_clip: float | None = None,
) -> None:
    values = np.concatenate([ground_truth.reshape(-1), prediction.reshape(-1)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if quantile_clip is None:
        lower = float(values.min())
        upper = float(values.max())
    else:
        lower, upper = np.quantile(
            values, [quantile_clip, 1.0 - quantile_clip]
        ).tolist()
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0e-6)
    else:
        padding = 0.03 * (upper - lower)
    lower -= padding
    upper += padding
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1.0)
    axis.set_aspect("equal", adjustable="box")


def save_energy_parity_plot(
    output_path: Path,
    parity_by_split: dict[str, SplitParityData],
    metrics: dict[str, dict[str, int | float]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    colors = {"train": "#2166ac", "validation": "#b2182b"}
    for row, split_name in enumerate(("train", "validation")):
        (
            ground_truth_per_atom,
            prediction_per_atom,
            ground_truth_per_structure,
            prediction_per_structure,
        ) = parity_by_split[split_name].energy_arrays()
        split_metrics = metrics[split_name]

        axis = axes[row, 0]
        axis.scatter(
            ground_truth_per_atom,
            prediction_per_atom,
            s=7,
            alpha=0.4,
            color=colors[split_name],
            rasterized=True,
        )
        set_parity_limits(axis, ground_truth_per_atom, prediction_per_atom)
        axis.set_xlabel("Reference-subtracted DFT energy (eV/atom)")
        axis.set_ylabel("Reference-subtracted predicted energy (eV/atom)")
        axis.set_title(
            f"{split_name.capitalize()} — per atom\n"
            f"MAE {float(split_metrics['energy_mae_meV_per_atom']):.3f} meV/atom, "
            f"RMSE {float(split_metrics['energy_rmse_meV_per_atom']):.3f} meV/atom"
        )

        axis = axes[row, 1]
        axis.scatter(
            ground_truth_per_structure,
            prediction_per_structure,
            s=7,
            alpha=0.4,
            color=colors[split_name],
            rasterized=True,
        )
        set_parity_limits(axis, ground_truth_per_structure, prediction_per_structure)
        axis.set_xlabel("Reference-subtracted DFT energy (eV/structure)")
        axis.set_ylabel("Reference-subtracted predicted energy (eV/structure)")
        axis.set_title(
            f"{split_name.capitalize()} — per structure\n"
            f"MAE {float(split_metrics['energy_mae_meV_per_structure']):.3f} "
            "meV/structure, "
            f"RMSE {float(split_metrics['energy_rmse_meV_per_structure']):.3f} "
            "meV/structure"
        )

    figure.suptitle("Energy parity after subtracting checkpoint atomic references")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_force_parity_plot(
    output_path: Path,
    parity_by_split: dict[str, SplitParityData],
    metrics: dict[str, dict[str, int | float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"train": "#2166ac", "validation": "#b2182b"}
    for axis, split_name in zip(axes, ("train", "validation"), strict=True):
        force_ground_truth, force_prediction = parity_by_split[
            split_name
        ].force_sample.arrays()
        split_metrics = metrics[split_name]
        axis.scatter(
            force_ground_truth,
            force_prediction,
            s=1,
            alpha=0.12,
            color=colors[split_name],
            rasterized=True,
        )
        set_parity_limits(
            axis,
            force_ground_truth,
            force_prediction,
            quantile_clip=0.001,
        )
        axis.set_xlabel("DFT force component (eV/A)")
        axis.set_ylabel("Predicted force component (eV/A)")
        axis.set_title(
            f"{split_name.capitalize()} forces\n"
            f"MAE {float(split_metrics['forces_mae_meV_per_A']):.3f} meV/A, "
            f"RMSE {float(split_metrics['forces_rmse_meV_per_A']):.3f} meV/A"
        )
        axis.text(
            0.02,
            0.98,
            f"plotted {force_ground_truth.size:,} / "
            f"{parity_by_split[split_name].force_sample.n_seen:,} components",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )

    figure.suptitle(
        "Force-component parity (axes clipped to the central 99.8% for readability)"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_parity_data(
    output_path: Path, parity_by_split: dict[str, SplitParityData]
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for split_name, parity in parity_by_split.items():
        (
            ground_truth_per_atom,
            prediction_per_atom,
            ground_truth_per_structure,
            prediction_per_structure,
        ) = parity.energy_arrays()
        force_ground_truth, force_prediction = parity.force_sample.arrays()
        arrays.update(
            {
                f"{split_name}_energy_ground_truth_ref_subtracted_eV_per_atom": (
                    ground_truth_per_atom
                ),
                f"{split_name}_energy_prediction_ref_subtracted_eV_per_atom": (
                    prediction_per_atom
                ),
                f"{split_name}_energy_ground_truth_ref_subtracted_eV_per_structure": (
                    ground_truth_per_structure
                ),
                f"{split_name}_energy_prediction_ref_subtracted_eV_per_structure": (
                    prediction_per_structure
                ),
                f"{split_name}_force_ground_truth_eV_per_A": force_ground_truth,
                f"{split_name}_force_prediction_eV_per_A": force_prediction,
                f"{split_name}_force_components_seen": np.asarray(
                    parity.force_sample.n_seen, dtype=np.int64
                ),
            }
        )
    np.savez_compressed(output_path, **arrays)


def default_output_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "train_val_eval"
    return checkpoint_path.parent / "train_val_eval"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def print_metrics_table(metrics: dict[str, dict[str, int | float]]) -> None:
    rows = [
        ("Structures", "n_structures"),
        ("Energy MAE (meV/atom)", "energy_mae_meV_per_atom"),
        ("Energy RMSE (meV/atom)", "energy_rmse_meV_per_atom"),
        ("Energy MAE (meV/structure)", "energy_mae_meV_per_structure"),
        ("Energy RMSE (meV/structure)", "energy_rmse_meV_per_structure"),
        ("Forces MAE (meV/A)", "forces_mae_meV_per_A"),
        ("Forces RMSE (meV/A)", "forces_rmse_meV_per_A"),
    ]
    print("\nMetric                              Train          Validation")
    print("-" * 64)
    for label, key in rows:
        train_value = metrics["train"][key]
        validation_value = metrics["validation"][key]
        if key == "n_structures":
            train_text = f"{int(train_value):,}"
            validation_text = f"{int(validation_value):,}"
        else:
            train_text = f"{float(train_value):.6f}"
            validation_text = f"{float(validation_value):.6f}"
        print(f"{label:<34}{train_text:>14}{validation_text:>18}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.force_plot_max_points <= 0:
        raise ValueError(
            "force_plot_max_points must be positive, got "
            f"{args.force_plot_max_points}"
        )

    checkpoint_path = args.ckpt.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    per_atom_references = load_per_atom_references(checkpoint)
    data_path, train_split, validation_split, shuffle, shuffle_seed = (
        resolve_dataset_and_split(args, checkpoint)
    )
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(checkpoint_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint:       {checkpoint_path}")
    print(
        f"Checkpoint epoch: {checkpoint.get('epoch')} "
        f"(completed epoch count: {int(checkpoint.get('epoch', -1)) + 1})"
    )
    print(f"Dataset:          {data_path}")
    print(f"Atomic references:{len(per_atom_references):>3} elements")
    print("Counting extxyz structures for exact split reconstruction...")
    n_structures = count_extxyz_structures(data_path)
    assignments, train_len, validation_len, unused_len = build_split_assignments(
        n_structures,
        train_split=train_split,
        validation_split=validation_split,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    print(
        "Split:            "
        f"train={train_len:,}, validation={validation_len:,}, unused={unused_len:,}, "
        f"shuffle={shuffle}, seed={shuffle_seed}"
    )

    torch.set_float32_matmul_precision("highest")
    device = resolve_device(args.device)
    print(f"Device:           {device}")
    print(f"Batch size:       {args.batch_size}")
    print("Loading checkpoint model...")
    model = load_finetuned_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval()
    model.to_device(device)
    model.hparams.using_partition = False

    metrics_by_split = {
        "train": RunningMetrics(),
        "validation": RunningMetrics(),
    }
    parity_by_split = {
        "train": SplitParityData(args.force_plot_max_points, args.plot_seed),
        "validation": SplitParityData(
            args.force_plot_max_points, args.plot_seed + 1
        ),
    }
    predictions_path = output_dir / "per_structure_predictions.csv"
    records: list[tuple[int, int, Atoms]] = []
    structures_read = 0
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "structure_index",
                "split",
                "n_atoms",
                "energy_ground_truth_eV",
                "energy_prediction_eV",
                "energy_atomic_reference_eV",
                "energy_ground_truth_ref_subtracted_eV_per_structure",
                "energy_prediction_ref_subtracted_eV_per_structure",
                "energy_ground_truth_ref_subtracted_eV_per_atom",
                "energy_prediction_ref_subtracted_eV_per_atom",
                "energy_error_meV_per_structure",
                "energy_error_meV_per_atom",
                "n_force_components",
                "forces_mae_meV_per_A_for_structure",
                "forces_rmse_meV_per_A_for_structure",
            ]
        )

        for structure_index, atoms in enumerate(
            tqdm(
                iread(data_path, index=":"),
                total=n_structures,
                desc="Evaluating train/validation",
                unit="structure",
            )
        ):
            structures_read += 1
            split_id = int(assignments[structure_index])
            if split_id == 0:
                continue
            records.append((structure_index, split_id, atoms))
            if len(records) >= args.batch_size:
                evaluate_batch(
                    model,
                    device,
                    records,
                    metrics_by_split,
                    parity_by_split,
                    per_atom_references,
                    writer,
                )
                records.clear()

        if records:
            evaluate_batch(
                model,
                device,
                records,
                metrics_by_split,
                parity_by_split,
                per_atom_references,
                writer,
            )

    if structures_read != n_structures:
        raise RuntimeError(
            f"Dataset changed during evaluation: counted {n_structures} structures "
            f"but ASE read {structures_read}."
        )

    split_metrics = {
        name: running_metrics.as_dict()
        for name, running_metrics in metrics_by_split.items()
    }
    result = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch_index": checkpoint.get("epoch"),
            "completed_epoch_count": int(checkpoint.get("epoch", -1)) + 1,
            "global_step": checkpoint.get("global_step"),
        },
        "dataset": {
            "path": str(data_path),
            "n_structures_total": n_structures,
            "train_split": train_split,
            "validation_split": validation_split,
            "shuffle": shuffle,
            "shuffle_seed": shuffle_seed,
            "n_unused_after_integer_split": unused_len,
        },
        "evaluation": {
            "device": str(device),
            "batch_size": args.batch_size,
            "energy_aggregation": "one error value per structure",
            "force_aggregation": "all Cartesian force components",
            "n_atomic_reference_elements": len(per_atom_references),
            "force_plot_max_points_per_split": args.force_plot_max_points,
            "plot_seed": args.plot_seed,
        },
        "metrics": split_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    parity_data_path = output_dir / "parity_data.npz"
    energy_plot_path = output_dir / "energy_parity.png"
    force_plot_path = output_dir / "force_parity.png"
    save_parity_data(parity_data_path, parity_by_split)
    save_energy_parity_plot(energy_plot_path, parity_by_split, split_metrics)
    save_force_parity_plot(force_plot_path, parity_by_split, split_metrics)

    print_metrics_table(split_metrics)
    print(f"\nSaved metrics:     {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved parity data: {parity_data_path}")
    print(f"Saved energy plot: {energy_plot_path}")
    print(f"Saved force plot:  {force_plot_path}")


if __name__ == "__main__":
    main()
