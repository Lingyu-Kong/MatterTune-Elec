"""Evaluate one or more labeled XYZ datasets with a MatterTune checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_train_val_metrics import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_FORCE_PLOT_MAX_POINTS,
    DEFAULT_PLOT_SEED,
    ForceReservoir,
    RunningMetrics,
    atoms_energy,
    atoms_forces,
    count_extxyz_structures,
    default_output_dir,
    load_finetuned_checkpoint,
    load_per_atom_references,
    predict_batch,
    reference_energy,
    resolve_device,
    set_parity_limits,
)


class DatasetParityData:
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
            "Evaluate labeled XYZ datasets and save energy/force MAE, RMSE, and "
            "reference-subtracted parity plots."
        )
    )
    parser.add_argument(
        "xyz_paths",
        type=Path,
        nargs="+",
        help="One or more labeled extxyz datasets.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"MatterTune checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Prediction batch size (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root (default: <checkpoint run>/dataset_eval).",
    )
    parser.add_argument(
        "--force-plot-max-points",
        type=int,
        default=DEFAULT_FORCE_PLOT_MAX_POINTS,
        help=(
            "Maximum uniformly sampled force components in each parity plot "
            f"(default: {DEFAULT_FORCE_PLOT_MAX_POINTS:,})."
        ),
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=DEFAULT_PLOT_SEED,
        help=f"Force sampling seed (default: {DEFAULT_PLOT_SEED}).",
    )
    return parser.parse_args()


def evaluate_batch(
    model: Any,
    device: torch.device,
    records: list[tuple[int, Atoms]],
    metrics: RunningMetrics,
    parity: DatasetParityData,
    per_atom_references: dict[int, float],
    writer: csv.writer,
) -> None:
    predictions = predict_batch(model, [record[1] for record in records], device)
    if len(predictions) != len(records):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(predictions)} for {len(records)} inputs."
        )

    for (structure_index, atoms), prediction in zip(
        records, predictions, strict=True
    ):
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
        metrics.update(
            energy_error_ev=energy_error_ev,
            n_atoms=len(atoms),
            force_error_ev_per_a=force_error_ev_per_a,
        )

        energy_reference = reference_energy(atoms, per_atom_references)
        energy_ground_truth_ref_subtracted = energy_ground_truth - energy_reference
        energy_prediction_ref_subtracted = energy_prediction - energy_reference
        parity.energy_ground_truth_ref_subtracted_ev_per_atom.append(
            energy_ground_truth_ref_subtracted / len(atoms)
        )
        parity.energy_prediction_ref_subtracted_ev_per_atom.append(
            energy_prediction_ref_subtracted / len(atoms)
        )
        parity.energy_ground_truth_ref_subtracted_ev_per_structure.append(
            energy_ground_truth_ref_subtracted
        )
        parity.energy_prediction_ref_subtracted_ev_per_structure.append(
            energy_prediction_ref_subtracted
        )
        parity.force_sample.update(forces_ground_truth, forces_prediction)

        force_error_mev_per_a = force_error_ev_per_a.reshape(-1) * 1000.0
        writer.writerow(
            [
                structure_index,
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


def save_energy_plot(
    output_path: Path,
    dataset_name: str,
    parity: DatasetParityData,
    metrics: dict[str, int | float],
) -> None:
    (
        ground_truth_per_atom,
        prediction_per_atom,
        ground_truth_per_structure,
        prediction_per_structure,
    ) = parity.energy_arrays()
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))

    axis = axes[0]
    axis.scatter(
        ground_truth_per_atom,
        prediction_per_atom,
        s=8,
        alpha=0.45,
        color="#2166ac",
        rasterized=True,
    )
    set_parity_limits(axis, ground_truth_per_atom, prediction_per_atom)
    axis.set_xlabel("Reference-subtracted DFT energy (eV/atom)")
    axis.set_ylabel("Reference-subtracted predicted energy (eV/atom)")
    axis.set_title(
        "Per atom\n"
        f"MAE {float(metrics['energy_mae_meV_per_atom']):.3f} meV/atom, "
        f"RMSE {float(metrics['energy_rmse_meV_per_atom']):.3f} meV/atom"
    )

    axis = axes[1]
    axis.scatter(
        ground_truth_per_structure,
        prediction_per_structure,
        s=8,
        alpha=0.45,
        color="#2166ac",
        rasterized=True,
    )
    set_parity_limits(axis, ground_truth_per_structure, prediction_per_structure)
    axis.set_xlabel("Reference-subtracted DFT energy (eV/structure)")
    axis.set_ylabel("Reference-subtracted predicted energy (eV/structure)")
    axis.set_title(
        "Per structure\n"
        f"MAE {float(metrics['energy_mae_meV_per_structure']):.3f} meV/structure, "
        f"RMSE {float(metrics['energy_rmse_meV_per_structure']):.3f} meV/structure"
    )

    figure.suptitle(
        f"{dataset_name}: energy parity after subtracting checkpoint atomic references"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_force_plot(
    output_path: Path,
    dataset_name: str,
    parity: DatasetParityData,
    metrics: dict[str, int | float],
) -> None:
    force_ground_truth, force_prediction = parity.force_sample.arrays()
    figure, axis = plt.subplots(1, 1, figsize=(7, 6.5))
    axis.scatter(
        force_ground_truth,
        force_prediction,
        s=1,
        alpha=0.12,
        color="#2166ac",
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
        f"{dataset_name}: force-component parity\n"
        f"MAE {float(metrics['forces_mae_meV_per_A']):.3f} meV/A, "
        f"RMSE {float(metrics['forces_rmse_meV_per_A']):.3f} meV/A"
    )
    axis.text(
        0.02,
        0.98,
        f"plotted {force_ground_truth.size:,} / "
        f"{parity.force_sample.n_seen:,} components\n"
        "axes: central 99.8%",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_parity_data(output_path: Path, parity: DatasetParityData) -> None:
    (
        ground_truth_per_atom,
        prediction_per_atom,
        ground_truth_per_structure,
        prediction_per_structure,
    ) = parity.energy_arrays()
    force_ground_truth, force_prediction = parity.force_sample.arrays()
    np.savez_compressed(
        output_path,
        energy_ground_truth_ref_subtracted_eV_per_atom=ground_truth_per_atom,
        energy_prediction_ref_subtracted_eV_per_atom=prediction_per_atom,
        energy_ground_truth_ref_subtracted_eV_per_structure=(
            ground_truth_per_structure
        ),
        energy_prediction_ref_subtracted_eV_per_structure=(
            prediction_per_structure
        ),
        force_ground_truth_eV_per_A=force_ground_truth,
        force_prediction_eV_per_A=force_prediction,
        force_components_seen=np.asarray(parity.force_sample.n_seen, dtype=np.int64),
    )


def print_metrics_table(dataset_name: str, metrics: dict[str, int | float]) -> None:
    print(f"\n{dataset_name}")
    print("-" * 62)
    print(f"Structures                       {int(metrics['n_structures']):>16,}")
    print(
        "Energy MAE (meV/atom)         "
        f"{float(metrics['energy_mae_meV_per_atom']):>16.6f}"
    )
    print(
        "Energy RMSE (meV/atom)        "
        f"{float(metrics['energy_rmse_meV_per_atom']):>16.6f}"
    )
    print(
        "Energy MAE (meV/structure)    "
        f"{float(metrics['energy_mae_meV_per_structure']):>16.6f}"
    )
    print(
        "Energy RMSE (meV/structure)   "
        f"{float(metrics['energy_rmse_meV_per_structure']):>16.6f}"
    )
    print(
        "Forces MAE (meV/A)            "
        f"{float(metrics['forces_mae_meV_per_A']):>16.6f}"
    )
    print(
        "Forces RMSE (meV/A)           "
        f"{float(metrics['forces_rmse_meV_per_A']):>16.6f}"
    )


def evaluate_dataset(
    *,
    model: Any,
    device: torch.device,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    xyz_path: Path,
    output_dir: Path,
    batch_size: int,
    force_plot_max_points: int,
    plot_seed: int,
    per_atom_references: dict[int, float],
) -> dict[str, int | float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_structures = count_extxyz_structures(xyz_path)
    metrics_accumulator = RunningMetrics()
    parity = DatasetParityData(force_plot_max_points, plot_seed)
    predictions_path = output_dir / "per_structure_predictions.csv"
    records: list[tuple[int, Atoms]] = []
    structures_read = 0

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "structure_index",
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
                iread(xyz_path, index=":"),
                total=n_structures,
                desc=xyz_path.stem,
                unit="structure",
            )
        ):
            structures_read += 1
            records.append((structure_index, atoms))
            if len(records) >= batch_size:
                evaluate_batch(
                    model,
                    device,
                    records,
                    metrics_accumulator,
                    parity,
                    per_atom_references,
                    writer,
                )
                records.clear()
        if records:
            evaluate_batch(
                model,
                device,
                records,
                metrics_accumulator,
                parity,
                per_atom_references,
                writer,
            )

    if structures_read != n_structures:
        raise RuntimeError(
            f"Dataset changed during evaluation: counted {n_structures} structures "
            f"but read {structures_read}."
        )

    metrics = metrics_accumulator.as_dict()
    result = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch_index": checkpoint.get("epoch"),
            "completed_epoch_count": int(checkpoint.get("epoch", -1)) + 1,
            "global_step": checkpoint.get("global_step"),
        },
        "dataset": {"path": str(xyz_path), "name": xyz_path.stem},
        "evaluation": {
            "device": str(device),
            "batch_size": batch_size,
            "n_atomic_reference_elements": len(per_atom_references),
            "energy_aggregation": "one error value per structure",
            "force_aggregation": "all Cartesian force components",
            "force_plot_max_points": force_plot_max_points,
            "plot_seed": plot_seed,
        },
        "metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    save_parity_data(output_dir / "parity_data.npz", parity)
    save_energy_plot(output_dir / "energy_parity.png", xyz_path.stem, parity, metrics)
    save_force_plot(output_dir / "force_parity.png", xyz_path.stem, parity, metrics)
    print_metrics_table(xyz_path.stem, metrics)
    print(f"Output: {output_dir}")
    return metrics


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
    xyz_paths = [path.expanduser().resolve() for path in args.xyz_paths]
    for xyz_path in xyz_paths:
        if not xyz_path.is_file():
            raise FileNotFoundError(xyz_path)

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else default_output_dir(checkpoint_path).parent / "dataset_eval"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    per_atom_references = load_per_atom_references(checkpoint)
    device = resolve_device(args.device)

    print(f"Checkpoint:       {checkpoint_path}")
    print(
        f"Checkpoint epoch: {checkpoint.get('epoch')} "
        f"(completed epoch count: {int(checkpoint.get('epoch', -1)) + 1})"
    )
    print(f"Device:           {device}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Atomic references:{len(per_atom_references):>3} elements")
    print("Loading checkpoint model once for all datasets...")
    torch.set_float32_matmul_precision("highest")
    model = load_finetuned_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval()
    model.to_device(device)
    model.hparams.using_partition = False

    all_metrics: dict[str, dict[str, int | float]] = {}
    for dataset_index, xyz_path in enumerate(xyz_paths):
        print(f"\nEvaluating: {xyz_path}")
        all_metrics[xyz_path.stem] = evaluate_dataset(
            model=model,
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            xyz_path=xyz_path,
            output_dir=output_root / xyz_path.stem,
            batch_size=args.batch_size,
            force_plot_max_points=args.force_plot_max_points,
            plot_seed=args.plot_seed + dataset_index,
            per_atom_references=per_atom_references,
        )

    with (output_root / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, sort_keys=True)
    print(f"\nSaved combined summary: {output_root / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
