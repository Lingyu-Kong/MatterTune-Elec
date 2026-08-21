from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.io import read
from tqdm import tqdm

from mattertune.main import load_pretrained_model

if TYPE_CHECKING:
    from argparse import Namespace

    from config_schema import ReferenceSettings


def _training_sources(train_file: Path | list[Path]) -> list[Path]:
    return train_file if isinstance(train_file, list) else [train_file]


def _load_atoms(sources: list[Path]) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    for source in sources:
        source_atoms = read(source, index=":")
        if not isinstance(source_atoms, list):
            source_atoms = [source_atoms]
        atoms_list.extend(source_atoms)
    if not atoms_list:
        raise ValueError("Cannot fit an energy reference from an empty dataset.")
    return atoms_list


def _composition_matrix(atoms_list: list[Atoms]) -> np.ndarray:
    max_atomic_number = max(max(atoms.numbers) for atoms in atoms_list)
    matrix = np.zeros((len(atoms_list), max_atomic_number + 1), dtype=np.float64)
    for row, atoms in enumerate(atoms_list):
        for atomic_number, count in Counter(atoms.numbers).items():
            matrix[row, atomic_number] = count
    return matrix


def _fit_per_element_references(
    compositions: np.ndarray,
    residual_energies: np.ndarray,
    *,
    regression: str,
    ridge_alpha: float,
) -> dict[int, float]:
    if regression == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    elif regression == "linear":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(fit_intercept=False)
    else:
        raise ValueError(f"Unsupported reference regression: {regression}")

    coefficients = model.fit(compositions, residual_energies).coef_
    references = {
        atomic_number: float(reference)
        for atomic_number, reference in enumerate(coefficients.tolist())
        if atomic_number != 0
    }
    return references


def _pretrained_energies(atoms_list: list[Atoms], args: Namespace) -> np.ndarray:
    load_kwargs: dict[str, Any] = {}
    if args.model_type == "uma":
        load_kwargs["task_name"] = args.task_name
    model = load_pretrained_model(
        model_type=args.model_type,
        model_name=args.model_name,
        device=args.reference_device,
        **load_kwargs,
    )
    calculator = model.ase_calculator()

    energies: list[float] = []
    for atoms in tqdm(atoms_list, desc="pretrained reference energies"):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calculator
        energies.append(float(atoms_copy.get_potential_energy()))
    return np.asarray(energies, dtype=np.float64)


def _atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _fit_residual_reference(
    output: Path,
    args: Namespace,
    settings: ReferenceSettings,
) -> None:
    sources = _training_sources(args.train_file)
    atoms_list = _load_atoms(sources)
    target_energies = np.asarray(
        [float(atoms.get_potential_energy()) for atoms in atoms_list],
        dtype=np.float64,
    )
    baseline_energies = _pretrained_energies(atoms_list, args)
    residual_energies = target_energies - baseline_energies
    compositions = _composition_matrix(atoms_list)
    references = _fit_per_element_references(
        compositions,
        residual_energies,
        regression=settings.regression,
        ridge_alpha=settings.ridge_alpha,
    )

    reference_vector = np.asarray(
        [references.get(index, 0.0) for index in range(compositions.shape[1])],
        dtype=np.float64,
    )
    residual_after_reference = residual_energies - compositions @ reference_vector
    summary = {
        "method": "residual",
        "definition": "target_energy_minus_pretrained_energy",
        "baseline": settings.baseline,
        "regression": settings.regression,
        "ridge_alpha": settings.ridge_alpha,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "task_name": args.task_name,
        "force_mode": args.force_mode,
        "device": args.reference_device,
        "frames": len(atoms_list),
        "sources": [
            {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
            }
            for source in sources
        ],
        "mean_target_energy_eV": float(target_energies.mean()),
        "mean_pretrained_energy_eV": float(baseline_energies.mean()),
        "mean_residual_energy_eV": float(residual_energies.mean()),
        "residual_after_reference_mae_eV": float(
            np.abs(residual_after_reference).mean()
        ),
        "residual_after_reference_rmse_eV": float(
            np.sqrt(np.mean(residual_after_reference**2))
        ),
        "references": references,
    }
    _atomic_json_dump(output, references)
    _atomic_json_dump(output.with_suffix(".summary.json"), summary)


def ensure_energy_reference(args: Namespace, settings: ReferenceSettings) -> None:
    output = args.energy_reference
    if output.is_file() and not settings.refit:
        print(f"Using shared energy reference: {output}")
        return
    if not settings.auto_generate:
        raise FileNotFoundError(output)
    if settings.method != "residual":
        raise ValueError(f"Unsupported reference method: {settings.method}")
    if settings.baseline != "ase_pretrained":
        raise ValueError(f"Unsupported residual baseline: {settings.baseline}")

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(f"{output.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if output.is_file() and not settings.refit:
            print(f"Using shared energy reference: {output}")
            return
        print(f"Generating shared residual energy reference: {output}")
        _fit_residual_reference(output, args, settings)
        print(f"Saved shared residual energy reference: {output}")
