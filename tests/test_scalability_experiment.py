from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hidden"
    / "scalability-test"
)


def load_indexed_xyz_module():
    module_name = "mattertune_scalability_indexed_xyz"
    spec = importlib.util.spec_from_file_location(
        module_name, EXPERIMENT_DIR / "indexed_xyz.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def labeled_atoms(offset: float) -> Atoms:
    atoms = Atoms(
        "LiH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5 + offset]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-2.0 + offset,
        forces=np.full((2, 3), offset),
    )
    return atoms


def test_byte_offset_index_round_trip(tmp_path: Path):
    indexed_xyz = load_indexed_xyz_module()
    left = tmp_path / "left.xyz"
    right = tmp_path / "right.xyz"
    write(left, [labeled_atoms(0.0), labeled_atoms(0.1)], format="extxyz")
    write(right, [labeled_atoms(0.2)], format="extxyz")

    index = indexed_xyz.ensure_xyz_index(
        [left, right], tmp_path / "dataset-index.json"
    )
    assert index["num_frames"] == 3
    assert index["source_frame_counts"] == [2, 1]

    dataset = indexed_xyz.IndexedXYZSubset(index, [2, 0])
    assert len(dataset) == 2
    assert np.isclose(dataset[0].get_potential_energy(), -1.8)
    assert np.allclose(dataset[1].get_forces(), 0.0)


def test_nested_downsamples_preserve_split_ratio():
    indexed_xyz = load_indexed_xyz_module()
    small_train, small_validation = indexed_xyz.split_subset_indices(
        100, 20, 0.9, 42
    )
    large_train, large_validation = indexed_xyz.split_subset_indices(
        100, 50, 0.9, 42
    )

    assert (len(small_train), len(small_validation)) == (18, 2)
    assert (len(large_train), len(large_validation)) == (45, 5)
    assert set(small_train).issubset(set(large_train))
    assert set(small_validation).issubset(set(large_validation))
    assert set(large_train).isdisjoint(set(large_validation))
