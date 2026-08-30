from __future__ import annotations

import sys
from pathlib import Path

from ase import Atoms
from ase.io import write


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mattertune.data.xyz import XYZDataset, XYZDatasetConfig


def test_xyz_dataset_reads_multiple_sources_in_order(tmp_path):
    first = tmp_path / "first.xyz"
    second = tmp_path / "second.xyz"
    write(first, [Atoms("H"), Atoms("He")], format="extxyz")
    write(second, [Atoms("Li")], format="extxyz")

    dataset = XYZDataset(XYZDatasetConfig(src=[first, second]))

    assert [atoms.get_chemical_formula() for atoms in dataset.atoms_list] == [
        "H",
        "He",
        "Li",
    ]
