from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence
from typing import Literal

import ase
from ase import Atoms
from ase.io import read
import numpy as np
from torch.utils.data import Dataset
from typing_extensions import override
import copy

from ..registry import data_registry
from .base import DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class XYZDatasetConfig(DatasetConfigBase):
    type: Literal["xyz"] = "xyz"
    """Discriminator for the XYZ dataset."""

    src: str | Path | Sequence[str | Path]
    """One XYZ path, or an ordered collection of XYZ paths."""

    down_sample: int | None = None
    """Down sample the dataset"""

    down_sample_refill: bool = False
    """Refill the dataset after down sampling to achieve the same length as the original dataset"""

    @override
    def create_dataset(self):
        return XYZDataset(self)


class XYZDataset(Dataset[ase.Atoms]):
    def __init__(self, config: XYZDatasetConfig):
        super().__init__()
        self.config = config

        sources = (
            [self.config.src]
            if isinstance(self.config.src, (str, Path))
            else list(self.config.src)
        )
        atoms_list: list[Atoms] = []
        for source in sources:
            source_atoms = read(str(source), index=":")
            assert isinstance(source_atoms, list), "Expected a list of Atoms objects"
            atoms_list.extend(source_atoms)
        if self.config.down_sample is not None:
            ori_length = len(atoms_list)
            down_indices = np.random.choice(
                ori_length, self.config.down_sample, replace=False
            )
            if self.config.down_sample_refill:
                refilled_down_indices = []
                for _ in range((ori_length // self.config.down_sample)):
                    refilled_down_indices.extend(copy.deepcopy(down_indices))
                if len(refilled_down_indices) != ori_length:
                    res = np.random.choice(
                        len(down_indices),
                        ori_length - len(refilled_down_indices),
                        replace=False,
                    )
                    refilled_down_indices.extend([down_indices[i] for i in res])
                new_atoms_list = [
                    copy.deepcopy(atoms_list[i]) for i in refilled_down_indices
                ]
                atoms_list = new_atoms_list
            else:
                new_atoms_list = [copy.deepcopy(atoms_list[i]) for i in down_indices]
                atoms_list = new_atoms_list
        self.atoms_list: list[Atoms] = atoms_list
        log.info(
            "Loaded %d structures from %d XYZ source(s): %s",
            len(self.atoms_list),
            len(sources),
            ", ".join(str(source) for source in sources),
        )

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        return self.atoms_list[idx]

    def __len__(self) -> int:
        return len(self.atoms_list)
