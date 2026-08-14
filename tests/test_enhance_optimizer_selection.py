from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "examples" / "elec-Li-new" / "enhance-V1" / "train.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "enhance_v1_train_for_optimizer_test", TRAIN_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("optimizer_name", "config_class_name"),
    (("adam", "AdamConfig"), ("adamw", "AdamWConfig"), ("muon", "MuonConfig")),
)
def test_enhance_train_selects_optimizer(
    monkeypatch,
    tmp_path,
    optimizer_name,
    config_class_name,
):
    train = load_train_module()
    train_file = tmp_path / "train.xyz"
    energy_reference = tmp_path / "reference.json"
    train_file.touch()
    energy_reference.write_text("{}")
    output_dir = tmp_path / optimizer_name
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--model_type",
            "mattersim",
            "--optimizer",
            optimizer_name,
            "--train_file",
            str(train_file),
            "--energy_reference",
            str(energy_reference),
            "--output_dir",
            str(output_dir),
            "--skip_eval",
        ],
    )

    args = train.parse_args()
    config = train.build_config(args)

    assert type(config.model.optimizer).__name__ == config_class_name
    assert config.model.optimizer.lr == pytest.approx(3e-5)
    assert config.model.optimizer.weight_decay == pytest.approx(0.1)


def test_enhance_train_rejects_muon_for_unvalidated_backbones(monkeypatch, tmp_path):
    train = load_train_module()
    train_file = tmp_path / "train.xyz"
    energy_reference = tmp_path / "reference.json"
    train_file.touch()
    energy_reference.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--model_type",
            "uma",
            "--optimizer",
            "muon",
            "--train_file",
            str(train_file),
            "--energy_reference",
            str(energy_reference),
            "--skip_eval",
        ],
    )

    with pytest.raises(ValueError, match="only for the MatterSim backbone"):
        train.parse_args()
