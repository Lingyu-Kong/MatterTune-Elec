from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFACT_ROOT = ROOT / "examples" / "elec-refact"
sys.path.insert(0, str(REFACT_ROOT))

from config_loader import load_training_settings  # noqa: E402
from training import build_training_args  # noqa: E402


SINGLE_CONFIG = REFACT_ROOT / "Single-Sol" / "configs" / "default.yml"
MIX_CONFIG = REFACT_ROOT / "Mix-LHCE" / "configs" / "default.yml"
MUON_CONFIG = REFACT_ROOT / "configs" / "optimizer" / "muon.yml"


def test_single_sol_default_matches_existing_training_defaults():
    settings = load_training_settings([SINGLE_CONFIG])
    args = build_training_args(settings, validate_files=False)

    assert settings.experiment.name == "single-sol"
    assert settings.data.batch_size == 8
    assert settings.trainer.devices == [0, 1, 2, 3, 4, 5]
    assert settings.optimizer.name == "adamw"
    assert settings.optimizer.lr == pytest.approx(8e-5)
    assert args.training_mode == "train_without_delta_e"
    assert args.init_checkpoint is None


def test_mix_lhce_default_initializes_from_single_sol_checkpoint():
    settings = load_training_settings([MIX_CONFIG])
    args = build_training_args(settings, validate_files=False)

    assert settings.experiment.name == "mix-lhce"
    assert settings.data.batch_size == 1
    assert settings.trainer.devices == [2, 5, 6, 7]
    assert settings.checkpoint.init_checkpoint is not None
    assert "Li-electrolyte-V1/local_runs/enhance-V1" in str(args.init_checkpoint)
    assert args.resume_checkpoint is None


def test_later_optimizer_yaml_overrides_default_optimizer():
    settings = load_training_settings([SINGLE_CONFIG, MUON_CONFIG])
    args = build_training_args(settings, validate_files=False)

    assert settings.optimizer.name == "muon"
    assert settings.optimizer.muon_adjust_lr_fn == "match_rms_adamw"
    assert args.optimizer == "muon"
    assert args.muon_exclude_pattern == [
        "*embed*",
        "*lm_head*",
        "*output_head*",
    ]


def test_dotted_overrides_are_typed_by_yaml():
    settings = load_training_settings(
        [SINGLE_CONFIG],
        overrides=(
            "optimizer.name=adam",
            "optimizer.lr=5e-5",
            "trainer.devices=[1, 3]",
            "logging.offline=true",
        ),
    )

    assert settings.optimizer.name == "adam"
    assert settings.optimizer.lr == pytest.approx(5e-5)
    assert settings.trainer.devices == [1, 3]
    assert settings.logging.offline is True


def test_environment_interpolation_changes_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SINGLE_SOL_DATA_ROOT", str(tmp_path))
    settings = load_training_settings([SINGLE_CONFIG])

    assert settings.data.train_file == str(tmp_path / "Li_electrolyte_V1_all.xyz")
    assert settings.experiment.output_root.startswith(str(tmp_path))


def test_unknown_configuration_key_is_rejected(tmp_path):
    overlay = tmp_path / "invalid.yml"
    overlay.write_text("optimizer:\n  typo_learning_rate: 1e-4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="typo_learning_rate"):
        load_training_settings([SINGLE_CONFIG, overlay])
