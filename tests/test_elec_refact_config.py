from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFACT_ROOT = ROOT / "examples" / "elec-refact"
sys.path.insert(0, str(REFACT_ROOT))

from config_loader import load_training_settings  # noqa: E402
from training import build_training_args  # noqa: E402


SINGLE_CONFIG = REFACT_ROOT / "Single-Sol" / "configs" / "default.yml"
MIX_HOMO_CONFIG = REFACT_ROOT / "Mix-Homo-Sol" / "configs" / "default.yml"
MIX_CONFIG = REFACT_ROOT / "Mix-LHCE" / "configs" / "default.yml"
MUON_CONFIG = REFACT_ROOT / "configs" / "optimizer" / "muon.yml"


def test_single_sol_default_matches_existing_training_defaults():
    settings = load_training_settings([SINGLE_CONFIG])
    args = build_training_args(settings, validate_files=False)

    assert settings.experiment.name == "single-sol"
    assert settings.data.batch_size == 8
    assert settings.trainer.devices == [0, 1, 2, 3]
    assert settings.optimizer.name == "adamw"
    assert settings.optimizer.lr == pytest.approx(8e-5)
    assert isinstance(settings.data.train_file, list)
    assert len(settings.data.train_file) == 7
    assert settings.data.train_file[4].endswith("/Li-surface.xyz")
    assert settings.data.energy_reference.startswith(
        "/net/csefiles/coc-fung-cluster/lingyu/ElectrolyteResults/"
    )
    assert settings.reference.method == "residual"
    assert settings.reference.regression == "ridge"
    assert settings.reference.auto_generate is True
    assert args.training_mode == "train_without_delta_e"
    assert args.init_checkpoint is None


def test_mix_homo_sol_uses_all_new_data_sources():
    settings = load_training_settings([MIX_HOMO_CONFIG])
    args = build_training_args(settings, validate_files=True)

    assert settings.experiment.name == "mix-homo-sol"
    assert settings.data.batch_size == 8
    assert isinstance(args.train_file, list)
    assert len(args.train_file) == 15
    assert all(path.is_file() for path in args.train_file)
    assert [path.name for path in args.train_file[-7:]] == [
        "Li_bulk_lambda0_del.xyz",
        "Li_bulk_lambda0.xyz",
        "Li_bulk_lambda1_del.xyz",
        "Li_bulk_lambda1.xyz",
        "Li-bulk-lambda0_mdv1_del.xyz",
        "Li-bulk-lambda0_mdv1.xyz",
        "Li-surface.xyz",
    ]
    assert settings.experiment.output_root.endswith("/ElectrolyteResults/mix-homo-sol")


def test_mix_homo_sol_training_settings_match_single_sol():
    single = load_training_settings([SINGLE_CONFIG])
    mixed = load_training_settings([MIX_HOMO_CONFIG])

    for section in (
        "model",
        "reference",
        "objective",
        "optimizer",
        "scheduler",
        "trainer",
        "checkpoint",
        "evaluation",
    ):
        assert getattr(mixed, section) == getattr(single, section)

    for field in (
        "pair_train_file",
        "test_file",
        "train_split",
        "shuffle",
        "shuffle_seed",
        "batch_size",
        "num_workers",
        "pin_memory",
        "max_parent_frame",
    ):
        assert getattr(mixed.data, field) == getattr(single.data, field)


def test_mix_lhce_has_no_legacy_run_dependency():
    settings = load_training_settings([MIX_CONFIG])
    args = build_training_args(settings, validate_files=False)

    assert settings.experiment.name == "mix-lhce"
    assert settings.data.batch_size == 1
    assert settings.trainer.devices == [2, 5, 6, 7]
    assert settings.checkpoint.init_checkpoint is None
    assert args.init_checkpoint is None
    assert args.resume_checkpoint is None
    assert settings.data.train_file.startswith(
        "/net/csefiles/coc-fung-cluster/lingyu/ElectrolyteData/mix-LHCE/"
    )


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

    assert settings.data.train_file == [
        str(tmp_path / "Li_bulk_lambda0_del.xyz"),
        str(tmp_path / "Li_bulk_lambda0.xyz"),
        str(tmp_path / "Li_bulk_lambda1_del.xyz"),
        str(tmp_path / "Li_bulk_lambda1.xyz"),
        str(tmp_path / "Li-surface.xyz"),
        str(tmp_path / "Li-bulk-lambda0_mdv1_del.xyz"),
        str(tmp_path / "Li-bulk-lambda0_mdv1.xyz"),
    ]
    assert settings.experiment.output_root.endswith("/ElectrolyteResults/single-sol")


def test_job_directory_is_named_from_start_time():
    settings = load_training_settings([SINGLE_CONFIG])
    started_at = datetime(2026, 8, 13, 21, 4, 5, 123456)

    args = build_training_args(
        settings,
        validate_files=False,
        started_at=started_at,
    )

    assert args.job_name == "20260813-210405-123456-adamw"
    assert args.output_dir == Path(settings.experiment.output_root) / args.job_name
    assert args.checkpoint_dir == args.output_dir / "checkpoints"
    assert args.log_dir == args.output_dir / "logs"


def test_unknown_configuration_key_is_rejected(tmp_path):
    overlay = tmp_path / "invalid.yml"
    overlay.write_text("optimizer:\n  typo_learning_rate: 1e-4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="typo_learning_rate"):
        load_training_settings([SINGLE_CONFIG, overlay])
