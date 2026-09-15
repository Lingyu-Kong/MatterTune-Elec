from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TWO_STAGE_ROOT = ROOT / "examples" / "hidden" / "two-stage-training"
sys.path.insert(0, str(TWO_STAGE_ROOT))

import analyze as two_stage_analyze  # noqa: E402
import run as two_stage_run  # noqa: E402
from protocol import (  # noqa: E402
    build_stage_overlay,
    load_protocol,
    smoke_protocol,
    with_devices,
    with_stage1_max_epochs,
)
from results import epoch_metrics, metrics_at_best_epoch, summarize_run  # noqa: E402


def test_default_protocol_matches_decided_training_policy():
    protocol = load_protocol()

    assert protocol.seed == 42
    assert protocol.devices == (0, 1, 2, 3)
    assert protocol.expected_dataset_size == 28279
    assert protocol.stage1.batch_size == 16
    assert protocol.stage1.max_epochs == 50
    assert protocol.stage1.early_stopping_patience == 20
    assert protocol.stage1.monitor == "val/energy_loss"
    assert protocol.stage2.batch_size == 8
    assert protocol.stage2.max_epochs == 5000
    assert protocol.stage2.early_stopping_patience == 100
    assert protocol.stage2.monitor == "val/total_loss"
    assert protocol.success.relative_mae_tolerance == pytest.approx(0.01)


def test_stage_overlays_enforce_supervision_and_fresh_stage2_state(tmp_path):
    protocol = load_protocol()
    source_checkpoint = tmp_path / "stage1-best.ckpt"
    stage1 = build_stage_overlay(protocol, "stage1", tmp_path / "stage1")
    stage2 = build_stage_overlay(
        protocol,
        "stage2",
        tmp_path / "stage2",
        init_checkpoint=source_checkpoint,
    )
    baseline = build_stage_overlay(protocol, "baseline", tmp_path / "baseline")

    assert stage1["objective"]["force_weight"] == 0.0
    assert stage1["objective"]["force_training_strategy"] == "none"
    assert stage1["objective"]["validation_force_mode"] == "energy_only"
    assert stage1["data"]["batch_size"] == 16
    assert stage1["scheduler"]["monitor"] == "val/energy_loss"

    assert stage2["objective"] == baseline["objective"]
    assert stage2["optimizer"] == baseline["optimizer"]
    assert stage2["scheduler"] == baseline["scheduler"]
    assert stage2["trainer"] == baseline["trainer"]
    assert stage2["data"]["batch_size"] == baseline["data"]["batch_size"] == 8
    assert stage2["checkpoint"]["init_checkpoint"] == str(source_checkpoint)
    assert stage2["checkpoint"]["resume_checkpoint"] is None
    assert stage2["trainer"]["ema_decay"] == 0.0


def test_resume_and_weight_initialization_are_mutually_exclusive(tmp_path):
    protocol = load_protocol()
    checkpoint = tmp_path / "last.ckpt"
    overlay = build_stage_overlay(
        protocol,
        "stage2",
        tmp_path / "stage2",
        resume_checkpoint=checkpoint,
    )
    assert overlay["checkpoint"]["init_checkpoint"] is None
    assert overlay["checkpoint"]["resume_checkpoint"] == str(checkpoint)

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_stage_overlay(
            protocol,
            "stage2",
            tmp_path / "bad",
            init_checkpoint=checkpoint,
            resume_checkpoint=checkpoint,
        )


def test_cli_specific_k_and_smoke_settings_override_defaults():
    protocol = with_stage1_max_epochs(load_protocol(), 100)
    protocol = with_devices(protocol, [5])
    smoke = smoke_protocol(protocol, 5)

    assert protocol.stage1.max_epochs == 100
    assert protocol.devices == (5,)
    assert smoke.stage1.max_epochs == 1
    assert smoke.stage2.max_epochs == 1
    assert smoke.devices == (5,)
    overlay = build_stage_overlay(smoke, "stage2", Path("/tmp/stage2"), smoke=True)
    assert overlay["trainer"]["limit_train_batches"] == 2
    assert overlay["trainer"]["limit_val_batches"] == 1
    assert overlay["logging"]["log_loss_grad_norms"] is False


def test_prepare_manifest_refuses_overwrite_and_validates_resume(
    monkeypatch, tmp_path
):
    protocol = with_devices(load_protocol(), [0])
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        two_stage_run,
        "host_inventory",
        lambda devices: {"hostname": "test", "pid": 1, "python": "python", "gpus": []},
    )

    manifest, path = two_stage_run.prepare_manifest(
        protocol,
        variant="two-stage",
        run_dir=run_dir,
        resume=False,
        smoke=False,
    )
    assert path.is_file()
    assert manifest["status"] == "running"
    with pytest.raises(FileExistsError):
        two_stage_run.prepare_manifest(
            protocol,
            variant="two-stage",
            run_dir=run_dir,
            resume=False,
            smoke=False,
        )

    resumed, _ = two_stage_run.prepare_manifest(
        protocol,
        variant="two-stage",
        run_dir=run_dir,
        resume=True,
        smoke=False,
    )
    assert resumed["protocol"] == protocol.to_dict()
    changed = with_stage1_max_epochs(protocol, 100)
    with pytest.raises(ValueError, match="Resume protocol differs"):
        two_stage_run.prepare_manifest(
            changed,
            variant="two-stage",
            run_dir=run_dir,
            resume=True,
            smoke=False,
        )


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_metric_reader_merges_sparse_rows_and_selects_best_epoch(tmp_path):
    stage_dir = tmp_path / "stage"
    _write_metrics(
        stage_dir / "logs/lightning_logs/version_0/metrics.csv",
        [
            {"epoch": 0, "train/total_loss": 2.0},
            {
                "epoch": 0,
                "val/total_loss": 1.0,
                "val/energy_mae": 0.2,
                "val/forces_mae": 0.3,
            },
            {
                "epoch": 1,
                "val/total_loss": 0.8,
                "val/energy_mae": 0.1,
                "val/forces_mae": 0.2,
            },
        ],
    )

    merged = epoch_metrics(stage_dir)
    assert merged[0]["train/total_loss"] == pytest.approx(2.0)
    selected_epoch, selected = metrics_at_best_epoch(
        stage_dir, best_epoch=None, monitor="val/total_loss"
    )
    assert selected_epoch == 1
    assert selected["val/energy_mae"] == pytest.approx(0.1)


def _manifest(
    run_dir: Path,
    *,
    variant: str,
    wall_seconds: float,
    energy_mae: float,
    forces_mae: float,
    k: int = 50,
) -> dict[str, object]:
    role = "baseline" if variant == "baseline" else "stage2"
    return {
        "run_dir": str(run_dir),
        "variant": variant,
        "seed": 42,
        "num_devices": 4,
        "host": {"gpus": [{"name": "NVIDIA A40"}] * 4},
        "protocol": {"stage1": {"max_epochs": k}},
        "stages": {
            role: {
                "epochs_completed": 10,
                "wall_seconds": wall_seconds,
                "best_checkpoint": str(run_dir / "best.ckpt"),
                "best_metrics": {
                    "val/total_loss": 1.0,
                    "val/energy_mae": energy_mae,
                    "val/forces_mae": forces_mae,
                },
            }
        },
    }


def test_success_requires_both_maes_time_and_matching_hardware(tmp_path):
    baseline = summarize_run(
        _manifest(
            tmp_path / "baseline",
            variant="baseline",
            wall_seconds=100.0,
            energy_mae=1.0,
            forces_mae=2.0,
        )
    )
    good = summarize_run(
        _manifest(
            tmp_path / "good",
            variant="two-stage",
            wall_seconds=80.0,
            energy_mae=1.01,
            forces_mae=2.02,
        )
    )
    inaccurate = summarize_run(
        _manifest(
            tmp_path / "bad",
            variant="two-stage",
            wall_seconds=70.0,
            energy_mae=1.02,
            forces_mae=2.0,
        )
    )

    rows = two_stage_analyze.compare_rows(
        baseline, [good, inaccurate], tolerance=0.01
    )
    assert rows[1]["success"] is True
    assert rows[1]["speedup_vs_baseline"] == pytest.approx(1.25)
    assert rows[2]["accuracy_within_tolerance"] is False
    assert rows[2]["success"] is False
