from __future__ import annotations

import copy
import csv
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "examples" / "hidden" / "finite-difference-training"
sys.path.insert(0, str(EXPERIMENT_ROOT))

import analyze as fd_analyze  # noqa: E402
import callbacks as fd_callbacks  # noqa: E402
import one_epoch_benchmark as fd_one_epoch  # noqa: E402
import run as fd_run  # noqa: E402
import worker as fd_worker  # noqa: E402
from callbacks import DatasetAuditCallback, RNGStateCallback  # noqa: E402
from datamodule import complete_split_indices  # noqa: E402
from estimators import (  # noqa: E402
    DirectionalSettings,
    attach_directional_training,
    finite_difference_from_energies,
    graph_coordinate_dimensions,
    graph_projection,
    perturbed_batch,
    unit_rademacher_directions,
    weighted_directional_mse,
)
from protocol import (  # noqa: E402
    build_training_overlay,
    load_protocol,
    resolve_run,
    with_devices,
)
from results import epoch_metrics, metrics_at_best_epoch  # noqa: E402
from results import checkpoint_files  # noqa: E402
from mattertune.finetune.loss import MSELossConfig  # noqa: E402
from mattertune.finetune.properties import (  # noqa: E402
    EnergyPropertyConfig,
    ForcesPropertyConfig,
)

# These examples intentionally remain standalone scripts with local module names
# such as ``protocol`` and ``results``. Keep them out of the global import cache
# so other standalone-example tests can import their own modules in one pytest run.
for _module_name in (
    "analyze",
    "callbacks",
    "datamodule",
    "estimators",
    "one_epoch_benchmark",
    "protocol",
    "results",
    "run",
    "worker",
):
    sys.modules.pop(_module_name, None)


def test_default_protocol_matches_decided_method_policy():
    protocol = load_protocol()
    assert protocol.seed == 42
    assert protocol.devices == (0, 1, 2, 3)
    assert protocol.expected_dataset_size == 28279
    assert protocol.logging_backend == "wandb"
    assert protocol.num_directions == 1
    assert protocol.methods["exact"].batch_size == 8
    assert protocol.methods["ad-directional"].batch_size == 8
    assert protocol.methods["forward-fd"].batch_size == 16
    assert protocol.methods["forward-fd"].epsilon == pytest.approx(0.01)
    assert protocol.methods["central-fd"].batch_size == 12
    assert protocol.methods["central-fd"].epsilon == pytest.approx(0.1)
    assert protocol.optimizer.name == "muon"
    assert protocol.optimizer.lr == pytest.approx(8e-5)
    assert protocol.success.relative_mae_tolerance == pytest.approx(0.01)


def test_complete_split_uses_all_structures_and_is_deterministic():
    train, validation = complete_split_indices(
        28279, 0.9, shuffle=True, seed=42
    )
    repeated_train, repeated_validation = complete_split_indices(
        28279, 0.9, shuffle=True, seed=42
    )
    assert (len(train), len(validation)) == (25451, 2828)
    assert len(set(train) | set(validation)) == 28279
    assert set(train).isdisjoint(set(validation))
    assert np.array_equal(train, repeated_train)
    assert np.array_equal(validation, repeated_validation)


def test_energy_normalizer_audit_requires_elec_refact_chain():
    reference = type("PerAtomReferencingNormalizerModule", (), {})()
    per_atom = type("PerAtomNormalizerModule", (), {})()
    model = SimpleNamespace(
        normalizers={"energy": SimpleNamespace(normalizers=[reference, per_atom])}
    )
    assert fd_worker.audit_energy_normalizers(model) == [
        "PerAtomReferencingNormalizerModule",
        "PerAtomNormalizerModule",
    ]
    model.normalizers["energy"].normalizers.reverse()
    with pytest.raises(RuntimeError, match="differs from elec-refact"):
        fd_worker.audit_energy_normalizers(model)


def test_launcher_worker_stdout_has_a_pseudoterminal(tmp_path):
    log_path = tmp_path / "launcher.log"
    returncode, _ = fd_run._tee_process(
        [sys.executable, "-c", "import sys; print(sys.stdout.isatty())"],
        log_path,
        wandb_run_id="test-run-id",
    )
    assert returncode == 0
    assert b"True" in log_path.read_bytes()


def test_estimator_specific_cli_validation_and_overlay(tmp_path):
    protocol = load_protocol()
    central = resolve_run(protocol, "central-fd")
    assert central.batch_size == 12
    assert central.num_directions == 1
    assert central.epsilon == pytest.approx(0.1)
    changed = resolve_run(
        protocol,
        "central-fd",
        batch_size=6,
        epsilon=0.03,
        epsilon_was_set=True,
        num_directions=4,
    )
    assert changed.to_dict() == {
        "estimator": "central-fd",
        "batch_size": 6,
        "epsilon": 0.03,
        "num_directions": 4,
    }
    auto_k2 = resolve_run(protocol, "central-fd", num_directions=2)
    auto_k4 = resolve_run(protocol, "central-fd", num_directions=4)
    assert auto_k2.batch_size == 7
    assert auto_k4.batch_size == 4
    with pytest.raises(ValueError, match="only valid"):
        resolve_run(
            protocol, "exact", epsilon=0.1, epsilon_was_set=True
        )
    with pytest.raises(ValueError, match="not valid for exact"):
        resolve_run(protocol, "exact", num_directions=1)

    overlay = build_training_overlay(protocol, central, tmp_path / "run")
    assert overlay["optimizer"]["name"] == "muon"
    assert overlay["optimizer"]["lr"] == pytest.approx(8e-5)
    assert overlay["objective"]["energy_weight"] == pytest.approx(200.0)
    assert overlay["objective"]["force_weight"] == pytest.approx(20.0)
    assert overlay["scheduler"]["monitor"] == "val/total_loss"
    assert overlay["trainer"]["ema_decay"] == 0.0
    assert overlay["logging"]["log_loss_grad_norms"] is False
    assert overlay["logging"]["backend"] == "wandb"
    assert overlay["logging"]["project"] == "MatterTune-Finite-Difference"
    assert overlay["logging"]["offline"] is False
    assert overlay["checkpoint"]["save_last"] is False

    init = tmp_path / "external.ckpt"
    init.touch()
    init_overlay = build_training_overlay(
        protocol, central, tmp_path / "init-run", init_checkpoint=init
    )
    assert init_overlay["checkpoint"]["init_checkpoint"] == str(init)
    assert init_overlay["checkpoint"]["resume_checkpoint"] is None

    latest = tmp_path / "last.ckpt"
    latest.touch()
    resume_overlay = build_training_overlay(
        protocol, central, tmp_path / "resume-run", resume_checkpoint=latest
    )
    assert resume_overlay["checkpoint"]["init_checkpoint"] is None
    assert resume_overlay["checkpoint"]["resume_checkpoint"] == str(latest)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_training_overlay(
            protocol,
            central,
            tmp_path / "bad-run",
            init_checkpoint=init,
            resume_checkpoint=latest,
        )


def test_one_epoch_benchmark_runs_all_methods_without_smoke_limits(tmp_path):
    cli = fd_one_epoch.parse_args(
        ["--devices", "0", "1", "2", "3", "--output-dir", str(tmp_path)]
    )
    fd_one_epoch.validate_cli(cli)
    for estimator in ("exact", "ad-directional", "forward-fd", "central-fd"):
        command = fd_one_epoch.build_command(
            cli,
            estimator,
            fd_one_epoch.estimator_run_dir(tmp_path, estimator),
            resume=False,
        )
        assert "--smoke" not in command
        assert command.count("--estimator") == 1
        assert command[command.index("--estimator") + 1] == estimator
        assert "trainer.max_epochs=1" in command


def test_one_epoch_benchmark_reports_speedups_against_exact():
    rows = [
        {
            "estimator": "exact",
            "train_seconds": 100.0,
            "full_epoch_seconds": 120.0,
            "launcher_wall_seconds": 140.0,
        },
        {
            "estimator": "central-fd",
            "train_seconds": 50.0,
            "full_epoch_seconds": 70.0,
            "launcher_wall_seconds": 100.0,
        },
    ]
    fd_one_epoch.add_speedups(rows)
    assert rows[1]["train_speedup_vs_exact"] == pytest.approx(2.0)
    assert rows[1]["full_epoch_speedup_vs_exact"] == pytest.approx(120.0 / 70.0)
    assert rows[1]["wall_speedup_vs_exact"] == pytest.approx(1.4)


def test_one_epoch_benchmark_requires_complete_timing_metrics():
    summary = {
        "estimator": "exact",
        "batch_size_per_gpu": 8,
        "num_devices": 4,
        "train_size": 25451,
        "validation_size": 2828,
        "mean_train_epoch_seconds": 10.0,
        "mean_validation_epoch_seconds": None,
        "mean_full_epoch_seconds": None,
        "total_wall_seconds": 12.0,
        "train_structures_per_second": 2545.1,
        "peak_memory_gib": 20.0,
        "val_total_loss": 1.0,
        "val_energy_mae": 0.1,
        "val_forces_mae": 0.2,
        "run_dir": "/tmp/exact",
    }
    with pytest.raises(RuntimeError, match="missing required timing metrics"):
        fd_one_epoch.benchmark_row(summary)


def test_runtime_callback_separates_train_and_validation_time(monkeypatch):
    callback = fd_callbacks.RuntimeStatsCallback()
    logged = {}

    class Strategy:
        def barrier(self, name):
            assert name.startswith("finite_difference_")

    trainer = SimpleNamespace(
        strategy=Strategy(), sanity_checking=False, world_size=1
    )
    module = SimpleNamespace(
        device=torch.device("cpu"),
        log=lambda name, value, **kwargs: logged.__setitem__(name, value),
    )
    clock = iter((10.0, 20.0, 35.0, 40.0, 40.0))
    monkeypatch.setattr(fd_callbacks.time, "perf_counter", lambda: next(clock))

    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_start(trainer, module)
    callback.on_validation_start(trainer, module)
    callback.on_validation_end(trainer, module)
    callback.on_train_epoch_end(trainer, module)

    assert logged["time/train_epoch_seconds"] == pytest.approx(15.0)
    assert logged["time/validation_epoch_seconds"] == pytest.approx(5.0)
    assert logged["time/full_epoch_seconds"] == pytest.approx(20.0)
    assert logged["time/elapsed_seconds"] == pytest.approx(30.0)


def test_unit_rademacher_direction_norms_each_graph():
    atom_pos = torch.zeros((3, 3))
    graph_index = torch.tensor([0, 0, 1])
    dimensions = graph_coordinate_dimensions(graph_index, 2, dtype=atom_pos.dtype)
    generator = torch.Generator().manual_seed(7)
    direction = unit_rademacher_directions(
        atom_pos, graph_index, dimensions, generator=generator
    )
    squared = direction.square().sum(dim=1)
    graph_norms = torch.zeros(2).index_add(0, graph_index, squared)
    assert dimensions.tolist() == [6.0, 3.0]
    assert torch.allclose(graph_norms, torch.ones(2))


def test_enumerated_rademacher_estimator_equals_coordinate_force_mse():
    error = torch.tensor([[1.0, -2.0, 4.0]])
    graph_index = torch.tensor([0])
    dimensions = torch.tensor([3.0])
    residuals = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        direction = torch.tensor([signs]) / (3.0**0.5)
        residuals.append(graph_projection(direction, error, graph_index, 1))
    estimated = weighted_directional_mse(torch.stack(residuals), dimensions)
    assert estimated == pytest.approx(float(error.square().mean()))


def test_forward_and_central_difference_and_parameter_gradient():
    theta = torch.tensor(2.5, requires_grad=True)
    position = torch.tensor([0.3, -0.7, 1.1])
    direction = torch.tensor([1.0, -1.0, 1.0]) / (3.0**0.5)
    epsilon = 0.1

    def energy(pos: torch.Tensor) -> torch.Tensor:
        return 0.5 * theta * pos.square().sum()

    base = energy(position)
    plus = energy(position + epsilon * direction)
    minus = energy(position - epsilon * direction)
    exact = theta * torch.dot(position, direction)
    central = finite_difference_from_energies(
        estimator="central-fd",
        epsilon=epsilon,
        base_energy=base,
        plus_energy=plus,
        minus_energy=minus,
    ).squeeze()
    forward = finite_difference_from_energies(
        estimator="forward-fd",
        epsilon=epsilon,
        base_energy=base,
        plus_energy=plus,
    ).squeeze()
    assert float(central.detach()) == pytest.approx(float(exact.detach()), rel=1e-5)
    assert float(forward.detach()) == pytest.approx(
        float((exact + 0.5 * theta * epsilon).detach()), rel=1e-5
    )
    central.square().backward()
    expected_grad = 2.0 * float(theta.detach()) * float(torch.dot(position, direction)) ** 2
    assert theta.grad == pytest.approx(expected_grad, rel=1e-5)


class FakeBatch:
    def __init__(self):
        self.atom_pos = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, -0.2, 0.5]], dtype=torch.float32
        )
        self.batch = torch.tensor([0, 1])
        self.num_graphs = 2
        self.energy = torch.tensor([0.0, 0.0])
        self.forces = torch.tensor(
            [[-0.1, -0.2, -0.3], [-0.4, 0.2, -0.5]], dtype=torch.float32
        )
        self.cell = torch.eye(3).repeat(2, 1, 1)
        self.pbc = torch.tensor([[True, True, True], [False, False, False]])
        self.edge_index = torch.tensor([[0, 1], [0, 1]])

    def clone(self):
        return copy.deepcopy(self)


class FakeDirectionalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(1.5))
        self.calc_forces = True
        self.force_forward_calls = 0
        self.validation_calls = 0
        self.logged = {}
        self.hparams = SimpleNamespace(
            properties=[
                EnergyPropertyConfig(
                    loss=MSELossConfig(), loss_coefficient=200.0
                ),
                ForcesPropertyConfig(
                    loss=MSELossConfig(),
                    loss_coefficient=20.0,
                    conservative=True,
                ),
            ]
        )

    def forward(self, batch, mode):
        per_atom = 0.5 * self.theta * batch.atom_pos.square().sum(dim=1)
        energy = torch.zeros(batch.num_graphs).index_add(0, batch.batch, per_atom)
        predictions = {"energy": energy}
        if self.calc_forces:
            self.force_forward_calls += 1
            predictions["forces"] = -self.theta * batch.atom_pos
        return {"predicted_properties": predictions}

    def batch_to_labels(self, batch):
        return {"energy": batch.energy, "forces": batch.forces}

    def create_normalization_context_from_batch(self, batch):
        return None

    def normalize(self, predictions, labels, context):
        return predictions, labels

    def denormalize(self, predictions, labels, context):
        return predictions, labels

    def log_dict(self, values, *args, **kwargs):
        self.logged.update(values)

    def _common_step(self, batch, mode, metrics, log=True):
        self.validation_calls += 1
        output = self(batch, mode)
        return output, output["predicted_properties"]["energy"].sum()


def test_central_training_does_not_request_forces_and_keeps_batch_unchanged():
    torch.manual_seed(42)
    module = FakeDirectionalModule()
    batch = FakeBatch()
    original_pos = batch.atom_pos.clone()
    original_cell = batch.cell.clone()
    original_edges = batch.edge_index.clone()
    original_pbc = batch.pbc.clone()
    attach_directional_training(
        module,
        DirectionalSettings(estimator="central-fd", num_directions=2, epsilon=0.1),
    )
    _, loss = module._common_step(batch, "train", None, log=False)
    loss.backward()
    assert module.force_forward_calls == 0
    assert module.theta.grad is not None
    assert float(module.theta.grad.abs()) > 0
    assert torch.equal(batch.atom_pos, original_pos)
    assert torch.equal(batch.cell, original_cell)
    assert torch.equal(batch.edge_index, original_edges)
    assert torch.equal(batch.pbc, original_pbc)

    module._common_step(batch, "val", None, log=False)
    assert module.validation_calls == 1
    assert module.force_forward_calls == 1


def test_logged_energy_mae_is_denormalized_to_physical_energy():
    class ReferencedModule(FakeDirectionalModule):
        def normalize(self, predictions, labels, context):
            return (
                {"energy": predictions["energy"] / 2.0},
                {"energy": (labels["energy"] - 10.0) / 2.0},
            )

        def denormalize(self, predictions, labels, context):
            return (
                {"energy": predictions["energy"] * 2.0 + 10.0},
                {"energy": labels["energy"] * 2.0 + 10.0},
            )

    torch.manual_seed(42)
    module = ReferencedModule()
    batch = FakeBatch()
    batch.energy.fill_(10.0)
    with torch.no_grad():
        expected = module(batch, "train")["predicted_properties"]["energy"].abs().mean()
    module.force_forward_calls = 0
    attach_directional_training(
        module,
        DirectionalSettings(estimator="forward-fd", num_directions=1, epsilon=0.1),
    )
    module._common_step(batch, "train", None, log=True)
    logged_mae = float(module.logged["train/energy_mae"].detach())
    assert logged_mae == pytest.approx(float(expected), rel=1e-5)
    assert logged_mae < 1.0


def test_perturbed_batch_only_changes_positions():
    batch = FakeBatch()
    direction = torch.ones_like(batch.atom_pos)
    changed = perturbed_batch(batch, direction, 0.25)
    assert torch.allclose(changed.atom_pos, batch.atom_pos + 0.25)
    assert torch.equal(changed.cell, batch.cell)
    assert torch.equal(changed.edge_index, batch.edge_index)
    assert torch.equal(changed.pbc, batch.pbc)
    assert not torch.equal(changed.atom_pos, batch.atom_pos)


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True)
    rows = [
        {"epoch": 0, "val/total_loss": 2.0, "val/energy_mae": 0.2, "val/forces_mae": 0.3},
        {"epoch": 1, "train/total_loss_epoch": 1.5},
        {
            "epoch": 1,
            "val/total_loss": 1.0,
            "val/energy_mae": 0.1,
            "val/forces_mae": 0.2,
            "time/train_epoch_seconds": 100.0,
            "time/validation_epoch_seconds": 20.0,
            "time/full_epoch_seconds": 120.0,
        },
    ]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_result_parser_merges_sparse_rows_and_selects_best(tmp_path):
    _write_metrics(tmp_path / "logs/lightning_logs/version_0/metrics.csv")
    merged = epoch_metrics(tmp_path)
    assert merged[1]["train/total_loss_epoch"] == pytest.approx(1.5)
    assert merged[1]["time/full_epoch_seconds"] == pytest.approx(120.0)
    epoch, metrics = metrics_at_best_epoch(tmp_path, best_epoch=None)
    assert epoch == 1
    assert metrics["val/energy_mae"] == pytest.approx(0.1)
    assert metrics["val/forces_mae"] == pytest.approx(0.2)


def test_checkpoint_parser_selects_best_and_true_latest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    older_best = checkpoint_dir / "model-best.ckpt"
    older_best.touch()
    latest = checkpoint_dir / "last.ckpt"
    latest.touch()
    best, last = checkpoint_files(tmp_path)
    assert best == older_best
    assert last == latest


def test_dataset_audit_records_actual_split_and_rejects_wrong_total(tmp_path):
    underlying = list(range(10))
    # Special methods are looked up on the type, so use small concrete wrappers.
    class Split:
        def __init__(self, dataset, size):
            self.dataset = dataset
            self.size = size

        def __len__(self):
            return self.size

    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            datasets={"train": Split(underlying, 8), "validation": Split(underlying, 1)}
        ),
        is_global_zero=True,
    )
    DatasetAuditCallback(tmp_path, expected_size=10).on_fit_start(trainer, None)
    assert (tmp_path / "dataset-summary.json").read_text().strip()
    with pytest.raises(RuntimeError, match="expected=11, actual=10"):
        DatasetAuditCallback(tmp_path, expected_size=11).on_fit_start(trainer, None)


def test_rng_callback_round_trips_python_numpy_and_torch_state():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    callback = RNGStateCallback()
    callback.on_train_epoch_end(
        SimpleNamespace(), SimpleNamespace(device=torch.device("cpu"))
    )
    state = callback.state_dict()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored = RNGStateCallback()
    restored.load_state_dict(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == pytest.approx(expected)


def test_manifest_refuses_overwrite_and_validates_resume(monkeypatch, tmp_path):
    protocol = with_devices(load_protocol(), [0])
    run = resolve_run(protocol, "central-fd")
    run_dir = tmp_path / "run"
    inventory = {
        "hostname": "test",
        "pid": 1,
        "python": "python",
        "gpus": [{"name": "NVIDIA A40"}],
    }
    monkeypatch.setattr(fd_run, "host_inventory", lambda devices: inventory)
    manifest, path = fd_run.prepare_manifest(
        protocol,
        run,
        run_dir,
        resume=False,
        smoke=False,
        init_checkpoint=None,
    )
    assert path.is_file()
    assert manifest["run_config"] == run.to_dict()
    wandb_run_id = manifest["monitoring"]["wandb_run_id"]
    assert wandb_run_id
    with pytest.raises(FileExistsError):
        fd_run.prepare_manifest(
            protocol,
            run,
            run_dir,
            resume=False,
            smoke=False,
            init_checkpoint=None,
        )
    resumed, _ = fd_run.prepare_manifest(
        protocol,
        run,
        run_dir,
        resume=True,
        smoke=False,
        init_checkpoint=None,
    )
    assert len(resumed["hosts"]) == 2
    assert resumed["monitoring"]["wandb_run_id"] == wandb_run_id
    changed = resolve_run(protocol, "central-fd", batch_size=6)
    with pytest.raises(ValueError, match="estimator/K/epsilon/batch"):
        fd_run.prepare_manifest(
            protocol,
            changed,
            run_dir,
            resume=True,
            smoke=False,
            init_checkpoint=None,
        )


def test_success_requires_both_maes_and_cost_to_pass():
    protocol = load_protocol().to_dict()
    manifest = {
        "protocol": protocol,
        "num_devices": 4,
        "host": {"gpus": [{"name": "NVIDIA A40"}] * 4},
    }
    exact = {
        "val_energy_mae": 1.0,
        "val_forces_mae": 2.0,
        "total_wall_seconds": 100.0,
        "gpu_hours": 4.0,
    }
    candidate = {
        "estimator": "central-fd",
        "val_energy_mae": 1.005,
        "val_forces_mae": 2.01,
        "total_wall_seconds": 80.0,
        "gpu_hours": 3.2,
    }
    result = fd_analyze.compare_summaries(
        manifest, exact, manifest, candidate
    )
    assert result["successful"] is True
    candidate["val_forces_mae"] = 2.03
    result = fd_analyze.compare_summaries(
        manifest, exact, manifest, candidate
    )
    assert result["successful"] is False
