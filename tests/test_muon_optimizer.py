from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mattertune.configs as MC  # noqa: E402
from mattertune.finetune.optimizer import create_optimizer  # noqa: E402


class FakeMuon(torch.optim.SGD):
    """SGD-backed test double with torch.optim.Muon's constructor signature."""

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn=None,
    ):
        del ns_coefficients, eps, ns_steps, adjust_lr_fn
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.hidden = nn.Linear(4, 4)
        self.output_head = nn.Linear(4, 1)


@pytest.fixture
def fake_muon(monkeypatch):
    monkeypatch.setattr(torch.optim, "Muon", FakeMuon, raising=False)


def test_muon_config_is_exported():
    config = MC.MuonConfig(lr=8e-5)

    assert config.name == "Muon"
    assert config.adjust_lr_fn == "match_rms_adamw"
    assert config.adamw_lr is None


def test_muon_splits_hidden_matrices_from_auxiliary_parameters(fake_muon):
    model = TinyModel()
    optimizer = create_optimizer(MC.MuonConfig(lr=8e-5), model.named_parameters())

    groups = {group["optimizer_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"muon", "adamw"}
    assert groups["muon"]["params"] == [model.hidden.weight]
    assert {id(parameter) for parameter in groups["adamw"]["params"]} == {
        id(model.embedding.weight),
        id(model.hidden.bias),
        id(model.output_head.weight),
        id(model.output_head.bias),
    }


def test_composite_optimizer_runs_closure_once_and_updates_all_parameters(fake_muon):
    model = TinyModel()
    optimizer = create_optimizer(MC.MuonConfig(lr=1e-3), model.named_parameters())
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = sum(parameter.square().sum() for parameter in model.parameters())
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    assert loss is not None
    assert closure_calls == 1
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name]), name


def test_composite_optimizer_state_and_scheduler_cover_both_children(fake_muon):
    model = TinyModel()
    optimizer = create_optimizer(MC.MuonConfig(lr=1e-3), model.named_parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
    )

    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step(1.0)
    scheduler.step(2.0)
    state_dict = optimizer.state_dict()
    scheduler_state_dict = scheduler.state_dict()

    restored = create_optimizer(MC.MuonConfig(lr=1e-3), model.named_parameters())
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        restored,
        factor=0.5,
        patience=0,
    )
    restored.load_state_dict(state_dict)
    restored_scheduler.load_state_dict(scheduler_state_dict)
    assert len(state_dict["optimizers"]) == 2
    restored_scheduler.step(3.0)
    assert [group["lr"] for group in restored.param_groups] == [2.5e-4, 2.5e-4]

    child_groups = [
        group
        for child_optimizer in restored.optimizers
        for group in child_optimizer.param_groups
    ]
    assert all(
        wrapper_group is child_group
        for wrapper_group, child_group in zip(
            restored.param_groups, child_groups, strict=True
        )
    )
    assert [group["lr"] for group in child_groups] == [2.5e-4, 2.5e-4]


def test_muon_requires_an_eligible_hidden_matrix(fake_muon):
    model = nn.LayerNorm(4)

    with pytest.raises(ValueError, match="No eligible 2D hidden matrix"):
        create_optimizer(MC.MuonConfig(lr=1e-3), model.named_parameters())


def test_muon_reports_unsupported_torch(monkeypatch):
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    model = TinyModel()

    with pytest.raises(RuntimeError, match="PyTorch 2.9 or newer"):
        create_optimizer(MC.MuonConfig(lr=1e-3), model.named_parameters())
