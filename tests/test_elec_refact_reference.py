from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFACT_ROOT = ROOT / "examples" / "elec-refact"
sys.path.insert(0, str(REFACT_ROOT))

from config_schema import ReferenceSettings  # noqa: E402
from reference import ensure_energy_reference  # noqa: E402


def test_existing_shared_reference_is_reused(monkeypatch, tmp_path):
    output = tmp_path / "reference.json"
    output.write_text("{}\n", encoding="utf-8")
    args = Namespace(energy_reference=output)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("existing reference should not be regenerated")

    monkeypatch.setattr("reference._fit_residual_reference", fail_if_called)
    ensure_energy_reference(args, ReferenceSettings())


def test_missing_shared_reference_is_generated(monkeypatch, tmp_path):
    output = tmp_path / "references" / "reference.json"
    args = Namespace(energy_reference=output)
    calls = []

    def fake_fit(path, received_args, settings):
        calls.append((path, received_args, settings))
        path.write_text('{"3": -1.0}\n', encoding="utf-8")

    monkeypatch.setattr("reference._fit_residual_reference", fake_fit)
    settings = ReferenceSettings()
    ensure_energy_reference(args, settings)

    assert output.is_file()
    assert calls == [(output, args, settings)]
