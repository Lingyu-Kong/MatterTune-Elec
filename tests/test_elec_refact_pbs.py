from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFACT_ROOT = ROOT / "examples" / "elec-refact"
SINGLE_NODE_SCRIPT = REFACT_ROOT / "run_single_node_multi_gpu.pbs"
MULTI_NODE_SCRIPT = REFACT_ROOT / "run_multi_node_multi_gpu.pbs"


def _fake_runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text(
        "conda() { [[ \"$1\" == activate ]]; }\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for executable in ("torchrun", "mpiexec"):
        path = fake_bin / executable
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PBS_O_WORKDIR": str(ROOT),
            "MATTERTUNE_DIR": str(ROOT),
            "CONDA_SH": str(conda_sh),
            "CUDA_MODULE": "",
            "DRY_RUN": "1",
            "RUN_NAME": "test-run",
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )
    return fake_bin, environment


def _run_dry(script: Path, node_names: list[str], tmp_path: Path) -> str:
    _, environment = _fake_runtime(tmp_path)
    nodefile = tmp_path / "pbs-nodefile"
    nodefile.write_text("".join(f"{name}\n" for name in node_names), encoding="utf-8")
    environment.update(
        {
            "PBS_JOBID": "1234.server",
            "PBS_NODEFILE": str(nodefile),
        }
    )
    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def test_pbs_scripts_have_valid_bash_syntax():
    for script in (SINGLE_NODE_SCRIPT, MULTI_NODE_SCRIPT):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_single_node_dry_run_builds_eight_worker_command(tmp_path):
    output = _run_dry(SINGLE_NODE_SCRIPT, ["node-a"], tmp_path)

    assert "scenario       = Mix-Homo-Sol" in output
    assert "nodes          = 1 (node-a)" in output
    assert "world_size     = 8" in output
    assert "--standalone" in output
    assert "--nproc-per-node=8" in output
    assert "trainer.num_nodes=1" in output
    assert "trainer.devices=\\[0\\,1\\,2\\,3\\,4\\,5\\,6\\,7\\]" in output
    assert "--prepare-reference-only" in output


def test_multi_node_dry_run_builds_two_by_eight_command(tmp_path):
    output = _run_dry(MULTI_NODE_SCRIPT, ["node-a", "node-b"], tmp_path)

    assert "nodes          = 2" in output
    assert "node_list      = node-a,node-b" in output
    assert "world_size     = 16" in output
    assert "master         = node-a:29500" in output
    assert "--map-by ppr:1:node" in output
    assert "--bind-to none" in output
    assert "--nnodes=2" in output
    assert "--nproc-per-node=8" in output
    assert "--rdzv-endpoint=node-a:29500" in output
    assert "trainer.num_nodes=2" in output
    assert "reference.refit=false" in output


def test_scenario_alias_selects_single_sol_entrypoint(tmp_path):
    _, environment = _fake_runtime(tmp_path)
    nodefile = tmp_path / "pbs-nodefile"
    nodefile.write_text("node-a\n", encoding="utf-8")
    environment.update(
        {
            "PBS_JOBID": "1234.server",
            "PBS_NODEFILE": str(nodefile),
            "SCENARIO": "single",
        }
    )

    completed = subprocess.run(
        ["bash", str(SINGLE_NODE_SCRIPT)],
        check=True,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert "scenario       = Single-Sol" in completed.stdout
    assert "examples/elec-refact/Single-Sol/train.py" in completed.stdout
