# Electrolyte fine-tuning

This directory provides YAML-composed training entry points for three
electrolyte dataset classes:

- `Single-Sol`: fine-tune MatterSim on single-solvent electrolyte structures.
- `Mix-Homo-Sol`: fine-tune on the eight homogeneous mixed-solvent files plus
  the seven `Single-Sol` files.
- `Mix-LHCE`: fine-tune on localized-high-concentration electrolyte structures.

The numerical training implementation is shared. Scenario directories contain
only their data, objective, checkpoint, logging, evaluation, and trainer
overrides. Model, optimizer, scheduler, and baseline trainer configurations are
reused from the top-level `configs` directory.

Run these entry points from the repository root inside the appropriate
MatterSim environment (currently `mattersim-elec`).

## Configuration composition

Each scenario's `configs/default.yml` is a manifest. Its `includes` are loaded
in order, and later values override earlier values. Parameters are grouped by
the component they affect:

- `model`
- `data`
- `reference`
- `objective`
- `optimizer`
- `scheduler`
- `trainer`
- `checkpoint`
- `logging`
- `evaluation`

`${env:NAME,default}` reads an environment variable with a fallback value.
`${section.key}` references another resolved configuration value.

## Inspect before training

```bash
PYTHONPATH=src python examples/elec-refact/Single-Sol/train.py --dry-run
```

```bash
PYTHONPATH=src python examples/elec-refact/Mix-Homo-Sol/train.py --dry-run
```

```bash
PYTHONPATH=src python examples/elec-refact/Mix-LHCE/train.py --dry-run
```

Dry runs do not require data files and print both the grouped settings and all
derived training arguments.

## Start training

```bash
PYTHONPATH=src python examples/elec-refact/Single-Sol/train.py
```

```bash
PYTHONPATH=src python examples/elec-refact/Mix-Homo-Sol/train.py
```

```bash
PYTHONPATH=src python examples/elec-refact/Mix-LHCE/train.py
```

Raw training inputs are read only from `/net/csefiles/coc-fung-cluster/lingyu/ElectrolyteData`
by default. Derived artifacts are written under
`/net/csefiles/coc-fung-cluster/lingyu/ElectrolyteResults`:

```text
ElectrolyteResults/
  single-sol/
    references/
    <job-start-time>-<optimizer>/
      resolved-config.yml
      checkpoints/
      logs/
  mix-homo-sol/
    references/
    <job-start-time>-<optimizer>/
  mix-LHCE/
    references/
    <job-start-time>-<optimizer>/
```

If a class-level energy reference is missing, the launcher computes it before
training. The reference fits per-element contributions to
`E_target - E_pretrained` using ridge regression without an intercept. It is
then shared by jobs in the same dataset class. Set `reference.refit=true` to
force regeneration.

Every job saves its fully resolved settings and derived arguments as
`resolved-config.yml` before reference generation or training begins.

## Select an optimizer

The default optimizer is AdamW. Add a reusable optimizer YAML as a later
overlay to select Adam or Muon:

```bash
PYTHONPATH=src python examples/elec-refact/Single-Sol/train.py \
  --config examples/elec-refact/configs/optimizer/muon.yml
```

```bash
PYTHONPATH=src python examples/elec-refact/Mix-LHCE/train.py \
  --config examples/elec-refact/configs/optimizer/adam.yml
```

For one-off experiments, use repeated dotted overrides:

```bash
PYTHONPATH=src python examples/elec-refact/Single-Sol/train.py \
  --set optimizer.name=muon \
  --set optimizer.lr=5e-5 \
  --set trainer.devices='[0, 1]'
```

An additional YAML overlay is preferable when a setting should be reused or
committed as an experiment definition.

## Submit multi-GPU PBS jobs

Two PBS scripts launch any of the three scenarios with one process per GPU.
Both default to `Mix-Homo-Sol`, eight A100 GPUs per node, 40 CPUs and 200 GB
of memory per node, and a 24-hour wall time:

```bash
qsub examples/elec-refact/run_single_node_multi_gpu.pbs
```

```bash
qsub examples/elec-refact/run_multi_node_multi_gpu.pbs
```

Select another scenario with a PBS environment variable:

```bash
qsub -v SCENARIO=Single-Sol \
  examples/elec-refact/run_multi_node_multi_gpu.pbs
```

Supported values are `Single-Sol`, `Mix-Homo-Sol`, and `Mix-LHCE`. Additional
configuration overlays can be passed as a colon-separated list of paths:

```bash
qsub -v SCENARIO=Mix-Homo-Sol,CONFIG_OVERLAYS=examples/elec-refact/configs/optimizer/muon.yml \
  examples/elec-refact/run_single_node_multi_gpu.pbs
```

The scripts also accept `RUN_NAME`, `OUTPUT_DIR`, `MATTERTUNE_DIR`,
`CONDA_SH`, `CONDA_ENV`, `GPUS_PER_NODE`, and (for multi-node runs)
`MASTER_PORT`. Existing data-root variables such as `SINGLE_SOL_DATA_ROOT`
and `MIX_LHCE_DATA_ROOT` are forwarded to every node. Use `DRY_RUN=1` to
print the resolved preparation and launch commands without checking GPUs,
data, or starting training.

If the resource request is changed with `qsub -l`, set `GPUS_PER_NODE` to the
number of GPUs allocated on each node. The multi-node script derives its node
count from `PBS_NODEFILE`; all nodes must see the repository, data, and output
directories through a shared filesystem.

Before distributed workers start, each script runs a serial
`--prepare-reference-only` step. This creates or validates the shared residual
energy reference once, preventing every distributed rank from fitting it.
