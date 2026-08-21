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
