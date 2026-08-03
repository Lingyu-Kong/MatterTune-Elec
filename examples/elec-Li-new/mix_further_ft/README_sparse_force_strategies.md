# Mix Further Fine-Tune Strategies

These scripts are optional entry points layered on top of `train.sh`. The
default `train.sh` behavior remains full conservative energy+force training.
All strategies write separate output scopes, but by default reuse the shared
`bulk_interface` residual reference through `REFERENCE_ROOT`.

- `train_A_energy_only.sh`: energy-only further finetuning. This is fast and
  omits force prediction during training.
- `train_B_force_polish_from_A.sh`: full conservative energy+force polishing
  initialized from the latest A checkpoint, or from `ENERGY_ONLY_CKPT`.
- `train_C_sparse_force_every_n.sh`: train energy every step and train forces
  every `FORCE_EVERY_N_STEPS` steps. Default: 5.
- `train_D_force_subset.sh`: randomly mark a fraction of structures for force
  training and train energy on all structures. Default force fraction: 0.2.
- `train_E_freeze_backbone.sh`: full energy+force training with backbone
  freezing enabled.

Common overrides:

```bash
REFIT_REFERENCE=0 DEVICES=0,1,2 BATCH_SIZE=1 bash train_C_sparse_force_every_n.sh
FORCE_EVERY_N_STEPS=10 bash train_C_sparse_force_every_n.sh
FORCE_SUBSET_FRACTION=0.1 FORCE_SUBSET_SEED=7 bash train_D_force_subset.sh
ENERGY_ONLY_CKPT=/path/to/A-best.ckpt bash train_B_force_polish_from_A.sh
```

Energy-only checkpoints are useful as warm starts, but the production checkpoint
should normally come from B, C, D, E, or the default full conservative training
path if downstream applications require forces from the model calculator.
