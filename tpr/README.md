# Tree Prefix Reuse (TPR) for VERL

TPR is a training-side prefix-reuse runtime for VERL's Megatron engine.

## Status

This is a private development draft. The standalone runtime has been migrated to
`verl_tpr/` and its package import has been checked in the existing Ascend
container. The VERL integration patch, runnable PPO launcher, and acceptance
checks are not yet included. Do not treat this draft as a runnable recipe.

## Planned contents

- `verl_tpr/`: TPR runtime implementation.
- `verl-tpr.patch`: the minimal, version-specific VERL integration patch.
- `run_qwen3_ppo.sh`: a single example launcher using VERL's native PPO trainer.
- `REQUIRED_VERL.txt`: the exact VERL revision validated with the recipe.
- `pyproject.toml`: package metadata for the standalone runtime.

## Intended workflow

1. Check out the VERL revision recorded in `REQUIRED_VERL.txt`.
2. Apply `verl-tpr.patch` to that VERL checkout.
3. Install this directory with `pip install -e .`.
4. Set model/data/resource paths in the launcher and run `bash run_qwen3_ppo.sh`.

The patch, launcher, and tested VERL revision will be added after the current
TPR experiments and integration checks are finalized. Until then, the workflow
above describes the target interface, not a verified end-to-end installation.
