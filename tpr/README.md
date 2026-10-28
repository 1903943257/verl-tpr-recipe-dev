# Tree Prefix Reuse (TPR) for VERL

TPR is a training-side prefix-reuse runtime for VERL's Megatron engine. This recipe packages the runtime separately and uses a small, version-specific patch to connect it to VERL.

## Status

The runtime was migrated to `verl_tpr/` and its package import was checked in the existing Ascend container. The integration patch and launcher in this draft still require `git apply --check`, config validation, and an end-to-end NPU run before this recipe is ready for public submission.

## Contents

- `verl_tpr/`: TPR runtime implementation.
- `verl-tpr.patch`: version-specific changes to VERL's engine configuration and Megatron execution path.
- `run_uniagent_rebench_ppo.sh`: one PPO example using UniAgent and SWE-ReBench.
- `REQUIRED_VERL.txt`: pinned VERL revision for the patch.
- `pyproject.toml`: standalone Python package metadata.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the Recipe file layout, the exact VERL integration points, the training execution flow, and the current support boundaries.

## Required VERL revision

Use the exact commit recorded in `REQUIRED_VERL.txt`. The patch is based on the upstream VERL source tree, not on the private `verl-tpr` development fork.

## Apply the patch

From the VERL source checkout:

```bash
git checkout 2ab35292039a51175a4b3d3b3b2d76e951a0d3e4
git apply --check /path/to/tpr/verl-tpr.patch
git apply /path/to/tpr/verl-tpr.patch
```

Install the runtime in the environment that already contains VERL, Megatron-LM, PyTorch, and the required Ascend dependencies:

```bash
python -m pip install -e /path/to/tpr
```

Do not install a second VERL distribution into an active experiment environment. For isolated validation, use a separate checkout/container with the same compatible image and dependencies.

## Run the example

The launcher expects a patched VERL checkout and external UniAgent/SWE-ReBench inputs. Set these paths before running:

```bash
export VERL_ROOT=/path/to/patched/verl
export UNIAGENT_ROOT=/path/to/uni-agent
export MODEL_PATH=/path/to/Qwen3-4B-128K
export TRAIN_FILE=/path/to/shortfix_candidates_task.parquet
export TASK_CONFIG=/path/to/task_config_swe_rebench_docker.yaml

# Defaults: one node, 8 NPUs, TP=2, CP=4, DP=1, N=8.
bash /path/to/tpr/run_uniagent_rebench_ppo.sh
```

The only TPR enable/disable control exposed by the recipe launcher is `TPR_ENABLED` (set it to `False` for the Full baseline). CP is configured through the existing `TRAIN_CP` / VERL `context_parallel_size`: CP=1 uses local TPR Attention and CP>1 automatically selects Ring CP. Users do not need to choose a separate TPR CP backend. Other useful overrides include `TOTAL_TRAINING_STEPS`, `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, `TRAIN_TP`, `TRAIN_CP`, and `TRAIN_SP`. The current packaged runtime requires `TRAIN_SP=False` when both TP and CP are greater than one; this will be revisited after the active SP+RingCP experiment is synchronized and revalidated. For the strict 16K+8K ReBench session-cap setup, set `SESSION_HARD_CAP=True` only when the selected UniAgent checkout implements that option.

The script does not stop Ray, modify the VERL checkout, install packages, or patch the UniAgent checkout. Run it in a dedicated container/resource allocation when another training job is active. `SESSION_HARD_CAP=True` requires a UniAgent build that implements the corresponding configuration option.

## Scope

TPR is an optional Megatron actor-training execution path. VERL continues to own PPO orchestration, rollout/reward integration, optimizer/scheduler, and distributed process groups. Experimental placement planners and unrelated profiling/test utilities are intentionally excluded from this recipe.
