# [Recipe] Add Tree Prefix Reuse (TPR) for Training-Side Prefix Reuse

## Checklist Before Starting

- This PR provides the Recipe implementation and execution entry point for RFC #7561.
- The TPR Runtime has been moved to the standalone `tpr/verl_tpr/` package. Package imports have been verified in the existing Ascend container.
- Verify `git apply --check` and successfully apply the patch in a clean checkout of a pinned VERL revision.
- Validate the Hydra / Engine configuration against the pinned revision.
- Complete a reproducible NPU smoke test / E2E validation using the patch, runtime, and launch scripts included in the current Recipe commit.

## 1. Background and Motivation

This PR implements [RFC #7561 — Tree Prefix Reuse for Multi-Trajectories Training](https://github.com/verl-project/verl/issues/7561).

In Agentic RL, the same prompt or intermediate interaction state can give rise to multiple PPO trajectories. The native training path processes each trajectory independently, repeatedly computing identical prefixes during both forward and backward passes. TPR identifies shared prefixes with exactly matching tokens within a logical Actor mini-batch, organizes the trajectories into a tree, and reuses the shared-prefix Attention computation while preserving the independent PPO objectives of individual trajectories.

This PR delivers TPR as a VERL Recipe, including the TPR Runtime, a patch targeting a pinned upstream revision, and example launch scripts. It does not duplicate the entire VERL source tree in the Recipe or modify the PPO algorithm definition.

## 2. Implementation

- **Standalone Runtime:** Provides trajectory tree construction, Segment scheduling, shared Prefix KV state management, `TPRSelfAttention`, PPO objective adaptation, and TPR-aware recomputation under `tpr/verl_tpr/`.
- **VERL Engine integration:** The patch adds `tpr_enabled` to `verl/workers/config/engine.py` and integrates TPR into model construction and training execution in `verl/workers/engine/megatron/transformer_impl.py`.
- **Attention replacement:** During model construction, the native SelfAttention module is replaced with `TPRSelfAttention` through the Megatron ModuleSpec mechanism. This adaptation is implemented in the Recipe's `adapter.py`, without directly modifying the core Megatron-LM / MindSpeed source code.
- **Training execution routing:** Before the native `prepare_micro_batches()` call, TPR receives the complete logical Actor mini-batch, constructs Forest / Segment plans, binds the existing VERL PPO loss callback, executes the tree, and finalizes gradients before returning unified loss and metrics.
- **Execution entry point:** Provides a UniAgent + SWE-ReBench PPO example script and a record of the pinned upstream version to facilitate reproducibility in existing VERL / Ascend environments.

## 3. Usage

The Recipe must be used with the VERL commit recorded in `tpr/REQUIRED_VERL.txt`.

```bash
# Inside the VERL checkout
git checkout 2ab35292039a51175a4b3d3b3b2d76e951a0d3e4
git apply --check /path/to/verl-recipe/tpr/verl-tpr.patch
git apply /path/to/verl-recipe/tpr/verl-tpr.patch

# In an existing VERL / Megatron / Ascend dependency environment
python -m pip install -e /path/to/verl-recipe/tpr
```

The example launch script is `tpr/run_uniagent_rebench_ppo.sh`. Before running it, configure `VERL_ROOT`, `UNIAGENT_ROOT`, `MODEL_PATH`, `TRAIN_FILE`, and `TASK_CONFIG`. See [tpr/README.md](README.md) for detailed parameters and resource requirements.

## 4. Experimental Results

The model is Qwen3-4B-128K, running on 8 Ascend NPUs. The prefix reuse rate is calculated as follows:

`prefix reuse rate = (logical tokens - unique tree tokens) / logical tokens`

The ratio of logical tokens to unique tree tokens is also reported as a measure of computation reduction. The online E2E run, fixed-shape replay, and training-side ablation use different measurement scopes and parallel configurations.

### UniAgent + SWE-ReBench

The task is `ASPP__pelita-875` from SWE-ReBench. Each trajectory is approximately 20K tokens in total, with a prefix reuse rate of approximately 76.8%. The online E2E experiment uses TP2 × CP4 × DP1, with recomputation disabled.

#### Trajectory Reuse Statistics

| Metric                         | Result    |
| ------------------------------ | --------- |
| Logical tokens                 | 147,456   |
| Unique tokens in the tree      | 34,210    |
| **Prefix reuse rate**          | **76.8%** |
| Logical tokens / unique tokens | **4.31×** |

#### Online PPO E2E (5 Steps Each for Full and TPR)

| Metric                               | Full      | TPR       | Observed Result        |
| ------------------------------------ | --------- | --------- | ---------------------- |
| Total wall-clock time                | 2,720 s   | 2,434 s   | 1.118× speedup         |
| Cumulative Actor update time         | 196.7 s   | 151.0 s   | 1.303× speedup         |
| Cumulative rollout / generation time | 2,162.0 s | 1,923.6 s | 1.124× speedup         |
| Peak allocated memory                | 11.24 GiB | 11.64 GiB | TPR uses 0.40 GiB more |
| Peak reserved memory                 | 12.31 GiB | 12.31 GiB | Identical              |

#### Offline Training-Side Replay: Full / TPR × Recompute OFF / ON

This experiment uses the same five Actor mini-batches from the above scenario, with TP8 × CP1 × DP1, excluding rollout. The results report both pure forward/backward (F/B) time and total training-side time.

Total training-side time includes F/B, gradient finalization, global gradient norm computation, and optimizer update / copy-back.

| Mode | Recompute | Status | Completed Steps | Cumulative Pure F/B Time | Cumulative Total Training-Side Time | Peak Allocated Memory |
| ---- | --------- | ------ | --------------- | ------------------------ | ----------------------------------- | --------------------- |
| Full | OFF       | OOM    | 0/5             | —                        | —                                   | —                     |
| TPR  | OFF       | PASS   | 5/5             | 29.453 s                 | 110.572 s                           | 23.794 GiB            |
| Full | ON        | PASS   | 5/5             | 76.742 s                 | 148.637 s                           | 8.801 GiB             |
| TPR  | ON        | PASS   | 5/5             | 36.751 s                 | 110.162 s                           | 7.448 GiB             |

For the **Recompute = ON** comparison:

| Metric                              | Full + Recompute | TPR + Recompute | Speedup / Memory Change       |
| ----------------------------------- | ---------------- | --------------- | ----------------------------- |
| Cumulative pure F/B time            | 76.742 s         | 36.751 s        | **2.088× speedup**            |
| Cumulative total training-side time | 148.637 s        | 110.162 s       | **1.349× speedup**            |
| Peak allocated memory               | 8.801 GiB        | 7.448 GiB       | **Approximately 15.4% lower** |

With recomputation disabled, TPR completes F/B in 29.453 s. Enabling recomputation increases this to 36.751 s, but reduces peak allocated memory from 23.794 GiB to 7.448 GiB. Total training-side time is 110.572 s without recomputation and 110.162 s with recomputation. Full with recomputation disabled runs out of memory (OOM), so no comparable timing result is available for that configuration.

## 5. Validation Status and Known Limitations

- Package imports for the current Recipe have passed validation in the existing Ascend container.
- The experimental results above were obtained from runs in the existing development worktree. Before public submission, the patch, runtime, and launch scripts in the current Recipe commit must be verified to be consistent on the pinned upstream revision, followed by validation in a clean checkout.
- The current implementation supports only the verified combinations of TP, DP, and CP parallelism. Unsupported configurations fail explicitly.
- Hybrid models are not yet supported and are planned for future work.

## 6. Related Discussions

- [RFC #7561 — Tree Prefix Reuse for Multi-Trajectories Training](https://github.com/verl-project/verl/issues/7561)
- [VERL PR #8300 — Shared-prefix packing for the FSDP engine](https://github.com/verl-project/verl/pull/8300)