# [Recipe] 添加 Tree Prefix Reuse（TPR）训练侧前缀复用方案

## Checklist Before Starting

- [x] 本 PR 是 RFC #7561 的 Recipe 实现与运行入口，不复制完整 VERL 仓库。
- [x] TPR Runtime 已迁移到独立的 `tpr/verl_tpr/` 包；包导入已在现有 Ascend 容器中验证。
- [ ] 在干净的固定版本 VERL checkout 上通过 `git apply --check`，并完成实际 Patch 应用。
- [ ] 在该固定版本 checkout 上完成 Hydra / Engine 配置校验。
- [ ] 使用 Recipe 当前提交的 Patch、Runtime 和启动脚本完成可复现的 NPU smoke / E2E 验收。
- [ ] 清理用户接口中的 `tpr_cp_backend`，保证只有 `tpr_enabled` 是 TPR 的启用开关，并重新验证 CP 路径。

## 1. 背景与动机

本 PR 跟进 [RFC #7561 — Tree Prefix Reuse for Multi-Trajectories Training](https://github.com/verl-project/verl/issues/7561)。

在 Agentic RL 中，同一个 prompt 或中间交互状态会派生多条 PPO 轨迹。原生训练路径分别执行完整轨迹，导致完全相同的前缀在前向和反向中被重复计算。TPR 在一个逻辑 Actor mini-batch 内识别 token 完全匹配的共享前缀，将轨迹组织为树，并复用共享前缀的 Attention 计算，同时保留各轨迹独立的 PPO 目标。

本 PR 以 VERL Recipe 的形式交付 TPR Runtime、固定版本 Patch 和示例启动脚本；不将整份 VERL 源码复制进 Recipe，也不修改 PPO 算法定义。

## 2. 实现内容

- **独立 Runtime：** 在 `tpr/verl_tpr/` 中提供轨迹树构建、Segment 调度、共享 Prefix KV 状态、TPRSelfAttention、PPO objective 适配及 TPR-aware 重计算实现。
- **VERL Engine 接入：** Patch 在 `verl/workers/config/engine.py` 增加 `tpr_enabled`，并在 `verl/workers/engine/megatron/transformer_impl.py` 接入模型构建和训练执行。
- **Attention 替换：** 模型构建时通过 Megatron ModuleSpec 将原生 SelfAttention 替换为 `TPRSelfAttention`；该适配由 Recipe 的 `adapter.py` 完成，而不是直接修改 Megatron-LM / MindSpeed 核心源码。
- **训练路由：** 在原生 `prepare_micro_batches()` 之前接收完整逻辑 Actor mini-batch，构建 Forest / Segment plans，绑定 VERL 原有 PPO loss callback，完成树执行和梯度收尾后返回统一的 loss / metrics。
- **运行入口：** 提供 UniAgent + SWE-ReBench PPO 示例脚本及固定上游版本记录，便于在已有 VERL / Ascend 环境中复现。

## 3. 使用方式

Recipe 需要与 `tpr/REQUIRED_VERL.txt` 中记录的 VERL commit 配套使用。

```bash
# 在 VERL checkout 中
git checkout 2ab35292039a51175a4b3d3b3b2d76e951a0d3e4
git apply --check /path/to/verl-recipe/tpr/verl-tpr.patch
git apply /path/to/verl-recipe/tpr/verl-tpr.patch

# 在已有 VERL / Megatron / Ascend 依赖环境中
python -m pip install -e /path/to/verl-recipe/tpr
```

示例启动脚本为 `tpr/run_uniagent_rebench_ppo.sh`。运行前需配置 `VERL_ROOT`、`UNIAGENT_ROOT`、`MODEL_PATH`、`TRAIN_FILE` 和 `TASK_CONFIG`。具体参数与资源约束见 [tpr/README.md](README.md)。

## 4. 实验结果

本节只保留两个场景：**P17S17 固定形状训练侧回放**和 **P16S2 SWE-ReBench 场景**。模型为 Qwen3-4B-128K，设备为 8 张 Ascend NPU。前缀复用率按下式计算：

`prefix reuse rate = (logical tokens - unique tree tokens) / logical tokens`

其中，`logical tokens / unique tree tokens` 另列为计算量压缩比。在线 E2E、固定形状回放与训练侧 2×2 消融的测量范围和并行配置不同，不能混为同一条性能数字。


### UniAgent + SWE-ReBench

任务场景为 SWE-ReBench 的 `ASPP__pelita-875`。Prefix=16,384 tokens，Suffix 上限=2,048 tokens，N=8。在线 E2E 使用 TP2 × CP4 × DP1，未开启重计算。

#### 轨迹复用统计

| 指标 | 结果 |
|---|---:|
| 逻辑 token 数 | 147,456 |
| 树中唯一 token 数 | 34,210 |
| **前缀复用率** | **76.8%** |
| 逻辑 token / 唯一 token | **4.31×** |

#### 在线 PPO E2E（Full / TPR 各运行 5 步）

| 指标 | Full | TPR | 观测结果 |
|---|---:|---:|---:|
| 总墙钟时间 | 2,720 s | 2,434 s | 1.118× 时间比 |
| Actor update 累计时间 | 196.7 s | 151.0 s | 1.303× 时间比 |
| Rollout / generation 累计时间 | 2,162.0 s | 1,923.6 s | 1.124× 时间比 |
| 峰值显存（allocated） | 11.24 GiB | 11.64 GiB | TPR 高 0.40 GiB |
| 峰值显存（reserved） | 12.31 GiB | 12.31 GiB | 相同 |


#### 离线训练侧回放：Full / TPR × Recompute OFF / ON

该实验使用同一 P16S2 场景捕获的 5 个 Actor mini-batch，TP8 × CP1 × DP1，不包含 rollout。下面报告纯前反向时间与完整训练侧时间。完整训练侧时间包含 F/B、gradient finalization、global gradient norm，以及 optimizer update / copy-back。

| 模式 | Recompute | 状态 | 完成步数 | 纯 F/B 累计时间 | 完整训练侧累计时间 | 峰值显存（allocated） |
|---|---|---|---:|---:|---:|---:|
| Full | OFF | OOM | 0/5 | — | — | — |
| TPR | OFF | PASS | 5/5 | 29.453 s | 110.572 s | 23.794 GiB |
| Full | ON | PASS | 5/5 | 76.742 s | 148.637 s | 8.801 GiB |
| TPR | ON | PASS | 5/5 | 36.751 s | 110.162 s | 7.448 GiB |

在 **Recompute=ON** 的对比结果：

| 指标 | Full + Recompute | TPR + Recompute | 加速 / 显存变化 |
|---|---:|---:|---:|
| 纯 F/B 累计时间 | 76.742 s | 36.751 s | **2.088×** |
| 完整训练侧累计时间 | 148.637 s | 110.162 s | **1.349×** |
| 峰值显存（allocated） | 8.801 GiB | 7.448 GiB | **降低约 15.4%** |

2×2 消融还显示，TPR 关闭重计算时 F/B 为 29.453 s，开启重计算后为 36.751 s；但峰值显存从 23.794 GiB 降至 7.448 GiB，而完整训练侧时间分别为 110.572 s 和 110.162 s。Full 关闭重计算会 OOM，因此该单元格没有可比较的耗时数据。

**优化器与复用率说明：** 这次离线回放使用持久化 CPU FP32-master AdamW（`β1=0.9, β2=0.95, eps=1e-8, weight_decay=0`）作为受控的 optimizer/update 回放路径，并非生产环境原生 Megatron optimizer 路径。因此 1.349× 是该训练侧回放设置的观测值，不应解释为生产优化器下的完整训练吞吐收益。76.8% 的复用率来自 P16S2 场景轨迹统计；回放摘要没有对 5 个捕获 mini-batch 另行汇总 unique-token 计数，所以这里不声称 2×2 回放对这 5 个 mini-batch 独立测得同一个复用率。

## 5. 验证状态与已知限制

- 当前 Recipe 包导入已在现有 Ascend 容器中通过。
- 上述实验来自现有开发工作树中的实验运行；在公开提交前，仍需确认 Recipe 当前提交的 Patch、Runtime 与启动脚本在固定上游 revision 上一致，并完成干净 checkout 验收。
- 在线 P16S2 E2E 目前存在训练信号为零的问题，不能作为有效训练收益的最终证明。
- 训练侧回放使用 CPU FP32-master AdamW，不代表生产 Megatron optimizer 路径。
- 当前实现只支持经过验证的模型和并行组合；超出支持范围应显式失败，不应宣称普遍支持。

## 6. 相关讨论

- [RFC #7561 — Tree Prefix Reuse for Multi-Trajectories Training](https://github.com/verl-project/verl/issues/7561)
- [VERL PR #8300 — Shared-prefix packing for the FSDP engine](https://github.com/verl-project/verl/pull/8300)
- [Schedule-Level Shared-Prefix Reuse for LLM RL Training (arXiv:2606.01143)](https://arxiv.org/abs/2606.01143)

