# MeanFlow 多步自适应：三个去留实验

这套脚本用于回答三个顺序问题：**固定多步有没有收益；收益是否因状态而异；不看未来结果的低成本路由器能不能在新初态上获得收益。** 它不预设自适应有效，也不把更高 NFE 当作真实动作标签。

先运行 `doctor` 和 toy smoke，再准备真实 MeanFlow checkpoint、匹配的数据集与机器人仿真环境。toy 只检查代码和文件管线，不能支持论文结论。

## 1. 开始前的缺口与检查

从仓库根目录执行：

```powershell
python scripts/meanflow_experiment.py doctor
```

这不等于真实机器人仿真已经跑通。正式实验仍需要：

- **MeanFlow checkpoint**：包含训练配置 `cfg` 与 `state_dicts` 的 workspace `.ckpt`，不能只提供裸 `state_dict`。当前适配 MP1 MeanFlow 与 EquiMeanFlow 的区间平均速度接口；普通 CFM/EfficientFlow checkpoint 不会被自动当作 MeanFlow。
- **匹配的 HDF5 数据**：路径、环境 metadata、观测键、动作维度、动作归一化、绝对/相对动作约定，应与 checkpoint 一致。数据路径不是随便找一个可读文件即可。
- **运行环境**：真实 backend 需要训练时相容的 PyTorch/CUDA、Hydra/OmegaConf/dill，以及 robomimic、robosuite/MuJoCo 和所需任务注册包。MimicGen 任务需要相应注册；渲染、点云和视觉观测也必须能重现。
- **可恢复的仿真状态**：脚本会从 reset 后重放动作前缀，核对仿真状态和观测 hash；不一致时停止，而不是继续生成无效反事实标签。随机传感器、渲染差异和环境版本都可能影响这一步。

脚本入口自行引导仓库 import，不通过旧训练 workspace/旧 env runner 启动评估。尤其不要通过把 `Stack_D1` 静默替换成原生 `Stack` 来“修复”任务注册。仓库已有的 `train.py`、环境 YAML 合并冲突需要另行解决；本实验不自动覆盖这些文件。

安装包以用户指定的根目录 `policy/mp1.py` 为唯一 MP1 来源，先备份再复制加入多步采样的完整文件，不使用文本补丁。评测入口固定使用项目根目录源码；`doctor --checkpoint` 显示实际 `policy_source` 和 hash。默认一步行为、观测编码、normalizer 和训练损失保持原有实现，增加显式时间网格和共享初始噪声输入。

检查动作时序配置：

\[
\text{horizon} \ge \text{n\_obs\_steps}-1+\text{execute\_steps}.
\]

默认执行长度自动取原策略动作切片的实际长度：`min(n_action_steps, horizon-n_obs_steps+1)`。例如 `horizon=4, n_obs_steps=2, n_action_steps=4` 时实际返回 3 个动作，评测统一执行 3 个；无需另外设置。显式 `--execute-steps` 仍会校验范围。所有候选必须使用同一执行长度。预测 horizon、历史观测长度、重规划频率、最大物理步数、normalizer、模型权重与任务版本都应固定。

## 2. Toy smoke：先验证管线

```powershell
python scripts/meanflow_experiment.py run --backend toy --output data/meanflow_smoke --episodes 8 --max-steps 30
python scripts/meanflow_experiment.py analyze --input data/meanflow_smoke
```

预期检查：JSONL 可读、同初态与同噪声比较成立、anchor 重放通过、分析报告可生成。toy 的误差结构是人为构造的；即使 oracle 或路由器在 toy 上有效，也不说明真实 MeanFlow 有自适应空间。

正式数据不要与 toy 输出混合。每轮实验使用独立输出目录；不要把两次不相容运行直接追加到同一组 JSONL。

## 3. 实验一：固定多步是否改善完整闭环任务

先做一个 50 个环境初态、每个初态 3 个噪声 trial 的 pilot。下面路径需要替换成真实文件：

```powershell
python scripts/meanflow_experiment.py run --backend robomimic --checkpoint /path/model.ckpt --dataset-path /path/data.hdf5 --output data/mf_pilot --episodes 50 --trials 3 --steps 1 2 4 --schedules uniform noise_first data_first --phase all --anchor-every 5 --max-anchors 3 --device cuda:0
```

`--phase all` 同时收集完整 fixed rollout 与后面的 saved-state 分支，便于先做小规模端到端检查。

也可以分阶段：先把上面命令的 `--phase all` 改成 `--phase fixed`；确认结果有效后，用同一输出目录、模型、seeds 和时序参数运行 `--phase branches`，复用已记录的 anchors。

```powershell
python scripts/meanflow_experiment.py analyze --input data/mf_pilot
```

采样时间统一采用 **1 为噪声端、0 为数据端**。`uniform`、`noise_first`、`data_first` 是相同 K 下的不同时间网格；1/2/4 NFE 都从同一个初始 latent noise 开始，不做 best-of-noise 选择。先区分“全局更好的固定网格”与“需要依状态路由”，不要将网格改进全部归因于自适应。

先读 `report.md` 的 Fixed complete rollouts 与 Paired success differences：

- 同一个 `episode_id` 表示同环境初态和同噪声 trial；成功率差值按这些记录配对。
- `cluster_id` 是环境 seed。bootstrap 以整个环境 seed 为单位重采样，保留其全部噪声 trials；不是把 150 条 trial 当作 150 个独立初态。
- 对照成功率、任务 reward、动作变化/jerk、碰撞字段和耗时。任务 reward 不是动作 MSE，也不是“更接近高 NFE 动作”的替代指标。环境不提供可靠碰撞信号时，`null` 不等于没有碰撞。

若 2/4 步有实用收益，再看实验二。若一步已经饱和，先换有提升空间的任务或减少训练数据，不能用天花板任务判定方向无效；若一步几乎从不成功，也应先确认底座和环境配置有效。

## 4. 实验二：同状态分支是否存在可利用的异质收益

anchors 从一步基线的执行轨迹采集。每个候选恢复到相同仿真状态、相同观测历史，并使用相同当前初始噪声。**仅当前动作 chunk 使用不同 K/网格，后续默认统一使用固定一步策略，且剩余最大物理 horizon 相同。** 成功提前终止可能使实际执行长度不同。

如果固定四步明显有效，但一步后续策略会让所有分支都失败，局部 oracle 可能看不到真实空间。此时另起实验目录，增加 `--continuation-nfe 4`，让所有候选使用同一个四步后续策略复测。这个参数改变的是标签定义，不应混合两种 continuation 的分支记录；结果会写入 `branch_configuration` 和每条分支。**“一步 continuation 下局部 oracle 没收益”不能直接否定所有自适应策略。**

因此 `branches.jsonl` 中的成功是一次干预下的 counterfactual continuation outcome，可看作该共同后续策略下的 Q 样本；它不是“整条轨迹都使用自适应采样”的成功率。由于 anchors 来自一步基线，结果也只反映该状态分布，不能直接外推到路由器自己的状态分布。

分析器在每个 grid family 内保留完整的 K=1/2/4 状态配对，不跨 family 拼接候选。分别报告 1-vs-2、1-vs-4：

- `upgrade_win / upgrade_hurt / unchanged`：额外步骤分别让失败变成功、成功变失败、成功标签不变的次数。相同的均值可能隐藏完全不同的个体收益。
- **Observed binary oracle**：离线看到每个分支的成功结果后，在平均 NFE 预算下优先选择正收益升级；整数状态数严格计费，负收益不选，预算可以不用满。它是已观察二元结果的最优分配，不是通用 1/2/4 多选择背包解。
- **Matched random expectation**：恰好升级相同数量状态、与状态无关地随机选择时的解析期望。随机分配的标准差不是统计置信区间。
- **Best nonadaptive convex mixture**：全局固定 K 或与状态无关的随机混合，按同一实际成本上限比较；同时给出全 K=1/2/4 的最佳凸混合。凸混合是期望成本，oracle 是有限状态的整数分配，报告已明确区分。

报告另给出 **Best ALL-grid nonadaptive**：它还包括在相同状态上评估的其他网格，避免一个更好的固定时间表被误当作自适应空间。优先看 `oracle_gain_over_matched_all_grid_nonadaptive`；不要只与较弱的同-family 随机对照比较。

oracle 使用评估结果选择动作，是乐观的 hindsight 诊断上限。它不能部署，也不是期望 Q 的无偏最优值。少量分支的随机结果、状态筛选和成功标签噪声可能放大表观空间；需要在更多独立环境 seeds 上复核。

应比较同一组分支、同一 continuation 上的 oracle 与非自适应混合，不能与完整 fixed rollout 成功率跨分布直接比较。如果在合理 continuation 下 oracle 都没有空间，复杂路由通常缺少动机；如果 oracle 有空间，但普通信号不预测收益，问题在于“可预测性”，不是立即证明方向无效。

## 5. 实验三：训练便宜路由器，再在全新 seeds 上跑完整闭环

```powershell
python scripts/meanflow_experiment.py fit-router --input data/mf_pilot --output data/router.json --target-mean-nfe 1.5 --high-nfe 2
```

这一步学习的是额外步骤的**有符号成功收益**，而不是一步动作绝对误差。训练/holdout 按整个 `cluster_id` 切分；阈值只在训练部分选择。初始轻量路由器使用采样前可得的低维观测/历史变化与初始噪声统计，不读取专家动作、未来结果或免费 composition defect。

`--target-mean-nfe 1.5` 是训练分布上的目标预算，不是新任务分布的硬预算保证。新 seeds 上必须检查实际 `mean_nfe`、升级率及耗时。阈值不要在最终测试 seeds 上反复调整。

固定新的环境 seeds 跑完整自适应轨迹，例如从 `200000` 开始：

```powershell
python scripts/meanflow_experiment.py run --backend robomimic --checkpoint /path/model.ckpt --dataset-path /path/data.hdf5 --output data/mf_router_test --episodes 50 --trials 3 --steps 1 2 4 --schedules uniform noise_first data_first --phase router --router data/router.json --seed-start 200000 --device cuda:0
python scripts/meanflow_experiment.py analyze --input data/mf_router_test
```

模型、环境、EMA/model 权重、执行长度和采样网格必须和 pilot 一致。CLI 要求模型/路由器旁的 `.manifest.json` 绑定文件，核验 checkpoint、时间网格、`power` 和执行长度；不同 `--power` 需显式保持原值。在相同新 seeds 上自动收集固定一步、固定高步数及 `random_budget` 对照。随机路由采用与收益路由器相同的**训练目标预算**，其测试分布实际预算不保证恰好相同；应检查实际成本后画成本—成功率曲线，必要时在独立开发集校准升级概率后重测。不要拿训练 anchors 的 oracle 成功率与这次完整 rollout 直接相减。

`router_rollouts.jsonl` 才记录真实闭环路由轨迹；训练集或 branch holdout 上的路由结果仍是局部干预诊断。

最初的路由器是标准化特征上的 ridge 收益回归，只用于低成本检查“收益是否可学”。其失败不代表更强的视觉/非线性控制器一定失败；也不应将廉价模型当成可达上限。

## 6. 信号消融：variance proxy 与 composition

在训练数据上为冻结底座拟合残差方差 probe：

```powershell
python scripts/meanflow_experiment.py fit-variance --checkpoint /path/model.ckpt --dataset-path /path/data.hdf5 --output data/variance.pt --batches 200 --batch-size 64
```

随后在分支收集命令增加 `--variance-probe data/variance.pt`，使用新的输出目录或先准备该目录的 fixed/anchors。其信号名为 `variance_proxy`。

这个 probe 学到的残差混合了条件目标歧义与底座拟合误差。它不等于真实 MeanFlow 跨区间误差，也不继承 AdaFlow 的步长—离散误差定理。先检查它能否在未用于选择符号/阈值的 clusters 上预测“升级有益”，而不是仅检查它是否拟合训练残差。

`composition_defect` 比较同一 latent 的整段一步与组合两步的完整归一化预测；`action_disagreement` 比较它们实际将执行的动作段。它们在离线分支中复用已计算的候选，但**在线获得该比较需要 1+2=3 NFE**。报告中的关联性不是免费三步以内的自适应收益；如果把这些信号放进在线路由，必须把候选计算和额外延迟一并计费。

分析器只从一步行读取信号，在训练 clusters 选择信号方向，在不相交 holdout clusters 上报告 benefit AUC 与 signed-benefit Spearman。不在 holdout 调阈值。NaN/Inf 按信号剔除并报告数量；无正负标签或信号恒定时相应指标为 `null`，不是 0。`variance_proxy` 的特征提取/头部推理也有成本，日志记录的相关耗时应单独核查。

## 7. 输出文件与计时边界

| 文件 | 含义 |
|---|---|
| `fixed_rollouts.jsonl` | 固定采样候选的完整闭环结果 |
| `anchors.jsonl` | 一步基线上的恢复点、前缀和一致性校验信息 |
| `branches.jsonl` | 同状态同噪声当前 chunk 干预及共同后续策略结果 |
| `router_rollouts.jsonl` | 全新 seeds 上的真实路由闭环结果 |
| `summary.json`、`report.md` | 机器可读分析与人工阅读报告 |
| `fixed_summary.csv`、`oracle_frontiers.csv` | 固定结果与离线预算前沿 |

NFE 是模型调用成本代理，**不等于相同 wall-clock**。CUDA 推理计时前后同步，但会受预热、并发任务、设备和运行次序影响；需要记录运行条件。fixed 表的 `latency_ms_p95` 是各 episode 的 p95 再取平均，不能称为全体请求合并后的全局 p95。

`action_delta_rms`、`action_jerk_rms` 是实际发送到环境的原始命令的一阶/三阶差分 RMS，没有除以物理采样间隔；不是实测关节加速度/力矩或带 SI 单位的物理 jerk。跨动作表示时不要直接比较。`mean_nfe` 是每 episode 的平均调用次数；`total_nfe`、`decision_count`、`physical_steps` 也写入原始结果与 summary，避免成功早停掩盖整条轨迹成本差异。

仿真不会自动按测得的推理耗时推进物理时间。因此本实验没有自动证明真实机器人 deadline 可满足，也没有模拟感知陈旧、异步执行或错过控制周期带来的损失。可以先用

\[
\text{单次决策时间预算} = \frac{\text{execute\_steps}}{\text{control\_frequency}}
\]

与 p95 推理延迟比较，作为排除明显不可用方案的工程检查，并给其他系统开销留余量；这不是 deadline 保证。真实实时优势需要单独的延迟注入/异步仿真或硬件实验。

## 8. 如何决定继续、补实验或停止

建议先为具体任务声明“最小有意义改善”，再看数据。例如把成功率绝对改善 2 个百分点设为实用 margin，或把同成本 oracle 相对最佳非自适应方案 3 个百分点视作值得继续的 pilot 信号。**这些数字只是可修改的工程选择，不是科学通用阈值，也不是脚本自动判决。**

| 观察 | 下一步 |
|---|---|
| 非饱和任务上，多步平均改善很小 | 先看 win/hurt 和 oracle；正负收益可能抵消，均值不涨不能单独否定自适应 |
| 合理 continuation 下可纠正的 wins 很少，同成本 oracle 相对最强全网格固定混合也无实用空间 | 可以考虑停止该 checkpoint/任务下的自适应方向，或先改变训练目标 |
| CI 跨 0、样本少或改进不稳定 | 证据不足；不能仅凭“不显著”判死刑 |
| 固定多步有收益，但同成本 oracle 几乎不超过最佳固定混合 | 优先用更好的固定采样；状态路由的必要性较弱 |
| oracle 有空间，holdout 信号没有预测力 | 先研究可预测的增量收益与低成本特征，再扩大模型 |
| 局部分支路由有效，完整新-seed rollout 无收益 | 检查状态分布迁移、后续策略改变、预算漂移与路由错误累积 |
| 新 seeds 上完整路由在实际耗时/预算匹配后仍稳定改善 | 再扩展任务、底座规模、训练 seeds 与真实延迟评估 |

50 个环境 seeds × 3 个噪声 trials 仍只有 **50 个独立环境 clusters**。若关心 2 个百分点级别的变化，通常需要数百个甚至更多独立初态；实际数量取决于配对不一致率与方差。增加同初态噪声试验不能完全替代增加初态，pilot CI 也不能被当作高精度结论。

多卡可以用于不同且不重叠的环境 seeds、不同任务或 checkpoint，输出目录保持分离。先保证单卡的重放与配对成立，再扩展；不要把训练、信号 holdout 与最终测试的环境 seeds 混用。完整论文还应跨独立训练 seeds 验证，而不仅重测一个 checkpoint。

当前脚本与单元测试提供实验工具；真实 checkpoint 的任务表现、数据兼容性与机器人仿真结果，需要按上面的步骤实际运行后才能报告。
