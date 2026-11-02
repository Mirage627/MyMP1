# Kitchen D1：现有 checkpoint 的采样对照

本轮不训练模型。使用同一份 EfficientAdaFlow checkpoint 和保存的 normalizer，对比固定步数与自适应采样，定位“后期 NFE 接近 1”是否使自适应调度失去收益。这是同权重的采样消融；完整的新旧方法比较还需要原始 EfficientFlow 的独立 checkpoint 和多训练 seed。

## 上传文件

本地入口是 `/path/to/EfficientFlow\run_kitchen_adaflow_ablation.py`，评估实现是 `/path/to/EfficientFlow\scripts\efficientadaflow_experiment.py`。

在本机 PowerShell 执行，沿用服务器外层入口、内层源码的布局：

```powershell
scp "/path/to/EfficientFlow\run_kitchen_adaflow_ablation.py" YOUR_USER@YOUR_SERVER:/home/YOUR_USER/PROJECT_EfficientFlow/run_kitchen_adaflow_ablation.py
scp "/path/to/EfficientFlow\scripts\efficientadaflow_experiment.py" YOUR_USER@YOUR_SERVER:/home/YOUR_USER/PROJECT_EfficientFlow/EfficientFlow/scripts/efficientadaflow_experiment.py
```

实验复用已用于训练的 `policy/`、`model/`、`common/` 和 `env_runner/`。本次没有修改这些模型或训练文件。`common/train_entry.py`、`common/equivariant_checkpoint.py` 必须已与当前训练代码一起存在；入口会检查所选源码位置并打印路径。不要为评估重新训练或替换模型结构。

## 运行

在服务器激活原训练环境，进入外层项目目录：

```bash
cd /home/YOUR_USER/PROJECT_EfficientFlow
python -u run_kitchen_adaflow_ablation.py --mode smoke --dry-run
python -u run_kitchen_adaflow_ablation.py --mode smoke
```

脚本已写入默认 checkpoint 目录：

```text
/home/YOUR_USER/PROJECT_EfficientFlow/data/outputs/2026.10.23/00.36.16_EfficientAdaFlow_demo200_kitchen_d1/checkpoints
```

优先选择 `latest.ckpt`；不存在时选择文件名 epoch 最大的一份。如果最高 epoch 有多份文件或文件名无法辨识，停止并要求通过 `--checkpoint /absolute/path/file.ckpt` 指定。每次启动打印所选文件，manifest 保存 SHA-256。默认显式使用 `ema_model`，缺失时停止，不会悄悄改用普通模型。

`smoke` 只跑固定一步与自适应 eta=0.1，每种 2 回合，总计 4 回合。先检查 checkpoint、normalizer、依赖和 HDF5 元信息，再开始真实 rollout。它用于检查流水线，不能用 2 回合判断成功率。

通过后运行正式诊断：

```bash
python -u run_kitchen_adaflow_ablation.py --mode pilot
```

| 设置 | 默认值 |
| --- | --- |
| 固定采样 | 1、2、4、8 步 |
| 自适应 eta | 0.1、0.05、0.02、0.01 |
| 自适应最大步数 | 8 |
| 每次生成的候选数 | 5，与现有配置一致 |
| 并行环境数 | 1，便于比较在线策略预测耗时 |
| 每种设置评测回合 | 50 |
| 环境 seed | 每种设置都用 200000～200049 |
| 策略随机数初始化 seed | 每种设置都重置为 0 |
| 总量 | 8 种设置 × 50 回合 = 400 回合 |

200000 起的新 seed 组用于本轮诊断/参数选择，避免反复在训练时已查看的 100000 组上调参。环境初始化 seed 一致；闭环轨迹变化时，不保证不同方法逐时刻使用的策略噪声严格配对。若挑选出 eta，最终结论需在锁定设置后换独立 seed 组复验。`--seeds` 表示同一训练模型的评估重复，不是新训练 seed。

脚本默认保留 checkpoint 内的数据路径、normalizer 和 variance_scale，不自动做残差校准。仅在数据搬迁时显式添加：

```bash
python -u run_kitchen_adaflow_ablation.py --mode smoke --dataset-path /absolute/path/kitchen_d1_abs.hdf5
```

若只想补充一个阈值、一个候选数设置，允许覆盖网格；例如以下是 1/2/4 步加 eta=0.05/0.02，每种 50 回合：

```bash
python -u run_kitchen_adaflow_ablation.py --mode pilot --fixed-steps 1 2 4 --etas 0.05 0.02
```

不需要运行 W&B 上传。结果逐变体保存为本地文件。EGL 环境变量仅在未设置时使用默认值；继续使用服务器已有的 MuJoCo 配置。

## 结果位置与反馈

每次运行创建新的目录：

```text
/home/YOUR_USER/PROJECT_EfficientFlow/data/diagnostics/kitchen_adaflow_pilot_<时间>/
  manifest.json
  results.jsonl
  summary.csv
  report.md
  rollouts/
```

每完成一种设置就更新一次表格；中断前已完成的记录会保留，重新启动会创建新目录，不覆盖旧结果。默认不录视频，需要时加 `--video`。

先查看 `report.md` 或 `summary.csv`。反馈整个结果目录，至少包括上述四个文件。`results.jsonl` 保留各测试环境的 reward，便于检查具体哪几个回合发生改善。奖励为 0/1 时才将均值称为成功率，汇总表自动检查并记录成功次数。

判断顺序：

1. 固定 2/4/8 步能否超过固定一步？不能提高时，当前任务/模型可能缺少增加积分的收益空间。
2. 固定多步提高、eta=0.1 自适应仍一步时，检查更小 eta 是否触发有效计算。
3. 自适应能否在相近成功率下减少耗时，或在相近耗时下提高成功率？这才能支持按需分配计算的价值。

时延是整个 `predict_action` 调用，包括编码器、残差头/控制器、候选筛选和指标采集，不包括环境仿真和观测 CPU→GPU 传输；没有排除首次调用。本轮各设置统一 n_envs=1，不能把毫秒值直接与原训练图的并行 batch 时延比较。Wilson 区间仅描述二值回合结果，不代表跨训练 seed 的不确定性。

## 本地验证范围

本地验证覆盖参数计划、两层目录发现、checkpoint 文件选择、spawn 子进程包发现、真实评测结果计数与报表。服务器权重和 MuJoCo/GPU 环境不在本机，真实加载和 rollout 由服务器上的 doctor 与 smoke 完成。
