# EfficientAdaFlow 运行与反馈

这是可迭代的研究版本，尚无机器人成功率结果。[研究说明](EfficientAdaFlow_research_zh.md)包含公式、文献、新颖性边界和完整失败排查表。

## 入口与环境

| 文件 | 用途 |
| --- | --- |
| `policy/EfficientAdaFlow.py` | joint/head_only 训练，固定/自适应采样和候选连续性 |
| `model/equi/invariant_variance_head.py` | 群轴一阶、二阶矩池化的不变残差头 |
| `common/adaptive_flow.py` | 独立候选步长，完成后移出批次，积分区间为 [0,1] |
| `config/EfficientAdaFlow.yaml` | 继承 EfficientFlow 的图像策略配置 |
| `train_efficientadaflow.py` | 独立的图像策略入口，避免旧服务器 `train.py` 固定调用 MP1 |
| `common/train_entry.py` | 共享解析器、训练器选择、入口和配置检查 |
| `scripts/efficientadaflow_experiment.py` | checkpoint 采样预算扫描与验证集残差校准 |

原 `policy/EfficientFlow.py` 保留。新模型沿用其编码器、动作表示和速度网络。训练入口及环境 YAML 的文件内冲突已清理；没有 stage 文件或修改用户原有 Git 合并索引。

建议在已有 Linux/NVIDIA 训练环境运行。`conda_environment.yaml` 保留原 Linux 依赖，不是验证过的 Windows 安装方案。需要仓库指定的 escnn、pytorch3d、robomimic、robosuite、MimicGen 等依赖。MimicGen 必须注册数据元信息中的真实环境，不能把 Stack_D1 映射到另一种 Stack 环境绕过错误。

本地可用的 EfficientFlow conda 环境有 PyTorch 2.1 CPU，但缺少 escnn/robomimic/pytorch3d 和真实训练数据。其旧 PyTorch 与已安装的新 transformers 还可能在 optimizer 初始化时触发 pytree API 不兼容。CPU 测试通过不表示该环境已可训练；请使用一致的原项目环境，先执行下面的真实 smoke。

数据沿用原仓库 absolute-action HDF5，例如 `data/robomimic/datasets/square_d2/square_d2_abs.hdf5`。所有命令在仓库根目录运行，输出目录使用新名字。

## 先做接口检查

```bash
python train_efficientadaflow.py --verify task_name=square_d2 n_demo=100
python train_efficientadaflow.py --cfg job task_name=square_d2 n_demo=100
python -m unittest discover -s tests -p 'test_adaflow_*.py' -v
python scripts/efficientadaflow_experiment.py --help
```

CPU 测试覆盖数值和策略接口，其中部分策略测试使用视觉依赖替身。下面才是在真实依赖/数据上的 forward、backward、checkpoint 小规模验证：

```bash
python train_efficientadaflow.py task_name=square_d2 n_demo=100 training.num_epochs=2 training.max_train_steps=3 training.max_val_steps=2 training.rollout_every=0 training.sample_every=1 training.checkpoint_every=1 policy.variance_warmup_steps=0 dataloader.batch_size=2 val_dataloader.batch_size=2 dataloader.num_workers=0 val_dataloader.num_workers=0 dataloader.persistent_workers=false val_dataloader.persistent_workers=false hydra.run.dir=data/outputs/adaflow_smoke
```

这只检查流水线，不能用其模型质量判断方向。显存不足先减 batch，正式对照保持相同有效 batch 和训练预算。

## 从头联合训练

先选有提升空间的任务做 50 epoch pilot，检查损失和校准，再扩大完整预算。Stack 可检查流水线，但原方法一步成绩可能已饱和。

```bash
python train_efficientadaflow.py task_name=square_d2 n_demo=100 training.seed=42 training.num_epochs=50 training.rollout_every=5 training.checkpoint_every=5 hydra.run.dir=data/outputs/adaflow_square_d2_pilot_s42
```

正式训练去掉 pilot 的轮数/频率覆盖，默认 n_demo=100 为 500 epoch。建议至少 3 个训练 seed，再扩展 threading_d2 和 three_piece_assembly_d2。

必做同预算的 `policy.fabo_weight=0` 消融。默认方差头在 1000 个训练 minibatch 后开始更新；`train/variance_ready` 和 checkpoint 中 `_variance_updates` 记录状态。这是 minibatch 次数，不是 optimizer 更新次数。

## 已有 EfficientFlow checkpoint 时先只训练头

此路线更快检验固定速度场上的自适应预算。encoder hidden dim、down_dims 等架构必须匹配来源 checkpoint。

```bash
python train_efficientadaflow.py task_name=square_d2 n_demo=100 policy.variance_training=head_only policy.variance_loss_weight=1.0 training.init_checkpoint=/absolute/path/to/efficientflow.ckpt training.init_use_ema=true training.num_epochs=20 training.lr_warmup_steps=50 training.rollout_every=0 optimizer.lr=0.001 hydra.run.dir=data/outputs/adaflow_square_d2_head_s42
```

init_checkpoint 只加载权重，不继承优化器或训练轮数；优先使用来源 EMA，可关闭。仅允许缺少新残差头和计数器，其他缺失会报错。head_only 冻结编码器和速度网络，并置为 eval，避免 BatchNorm 漂移；自动取消 warmup。此模式 loss 仍乘 variance_loss_weight。

暖启动与续训默认保留 checkpoint 中保存的 normalizer；head_only 不能通过重新拟合归一化改变冻结策略的坐标系。加载和 EMA 评估会重建 escnn 的派生卷积/线性缓存，防止参数已更新但推理仍使用旧 filter。

不要用未训练的零初始化残差头判断性能。先得到训练过头的新 EfficientAdaFlow checkpoint。

## 同 checkpoint 评估与校准

先固定候选数 1 扫描，再将 candidates 改成 5 重复；避免混淆多候选开销与自适应收益。参数见各子命令 --help。

```bash
python scripts/efficientadaflow_experiment.py sweep --checkpoint /absolute/path/to/new/checkpoints/latest.ckpt --output data/adaflow_eval_c1 --device cuda:0 --candidates 1 --fixed-steps 1 2 3 5 8 --etas 0.05 0.1 0.2 --seeds 0 1 2 --n-test 50 --n-envs 1
python scripts/efficientadaflow_experiment.py calibrate --checkpoint /absolute/path/to/new/checkpoints/latest.ckpt --output data/adaflow_calibration --device cuda:0 --max-batches 20
```

默认选择新 checkpoint 的 EMA 权重。输出含 checkpoint 哈希和配置。eta/variance_scale 应在验证数据或预留环境选择，最终测试集只评估锁定配置。汇总保留训练 seed 层级，不把同一模型的大量 rollout 当成独立训练实验。

可用 `--weights model` 显式评估非 EMA 权重，或用 `--dataset-path /new/path/data_abs.hdf5` 搬迁数据位置。`--variance-scale 2.0` 可在相同 checkpoint 上测试残差尺度校准，不需要重新训练；它乘的是方差 q，因此步长中的标准差乘 sqrt(2)。实际使用值会进入 manifest。

n_envs=1 接近在线控制延迟，再单独测部署并行 batch。NFE 统计覆盖所有候选；model_calls_per_prediction 是批次网络调用次数，candidate_evaluations_per_prediction 是每个环境决策的候选评估总量。批次最慢候选可能决定墙钟时间，均值 NFE 下降不等于加速。计时同步 CUDA，包含编码、步长控制、候选筛选和指标采集，属于当前实现的总策略时延。

## 第一轮反馈

请提供评估/校准产物、训练 logs.json.txt、完整命令及异常堆栈。负结果和失败视频也有用。

| 项目 | 需要的内容 |
| --- | --- |
| 设置 | task、n_demo、训练 seed、checkpoint epoch、GPU、batch、候选数、joint/head_only |
| 固定采样 | 同 checkpoint 的 1/2/3/5/8 步成功率与 P50/P95 时延 |
| 自适应 | 各 eta 成功率、NFE mean/P95、一步比例、批次调用数、P50/P95 时延 |
| 训练 | flow_mse、fabo、variance_nll、residual_mse、predicted_variance、variance_ratio 曲线 |
| 校准 | 总体/分时间/分预测方差桶残差比值，clamp/fallback 比例 |
| 行为 | 失败阶段、模式跳变/抖动、接触/插入失败和视频 |

排查顺序：固定 8 步也差，优先检查数据、速度场与 FABO 强度；固定多步好但自适应差，检查头的校准和时间分布，优先尝试 head_only；几乎用满预算，检查 eta 尺度、偏差和方差高估；NFE 降但时延不降，检查批次慢候选和同步开销；质量提升却多样性下降，检查 FABO 权重、候选数与 reset 周期。

一次实验用于定位下一次修改，不能单独证明方向成功或失败。

## `dataset must be BaseDataset` 错误

若报错来自数百行的服务器 `train.py`，检查它是否仍内嵌并固定调用 MP1 训练循环。EfficientAdaFlow 的图像数据集继承 `BaseImageDataset`；MP1 使用的 `BaseDataset` 是另一个并列的基类，不能直接用其训练循环。不能只删掉断言或更改数据集继承来掩盖训练器错配。

完整同步后，使用上面的独立入口，先看 `--verify` 的入口路径和目标，再启动训练。它不会导入旧 `train.py`，即使旧嵌套 `EfficientFlow/` 目录仍存在，也会优先使用新入口旁的平铺源码。`TrainEquiWorkspace` 还会在读取图像/创建缓存之前检查数据集类，配置不匹配时立即报错。此次问题无需删除刚生成的数据缓存。
