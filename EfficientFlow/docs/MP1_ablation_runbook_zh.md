# MP1 / MGAD / H 六组消融运行说明

2026-11-01：已修复带相机观测时训练/推理归一化不一致的问题，并加入录像数量检查与旧 checkpoint 独立评估工具。旧实验全零排查、修复部署和离线同步命令见 [本次修复说明](MP1_zero_score_fix_20261101_zh.md)。

独立入口是 `train_mp1_ablation.py`，默认配置为 `MP1_baseline`。六组共用新的 `TrainMP1AblationWorkspace`，不导入旧 `train.py` 的内嵌训练器，也不使用旧 `train_mp1_workspace.py` 中覆盖 epoch 数的循环。新入口使用 `training.num_epochs`，不会按旧逻辑强制改成 1005。

公式与 PDF 纠错以 [MP1 理论审核](mp1_theory_audit_zh.md) 为准。这里的 H 指用户 HTML 中“对应修改 H”的算法：学习区间残差二阶矩，采用分离梯度和有界矩阵度量；SUREFlow 仅作为代码结构参考，不能用其联合 NLL 或终点修正替代 H。默认超参数仅供可重复运行，尚未经过任务实证选择，也没有成功率提升保证。

## 六个配置与英文源文件

| 配置名称 | policy 文件 | MGAD | 训练统计头 |
| --- | --- | --- | --- |
| `MP1_baseline` | `policy/mp1_baseline.py` | 关 | 无 |
| `MP1_mgad` | `policy/mp1_mgad.py` | 开 | 无 |
| `MP1_diag` | `policy/mp1_diag.py` | 关 | 对角二阶矩 |
| `MP1_full` | `policy/mp1_full.py` | 关 | 完整二阶矩 |
| `MP1_mgad_diag` | `policy/mp1_mgad_diag.py` | 开 | 对角二阶矩 |
| `MP1_mgad_full` | `policy/mp1_mgad_full.py` | 开 | 完整二阶矩 |

六个 YAML 位于 `config/`，各自包含完整训练配置，仅通过 `task: mimicgen_pc_abs` 复用观察、数据集和评估环境定义，不再继承 `MP1.yaml` 或公共训练 base。`MP1_ablation_base.yaml` 仅保留为指向新版 baseline 的兼容入口。共享训练逻辑在 `policy/mp1_ablation.py`；MGAD 在 `model/meanflow_mgad.py`；统计头、NLL 和矩阵权重在 `common/meanflow_statistics.py`。

## 两档实验必须分别比较

| 共同设置 | `profile=mp1`（默认） | `profile=theory` |
| --- | --- | --- |
| 实际标签的混合系数 w | 2 | 1 |
| dispersive loss 系数 | 0.5 | 0 |
| 无统计头两组的回归损失 | 原 adaptive L2 | 普通 MSE |
| 四个统计头组的回归损失 | H 有界矩阵平方损失 | H 有界矩阵平方损失 |
| 统计头目标 | 实际 guided 标签 `y_w` 的残差二阶矩 | 严格 H 标签的残差二阶矩 |

`mp1` 档保留原底座的 w 和辅助损失，但头组会**替换** adaptive L2，不把两种权重相乘。`theory` 档是与修正后 H 公式对齐的隔离实验。每一档内部固定数据划分、训练 seed、有效 batch、epoch/更新预算、观察编码器、动作表示、评估环境与 1 NFE；不要把跨档的差异全部归因于统计头或 MGAD。

默认 `mp1` 档中，“baseline 对比 full/diag”同时包含 adaptive L2 到平方矩阵回归的改变。为区分这项改变，应额外跑同档的 `MP1_baseline policy.baseline_loss=mse`；比较 MGAD 组时也加 `MP1_mgad policy.baseline_loss=mse`。`theory` 档无头组已经是 MSE，不需要这个额外覆盖。完整相关结构的收益用 full 对比 diag；MGAD 的收益分别在 none、diag、full 三种匹配条件下比较。

## 统计对象与计算预算

动作 chunk 按固定次序展平，`D=horizon*action_dim`。完整头输出 `D(D+1)/2` 个下三角参数，构造正定矩阵；对角头输出 D 个数。默认任务 action_dim=10、horizon=4，因此完整矩阵为 40×40，因子输出为 820 个数；任务或 horizon 改变时须重新计算。

头拟合的是 `M_w=E[e_w e_w^T | z]`，不是自动等于纯标签协方差。它还含均值偏差外积；固定 teacher 时，`w=2` 标签的协方差包含 `w^2=4` 的尺度。完整解释及梯度归属见 [审核第 3、4 节](mp1_theory_audit_zh.md#3-h-的执行契约)。头只能读取允许的条件/已 detach 的特征，标签与当前残差只用于监督。

默认推理只运行平均速度网络，统计头不执行；保留一步 `1→0`，不添加终点 refinement。MGAD 启用时包含额外 attention、投影与 FFN 算子，仍为 1 NFE，但不表示时延、显存或参数量与 baseline 相同。完整矩阵头增加训练成本；应同时报告训练时间、显存、分解失败与成功率。实现遇到非有限值或分解失败会明确报错，不静默丢弃样本或回退。

## MGAD 的适配理由与边界

参考 [SUREFlow 官方 ActionQueryDecoder](https://github.com/tanvirnwu/SUREFlow_IROS_2026/blob/main/SUREFlow/policy/policy.py) 的“自注意力 → 对观察 memory 的交叉注意力 → FFN”残差结构，接到 MP1 U-Net 最后解码特征与动作投影层之间。每个动作位置加入可学习 query；原 MP1 的逐观察编码结果恢复为 memory tokens，加观察时间位置编码，并使用 MeanFlow 的 t、r 两个端点条件。

其合理性是给动作 chunk 增加显式跨动作交互和直接读取观察特征的路径；H 则改变训练残差的方向度量，两者作用位置不同，因此组合值得做消融。MP1 原有 U-Net/FiLM 已经能融合上下文，MGAD 也可能冗余或过拟合，是否有效需要 none/diag/full 三组匹配对照。这里的 memory 是每个观察时刻的池化点云与状态特征，默认只有两个观察 token；并没有声称恢复 SUREFlow 的图像 patch 或单个点级 memory。

实现使用显式缩放点积注意力并关闭 dropout，保证 MeanFlow 输入 JVP 和反向传播可用；未直接依赖可能缺少高阶导数支持的融合 attention kernel。原 FiLM 的时间条件保留，本次没有额外加入 FCM。统计头读取经过 MGAD 的同一解码特征后 detach，因此不会通过头损失暗中训练 MGAD。

## 先检查配置和源码

命令从含 `data/` 的项目工作目录运行。入口支持两种同步布局：入口与 `config/`、`policy/` 等平铺；或入口旁有一个完整 `EfficientFlow/` 源包。完整平铺源包优先。入口会绑定同一个 `EfficientFlow` 包，不改变 cwd；不要只复制入口或放一个空 `__init__.py` 来掩盖缺少源码。

```bash
python train_mp1_ablation.py --verify
python train_mp1_ablation.py --verify --config-name MP1_mgad_full profile=theory
python train_mp1_ablation.py --config-name MP1_full --cfg job --resolve
```

`--verify` 校验新 workspace、所选 policy 是否属于六组白名单、六个 policy 类及共同必需源码是否存在，并打印 profile、入口、源包、目标类及数据路径。它不会导入机器人策略/仿真依赖，不读取 HDF5，不构造模型、不启动训练；因此通过后仍需在真实环境做 smoke。

可在 Linux Bash 一次检查六组：

```bash
for group in MP1_baseline MP1_mgad MP1_diag MP1_full MP1_mgad_diag MP1_mgad_full; do
  python train_mp1_ablation.py --verify --config-name "$group" profile=mp1 || exit 1
done
```

## 真实依赖与数据上的 smoke

需要与原 MP1 一致的点云/图像编码器、robomimic、robosuite、MimicGen、MuJoCo 及 GPU 训练环境，并准备配置指向的 voxel absolute-action HDF5。`--verify` 不验证这些运行时依赖。本次未在真实 HDF5 或 MuJoCo 上完成训练/rollout；CPU 数值与梯度检查不能代替它们。

先用单个有头组跑两个 epoch、每轮两个 minibatch；关闭 rollout 仅用于隔离训练流水线，所有正式比较组须使用相同设置：

```bash
python train_mp1_ablation.py --config-name MP1_mgad_full profile=mp1 task_name=kitchen_d1 n_demo=200 training.seed=20 training.device=cuda:0 training.num_epochs=2 training.max_train_steps=2 training.max_val_steps=1 training.enable_rollouts=false training.enable_validation=true training.checkpoint_every=1 dataloader.batch_size=2 val_dataloader.batch_size=2 dataloader.num_workers=0 val_dataloader.num_workers=0 hydra.run.dir=runs/mp1_mgad_full_smoke
```

检查 loss/NLL 是否有限、两组优化器是否更新、checkpoint 是否保存；随后在相同小预算下启用 `training.enable_rollouts=true training.rollout_every=1` 验证环境。短 smoke 只能验证接口，不能用其成功率评价方法。新版 task 中训练与 rollout 均引用同一个 `dataset_path`，一次覆盖会同时生效，入口也会检查两者一致。

## 正式六组训练

下面的 Bash 示例在同一 `mp1` 档、kitchen_d1、200 demonstrations、seed=20 运行六组。按用户标准，`num_epochs=int(50000/n_demo)`、`rollout_every=checkpoint_every=int(1000/n_demo)`；200 条示范对应 250 轮训练、每 5 轮评估和保存 latest checkpoint。各组使用独立输出目录；扩展多个训练 seed 时，不要把同一个 checkpoint 的多次 rollout 当作多个训练 seed。

```bash
for group in MP1_baseline MP1_mgad MP1_diag MP1_full MP1_mgad_diag MP1_mgad_full; do
  python train_mp1_ablation.py --config-name "$group" profile=mp1 task_name=kitchen_d1 n_demo=200 training.seed=20 || exit 1
done
```

严格 H 隔离实验使用相同六组，把上面所有命令统一改成 `profile=theory`，并换输出目录。改变 `metric_rho`、ridge、正定下界、统计头大小或 MGAD 结构时记录覆盖值；默认 `rho=0.25`、`ridge=0.001` 等只是初始工程设置，不能称为已验证最优。

## 结果与续训

新 workspace 写入输出目录中的 `metrics.jsonl`；配置由 Hydra 保存，checkpoint 由共享 `BaseWorkspace` 保存。结果应保留 profile、配置名、seed、数据路径、训练预算、实际主损失、MGAD/统计头开关，以及验证 MSE、NLL、训练成本和评估成功率。负结果同样需要保留。

2026-10-31 补充：新 workspace 现在默认接入 SwanLab online 日志，实验名包含消融组、profile、任务和 seed。runner 通过路径描述返回已完成编码的 MP4；logger 保留原片，并生成 GIF 预览调用 SwanLab Video，标量不再把视频一起过滤。SDK 当前的 Video 接口只接受 GIF，不能直接接收 `wandb.Video` 或 MP4，参见 [官方说明](https://docs.swanlab.cn/api/py-video.html)。预览降低分辨率和帧数，但保留完整播放时长。

本地输出含 `media/*.mp4`、`media/swanlab/*.gif`、`rollout_media.jsonl` 和 `swanlab_run.json`。最后一个文件保留公开 run ID，同目录 resume 可以继续同一个实验。`logging.mode=disabled` 时不导入 SDK，仍保留本地视频路径清单；`offline` 只记录本机，需后续 sync 才能在云端看见。200 条示范时首次 rollout 视频在第 5 epoch；修改频率时保持六组一致。

从已有输出续训，复用相同配置和输出目录，并显式设置 `training.resume=true`。新实验默认 `resume=false`，避免继承其他消融组的 checkpoint；不要用旧 MP1 checkpoint 的结构去自动推断新头已经训练。

```bash
python train_mp1_ablation.py --config-name MP1_full profile=mp1 task_name=kitchen_d1 n_demo=200 training.seed=20 training.resume=true hydra.run.dir=runs/MP1_full_mp1_kitchen_d1_demo200_s20
```

续训示例中的 task、n_demo、seed、模型维度与训练预算也必须与原运行保持一致；若原运行有额外覆盖，需要原样补回。代码检查 policy、优化器、数据、seed、EMA、batch、学习率计划和总 epoch 等签名，拒绝同尺寸但算法不同的 checkpoint。延长总 epoch 会改变 cosine 计划，不能当作完全相同实验的自动续训。跨设备读取可以使用 `TrainMP1AblationWorkspace.create_from_checkpoint(path, map_location="cpu")`；调度器纯状态随 checkpoint 保存，不要求先构造数据集才能加载模型。

原 MP1 的动作切片沿用现有实现：默认 horizon=4、n_obs_steps=2 时，返回执行片段实际为 3 个动作，不能仅根据 n_action_steps=4 声称执行了 4 个。原 dispersive loss 对 batch=1/所有特征相同会产生 0/0；新路径在这个退化情形明确取 0，正常非退化 batch 保留原公式。

## 本次验证记录（2026-10-30）

新增 22 项 CPU 测试通过，另有原 MeanFlow 采样的 11 项回归测试通过；六个配置的入口检查通过。测试包括：

- 对照原 `Meanpolicy` 的实际损失和全部参数梯度；MGAD 关闭时主干逐值一致。
- 六种组合均完成真实 AdamW 更新；统计头与观察编码器/MGAD 的梯度相互隔离。
- 仿射速度场解析检查：w=1/w=2 的实际标签、固定 r 的 JVP 和停止目标梯度。
- 矩阵谱界、迹、正定、旋转关系、跨时间相关性；rho=0 精确退化 MSE、各向同性权重等于 I、NLL 的最优对象含偏差外积。
- 六组推理均只有一次速度网络 forward，统计头被禁止调用时仍能推理。
- 合成点云/状态数据的两轮训练：连续运行与中断后迁移目录恢复的模型、EMA、两组 AdamW 和调度器状态逐值一致；normalizer 恢复且冻结；错误 profile 续训被拒绝。

测试使用实际点云编码器和缩小通道数的 U-Net/MGAD，并非模拟梯度或手工代替 AdamW；但没有运行默认大模型的 GPU 训练、真实 HDF5 或 MuJoCo rollout。本机缺少 robomimic，当前结果不能作为真实任务性能验收。多 worker / GPU 的逐位恢复也未验证。

标准复测命令：

```bash
python -m unittest discover -s tests -p 'test_mp1_*.py' -v
python -m unittest discover -s tests -p 'test_meanflow_sampling.py' -v
```

本机现有 PyTorch 2.1.0 与已安装 Transformers 不兼容，会在创建优化器时触发 `register_pytree_node` 错误。本次仅对测试进程设置 `USE_TORCH=0`，禁用与这些模型无关的 Transformers 的 Torch 集成；实际 PyTorch、autograd、AdamW 均照常执行，未替换实现或修改已安装环境。若同一环境复测，可在 PowerShell 先运行 `$env:USE_TORCH='0'`，或在 Bash 命令前加 `USE_TORCH=0`。正常匹配的训练环境不需要该设置；如果其他实验依赖 Transformers 模型，应使用其自身匹配的依赖环境。

2026-10-31 H11 复核：`test_mp1_*.py` 共运行 36 项，35 项通过，1 项因该环境缺少 PyAV 跳过（视频编解码，与 H11 无关）。新增两项集成测试分别覆盖 `theory/mp1 × MGAD开关 × diag/full` 的 8 种组合：使用非各向同性、依赖输入的统计头，验证 `compute_loss().backward()` 与分别反传的全部 theta/phi 梯度一致、交叉梯度为 None；验证 rho=0 时经过两轮真实 AdamW 和实际触发的独立梯度裁剪，theta 参数仍与匹配 MSE 对照一致，同时 phi 确实更新。另增加 `velocity_loss` 日志，区分速度目标和含 `head_nll` 的总损失。H11 的相加条件及 rho=0 的精确比较对象见 [理论审核](mp1_theory_audit_zh.md#h11-的损失相加与实际更新)。本次复核未改变现有 YAML 的默认 `profile=mp1`；严格 H 运行必须统一显式设置 `profile=theory`。

2026-10-31 视频修复验证：28 项 MP1 测试已覆盖通过，其中旧 Torch 环境缺少 PyAV 的一项编解码检查，在独立 PyAV 环境补测通过。新增覆盖真实 MP4→GIF（原片不变、时长保持）、标量与视频同时记录、disabled 无 SDK、run ID 恢复，以及 rollout 抛错后仍关闭环境和结束日志。另用真实 SwanLab 0.10.3、PyAV 19.0.2 执行合成视频离线记录成功；没有使用云端凭据或声称完成服务器 MuJoCo 视频上传。

可从项目工作目录运行 `python EfficientFlow/scripts/check_mp1_swanlab.py --mode online` 独立检查账号、网络和视频看板；默认 `--mode offline` 不上传。部署及 kitchen_d1/200 demonstrations/seed20 的正式命令见 [服务器说明](MP1_server_commands_zh.md)。
