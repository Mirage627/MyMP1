# 六组 MP1 YAML 核对记录（2026-10-31）

以用户提供的 MP1 标准为训练底座。六个实验文件均已展开为完整配置，只共享 `task: mimicgen_pc_abs`，不继承公共训练 YAML。`MP1_ablation_base.yaml` 仅作旧配置名的兼容别名，指向新版 baseline。

## 一致的训练底座

| 项目 | 六组统一设置 |
| --- | --- |
| 当前任务 / 示范 / 训练 seed | kitchen_d1 / 200 / 20，来自用户此前指定 |
| 数据表示 | point cloud、absolute action、rotation_6d，沿用 task schema |
| 数据路径 | `data/robomimic/datasets/${task_name}/${task_name}_voxel_abs.hdf5` |
| 主干 | FiLM U-Net；time embedding=128；channels=[512,1024,2048]；kernel=5；groups=8 |
| 点云编码 | out=64、XYZ 3 通道、LayerNorm，原 MP1 编码器不变 |
| 动作与观察 | horizon=4、n_obs_steps=2、n_action_steps=4；保留原动作切片语义 |
| batch / workers | train=128/8；val=128/8；persistent_workers=false |
| 主优化器 | AdamW，lr=1e-4，betas=[.95,.999]，eps=1e-8，weight_decay=1e-6 |
| EMA | update_after_step=0、inv_gamma=1、power=.75、max=.9999 |
| epochs | `int(50000/n_demo)`，200 条示范为 **250** |
| rollout/checkpoint 间隔 | `int(1000/n_demo)`，200 条示范为 **5** |
| warmup / clip | 500 optimizer steps / max_grad_norm=1 |

训练 seed=20 不改变 task 中原有的数据划分 seed=42；六组使用同一划分，便于匹配比较。新实验默认 `training.resume=false`，显式续训时设 true。

## 每组需要的附加配置

| 文件 | 策略入口 | MGAD 参数 | 统计头与有界度量 | 独立头优化器 |
| --- | --- | --- | --- | --- |
| MP1_baseline.yaml | policy.mp1_baseline.Policy | 无 | 无 | 无 |
| MP1_mgad.yaml | policy.mp1_mgad.Policy | 4 heads、MLP ratio=2 | 无 | 无 |
| MP1_diag.yaml | policy.mp1_diag.Policy | 无 | 对角 | 有 |
| MP1_full.yaml | policy.mp1_full.Policy | 无 | 完整矩阵 | 有 |
| MP1_mgad_diag.yaml | policy.mp1_mgad_diag.Policy | 有 | 对角 | 有 |
| MP1_mgad_full.yaml | policy.mp1_mgad_full.Policy | 有 | 完整矩阵 | 有 |

统计组统一 rho=.25、ridge=.001、endpoint_margin=.001、head hidden=128、variance floor=.0001、initial variance=1。头优化器统一 AdamW/lr=1e-4/weight_decay=0。无头组不包含无效的头优化器或统计超参数；无 MGAD 组不包含 MGAD 超参数。

六组都指向 `TrainMP1AblationWorkspace`，因为该训练器支持独立头优化器、正确的停止梯度及 SwanLab 媒体记录。不能改回旧 `TrainMP1Workspace` 后仍假定统计头会按本算法正确训练。

`profile=mp1` 保留原 w=2 与 dispersive=.5；无头组为 adaptive L2，有头组替换为 H 矩阵平方损失。`profile=theory` 的六组统一 w=1、dispersive=0、无头组 MSE。该开关仍保留，不能只改其中一组的 profile 比较。

## 删除和修正的字段

- 删除新训练器不读取的 debug、sample_every、training.tqdm_interval_sec、ema_decay。
- 删除旧 topk、save_last_ckpt、save_last_snapshot；新训练器按周期保存 `checkpoints/latest.ckpt`，不声称保存 best 或全部历史轮次。启用 `save_ckpt=true`，保留可恢复的模型和优化器状态。
- 删除旧 W&B multi_run 命名、未引用的 exp_name/keypoint_visible_rate 和 logging.resume。SwanLab 续接由实际 checkpoint 恢复状态和 run ID 文件控制。
- 删除 `use_point_crop` 和当前 MP1Encoder 未使用的 `crop_shape`，它们不改变现有点云主干行为。
- 保留 SwanLab backend/mode/name/tags/group/id 和视频参数，避免丢失前次视频修复。
- 将 task 的 rollout 数据路径改为 `${dataset_path}`；修改一次数据路径时训练与评估同步，入口拒绝两者不一致的配置。
- 保留 Hydra 的独立运行目录和 sweep 目录，显式 `job.chdir=false`。直接模块运行按时间生成独立输出；服务器脚本使用固定的每组目录并拒绝覆盖非空的新实验目录。

服务器脚本不再默认追加 3000/200 的旧周期；只有用户明确指定预算环境变量时才追加 override。不要把旧 3000 轮 checkpoint 当作新 250 轮标准实验续训，也不要在新命令里保留旧周期覆盖。

## 验证范围

新增 Hydra 检查覆盖六个独立配置、两种 profile、100/200/500 示范的周期计算、数据路径绑定、共同主干/优化器一致性，以及零周期/训练评估数据不一致的提前拒绝。

实际小模型检查逐组从六个完整 YAML 构造策略与 workspace，验证 head optimizer 只在四个统计组存在，并完成真实 AdamW 更新。检查使用缩小的网络通道与合成数据，不代表完成默认大模型 GPU 训练或服务器 rollout。
