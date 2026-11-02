# MP1 全零评估与离线视频修复（2026-11-01）

本次审查发现确定的训练/推理输入不一致；这足以破坏闭环控制，但尚未在用户服务器的 checkpoint、真实数据与 MuJoCo 上验证修复后的成功率，不能保证所有全零现象均由它引起。

## 已确认的问题与修改

`RobomimicReplayPointCloudDataset.get_normalizer()` 只为 action、point_cloud、eef_pos、eef_quat、gripper_qpos 拟合统计，没有为相机字段拟合统计。训练按字段归一化，推理旧实现调用 `normalizer.normalize(obs_dict)`。旧 `LinearNormalizer._normalize_impl()` 遇到任何未知字段就 `return x`，因此只要带有图像，整包点云和状态都会恢复为未归一化的原始值，与训练不同。此问题与字典顺序无关。

修复将未知字段逐项透传，继续归一化其余已知字段；MP1 的训练、推理和六组消融统一调用 `_normalize_observations()`。没有改动作的 6D rotation 表示、反归一化、时间方向、环境绝对控制、reward 定义、MGAD 结构或矩阵损失。原始 MP1 也共享这项输入修复，重新做对照时六组必须采用同一修复版本。

原 MP1 的 horizon=4、n_obs_steps=2，从索引 1 起取动作，实际可执行 3 步；这也是六组当前沿用的行为。此次保留该采样/对齐行为，避免同时改变实验协议。

新增检查：配置要求录像时，评估必须返回同等数量的有效媒体文件；全零奖励也应录像。`rollout/video_count` 默认应为 6（训练 2、测试 4）；缺文件或返回了不兼容的旧视频对象会报错。启动会打印实际 output/workspace/policy/normalizer 路径，便于发现加载旧源码或检查错输出目录。

负 train_loss 并不等于损失实现错误：本实验包含可为负的 Gaussian NLL 与 dispersive loss。`validation_mse` 是随机 MeanFlow 目标的残差，不是从纯噪声采样的动作误差，也不是成功率。

## 更新源码

本机修复包为 `artifacts/mp1_server_source_20261101.tar.gz`，顶层为 `EfficientFlow/`。请在旧训练停止后更换源码，再启动新进程；运行中的 Python 不会可靠地加载磁盘上的修改。

从 Windows PowerShell 上传：

```powershell
scp /path/to/EfficientFlow\artifacts\mp1_server_source_20261101.tar.gz YOUR_USER@YOUR_SERVER:/home/YOUR_USER/MyMP1/
```

在服务器 Bash 解压；备份目录名若已存在，应另选一个新名字：

```bash
cd /home/YOUR_USER/MyMP1
cp -a EfficientFlow EfficientFlow.before_20261101_fix
tar -xzf mp1_server_source_20261101.tar.gz -C /home/YOUR_USER/MyMP1
conda activate EfficientFlow
python EfficientFlow/train_mp1_ablation.py --verify --config-name MP1_mgad_diag
```

## 先重新评估已有 checkpoint

原先训练按字段归一化的路径本身没有这项推理错误，因此优先加载已有权重重新评估，不必先丢弃训练结果。工具默认使用 checkpoint 的 EMA 权重（若没有则使用 model）；加 `--raw-model` 可显式评估原始训练权重。模型结构必须与 checkpoint 对应；严格加载报错时不要强行忽略缺少的参数。评估只读 checkpoint，不执行训练，不改旧日志。

```bash
cd /home/YOUR_USER/MyMP1
RUN=/home/YOUR_USER/MyMP1/data/outputs/2026.10.31/15.15.35_mp1_mgad_diag_mp1_kitchen_d1_seed20
find "$RUN/checkpoints" -maxdepth 1 -type f
# latest.ckpt 存在时执行；如旧训练采用其他文件名，请替换 --checkpoint。
CUDA_VISIBLE_DEVICES=7 python EfficientFlow/scripts/eval_mp1_checkpoint.py \
  --checkpoint "$RUN/checkpoints/latest.ckpt" \
  --output-dir /home/YOUR_USER/MyMP1/runs/MP1_mgad_diag_reeval_20261101 \
  --device cuda:0 --mode offline --n-envs 2
```

默认沿用原训练的评估回合与录像数量。快速排查可以再加 `--n-train 2 --n-test 10`，但不能将此小样本分数与原 6+50 回合结果直接比较。若从另一个 cwd 运行或旧相对数据路径失效，用 `--dataset-path` 传入实际 HDF5 绝对路径。无 checkpoint 则不能恢复过去未录制的过程，需要重新训练/评估。

评估目录中应有 `eval_results.json`、`eval_config.yaml`、`rollout_media.jsonl`、`media/*.mp4`、`media/swanlab/*.gif` 和 `swanlog/run-.../`。

## 离线日志同步

同步的是整个 `swanlog/run-...` 目录，不是单独的 media、metrics.jsonl 或 checkpoint。使用生成日志时同一版本的 SwanLab，尤其不要跨 0.8 日志格式边界盲目升级。可在原训练环境执行 `python -c "import swanlab; print(swanlab.__version__)"`，如离线服务器不联网，将整个 run 目录复制到联网电脑后，用同版本 CLI 上传。

```bash
conda activate EfficientFlow
swanlab login
swanlab sync /home/YOUR_USER/MyMP1/data/outputs/2026.10.31/15.15.35_mp1_mgad_diag_mp1_kitchen_d1_seed20/swanlog/run-20261031_151757-6aoceruohwylds4ohihdj
```

新评估目录内选择实际生成的 run 名，再执行同样的 `swanlab sync 完整路径`。如果希望续传到已有云端实验，可加 `--id 云端实验ID`，不要仅凭旧目录后缀猜测云端实验已经存在。

SwanLab Video API 当前接受 GIF；源 MP4 留在实验输出的 `media/`，GIF 预览在 `media/swanlab/`。SDK 可以把 GIF 存进 `swanlog/run-.../media/image/`。因此“没有 mp4”不等于缺媒体，但如果上述两类源视频都不存在，且没有媒体清单，sync 无法补出从未录制的视频。手动把 MP4 复制进 swanlog 也不会产生可展示的指标记录。

官方说明：[同步离线数据](https://docs.swanlab.cn/guide_cloud/experiment_track/sync-logfile.html)、[Video API](https://docs.swanlab.cn/api/py-video.html)。

## 本地验证

MP1 回归套件共 39 项：CPU 环境通过 38 项，1 项 PyAV 编码测试在该环境跳过；随后媒体环境 6 项全部通过，覆盖了这项实际 MP4 编码/GIF 时长检查。六组策略包含相机及额外渲染字段的归一化一致性、梯度与参数更新、checkpoint 续训与独立重评估均通过。真实 SwanLab 0.10.3 的 offline MP4→GIF→run 日志写入已通过，确认生成了 `media/image/*.gif`；没有执行联网上传，也没有在用户服务器的真实 HDF5/MuJoCo 上测成功率。本机原训练环境存在包版本冲突，测试使用仓库 tmp 下的隔离依赖，没有修改该环境的已安装依赖。

## 本次服务器信息中的目录差异

用户实际检查的旧实验 overrides 为 task=kitchen_d1、200 demos、seed=20、mp1、3000 epochs、offline，未包含 hydra.run.dir 覆盖。后来贴出的启动命令却指定 `/home/YOUR_USER/MyMP1/runs/MP1_mgad_diag_mp1_kitchen_d1_demo200_s20`，不含 3000 epochs；两者不是同一份启动配置。旧实验配置虽要求 2+4 个视频，但已确认所检查旧输出下无 media/ 和 rollout_media.jsonl；仅凭本地当前源码无法确定远端旧进程加载了哪个实现。后续排查以输出内 `.hydra/config.yaml`、`.hydra/overrides.yaml`、启动打印的源码与输出路径为准。
