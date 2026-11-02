# MyMP1 空白目录部署与正式训练命令

目标：`YOUR_USER@YOUR_SERVER:/home/YOUR_USER/MyMP1`。本说明提供待执行命令，没有连接该服务器或声称远端已部署完成。源码包不包含 demonstration、checkpoint 或 Conda 环境。

约定解压后布局如下；从 `MyMP1` 运行，避免相对数据路径落到源码目录：

```text
/home/YOUR_USER/MyMP1/
  EfficientFlow/                  # 完整源码包
    train_mp1_ablation.py
    config/  policy/  model/  common/  workspace/  scripts/
  data -> 用户确认的已有数据根目录
  runs/                          # 六组独立输出，不写入共享数据根
```

## 1. 本机 Windows PowerShell 上传

完整包的上传文件名为 `mp1_server_source_20261031.tar.gz`，包内顶层应为 `EfficientFlow/`。先确认下面 `$mp1Bundle` 指向已经生成的实际完整包；不要把 GitHub 仓库链接或单个补丁当作包上传。

```powershell
$mp1Host = 'YOUR_USER@YOUR_SERVER'
$mp1Bundle = '/path/to/EfficientFlow\artifacts\mp1_server_source_20261031.tar.gz'
if (-not (Test-Path -LiteralPath $mp1Bundle -PathType Leaf)) { throw "找不到部署包：$mp1Bundle" }
ssh $mp1Host 'mkdir -p /home/YOUR_USER/MyMP1'
if ($LASTEXITCODE -ne 0) { throw '远端目录创建失败' }
scp $mp1Bundle "${mp1Host}:/home/YOUR_USER/MyMP1/mp1_server_source_20261031.tar.gz"
if ($LASTEXITCODE -ne 0) { throw '源码包上传失败' }
ssh $mp1Host
```

首次连接由 SSH 正常核验主机指纹；如果已有非默认端口，在 `ssh` 加 `-p`、`scp` 加 `-P` 并使用真实端口。命令没有设置自动接受新指纹。

## 2. 服务器解压与已有环境

在登录后的 Linux Bash 执行：

```bash
cd /home/YOUR_USER/MyMP1
tar -tzf mp1_server_source_20261031.tar.gz | head -n 20
# 确认包内为 EfficientFlow/...，不是绝对路径。
tar -xzf mp1_server_source_20261031.tar.gz -C /home/YOUR_USER/MyMP1
test -f EfficientFlow/train_mp1_ablation.py
test -f EfficientFlow/config/MP1_mgad_full.yaml
conda activate EfficientFlow
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

如果 `conda activate` 尚未初始化，先按该机器已有 Conda 安装加载其 shell 初始化脚本；不要猜一个 Miniconda 安装路径。空白项目目录不等于没有 Conda 环境。使用已能运行原 MP1/MimicGen 的环境；本 GitHub 快照的根目录提供整理后的 `conda_environment.yaml`；已有可用环境可继续复用。如 `EfficientFlow` 环境不存在，需要先准备匹配的训练依赖，再运行下面命令。

若服务器没有 SwanLab，在已激活的训练环境安装并登录：

```bash
python -m pip install swanlab pillow av
python -m pip install --no-deps -e EfficientFlow/mimicgen_environments
swanlab login
```

凭据通过 SwanLab 自身登录流程输入，不写进训练脚本或上传包。MuJoCo/robosuite 的无窗口渲染也须已可用；NVIDIA/EGL 环境通常使用 `export MUJOCO_GL=egl`，应以该机原有可用渲染配置为准。

## 3. 关联真实已有数据

已核对 MP1、EfficientFlowVoxel 等 YAML：它们使用 `data/robomimic/datasets/${task_name}/${task_name}_voxel_abs.hdf5` 相对路径，没有写死服务器绝对路径。本次保持相同层级，从 MyMP1 启动，将旧项目的 data 链接过来。

可先验证旧布局这个候选文件；只有 `test -f` 成功才据此选择数据根目录：

```bash
test -f /home/YOUR_USER/PROJECT_EfficientFlow/data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5 && ls -lh /home/YOUR_USER/PROJECT_EfficientFlow/data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5
```

```bash
cd /home/YOUR_USER/MyMP1
# 复用原项目的数据；先检查文件，避免链接到不存在的位置。
MP1_EXISTING_DATA='/home/YOUR_USER/PROJECT_EfficientFlow/data'
test -f "$MP1_EXISTING_DATA/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5" || { echo '旧项目中未找到 MP1 所需点云 HDF5，请核对服务器数据位置'; exit 1; }
if [ -e data ] || [ -L data ]; then
  echo 'MyMP1/data 已存在；先人工核对，不自动替换。'
else
  ln -s "$MP1_EXISTING_DATA" data
fi
readlink -f data
ls -lh data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5
```

正式任务为 `kitchen_d1`、200 条示范、训练 seed=20。链接后访问 `/home/YOUR_USER/MyMP1/data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5`。若数据根有变化，也可通过 `MP1_DATASET` 指定真实文件；脚本同时覆盖训练 dataset 和 rollout runner 的数据路径。

脚本只检查目录存在、HDF5 文件可读；不会据文件名保证其中含所需图像、点云、机器人状态或 absolute action。需用原项目转换后的 voxel 数据；普通 `_abs.hdf5` 不能仅改名冒充 `_voxel_abs.hdf5`。

## 4. 配置检查与视频 smoke

```bash
cd /home/YOUR_USER/MyMP1
python EfficientFlow/train_mp1_ablation.py --verify --config-name MP1_baseline profile=mp1
bash EfficientFlow/scripts/run_mp1_ablation_server.sh --help
python EfficientFlow/scripts/check_mp1_swanlab.py --mode online
MP1_TASK=kitchen_d1 MP1_RUN_ROOT=/home/YOUR_USER/MyMP1/smoke_runs MP1_EPOCHS=2 MP1_BATCH_SIZE=2 MP1_WORKERS=0 MP1_MAX_TRAIN_STEPS=2 MP1_MAX_VAL_STEPS=1 MP1_ROLLOUT_EVERY=1 MP1_CHECKPOINT_EVERY=1 MP1_N_ENVS=1 MP1_N_TRAIN=1 MP1_N_TEST=1 MP1_N_TRAIN_VIS=1 MP1_N_TEST_VIS=1 bash EfficientFlow/scripts/run_mp1_ablation_server.sh MP1_mgad_full
```

这会在一个带统计头/MGAD 的组上跑最小训练与 rollout，并以 `logging.backend=swanlab logging.mode=online` 上传日志和视频。核对 SwanLab 项目中的运行名、标量及视频，并检查输出目录保存的原始 MP4。SwanLab 视频路径由新日志适配器转换为 GIF 后上传；保留 MP4 用于原始回放。需要可用的在线网络和已登录账号。

smoke 两轮很短，只用于验证真实依赖、数据、两组优化器、渲染和视频上传，不能用其成功率判断方法。按当前标准，200 条示范的正式训练每 5 epoch rollout/保存 checkpoint；需要临时改变时用 `MP1_ROLLOUT_EVERY`、`MP1_CHECKPOINT_EVERY`，六组保持一致。

## 5. 正式单组或六组串行

默认值与六个独立完整 YAML 一致：task=`kitchen_d1`、demonstrations=200、seed=20、250 epoch、每 5 epoch rollout/保存、batch=128、GPU=0、profile=`mp1`。预算按 `int(50000/n_demo)` 和 `int(1000/n_demo)` 计算；脚本默认不追加这三个预算覆盖值，直接采用所选 YAML。GPU 是物理编号，脚本设置 `CUDA_VISIBLE_DEVICES`，策略内部使用 `cuda:0`；这不是多 GPU 分布式训练。

先预览命令与数据/输出路径：

```bash
cd /home/YOUR_USER/MyMP1
MP1_TASK=kitchen_d1 MP1_PROFILE=mp1 MP1_SEED=20 MP1_GPU=0 bash EfficientFlow/scripts/run_mp1_ablation_server.sh all --dry-run
```

六组各自的正式训练命令如下，均不覆盖标准预算。从 `/home/YOUR_USER/MyMP1` 运行，需要哪一组就执行对应一条：

```bash
# 原 MP1 基线
python -m EfficientFlow.train_mp1_ablation --config-name=MP1_baseline task_name=kitchen_d1 n_demo=200 training.seed=20 profile=mp1 logging.mode=online

# MP1 + MGAD
python -m EfficientFlow.train_mp1_ablation --config-name=MP1_mgad task_name=kitchen_d1 n_demo=200 training.seed=20 profile=mp1 logging.mode=online

# MP1 + 对角统计头
python -m EfficientFlow.train_mp1_ablation --config-name=MP1_diag task_name=kitchen_d1 n_demo=200 training.seed=20 profile=mp1 logging.mode=online

# MP1 + 完整统计头
python -m EfficientFlow.train_mp1_ablation --config-name=MP1_full task_name=kitchen_d1 n_demo=200 training.seed=20 profile=mp1 logging.mode=online

# MP1 + MGAD + 对角统计头
python -m EfficientFlow.train_mp1_ablation --config-name=MP1_mgad_diag task_name=kitchen_d1 n_demo=200 training.seed=20 profile=mp1 logging.mode=online

# MP1 + MGAD + 完整统计头
python -m EfficientFlow.train_mp1_ablation --config-name=MP1_mgad_full task_name=kitchen_d1 n_demo=200 training.seed=20 profile=mp1 logging.mode=online
```

从 `/home/YOUR_USER/MyMP1` 运行。这套消融使用新入口和六个 MP1 配置，入口不叫 train_fusion。希望同时使用固定输出命名、数据检查和 console.log 时，运行已封装脚本：

```bash
MP1_TASK=kitchen_d1 MP1_PROFILE=mp1 MP1_SEED=20 MP1_GPU=0 bash EfficientFlow/scripts/run_mp1_ablation_server.sh MP1_baseline
```

六组串行正式训练：

```bash
MP1_TASK=kitchen_d1 MP1_PROFILE=mp1 MP1_SEED=20 MP1_GPU=0 bash EfficientFlow/scripts/run_mp1_ablation_server.sh all
```

顺序是 `MP1_baseline`、`MP1_mgad`、`MP1_diag`、`MP1_full`、`MP1_mgad_diag`、`MP1_mgad_full`。任一组失败会立即停止，`tee` 保留真实 Python 退出状态；脚本不会静默跳过某组。默认 `MP1_N_ENVS=28`，rollout/checkpoint 频率跟随 YAML；显存/进程数不足可调小 env 数，比较组应保持相同评估设置。

如需严格 H 隔离档，整批统一设置 `MP1_PROFILE=theory`。只有显式设置 `MP1_EPOCHS`、`MP1_ROLLOUT_EVERY`、`MP1_CHECKPOINT_EVERY` 时，脚本才添加相应正整数覆盖；也接受 `ROLLOUT_EVERY`、`CHECKPOINT_EVERY` 作为后两项的备用环境变量名。改预算时全部六组统一，并记录覆盖值；新 workspace 不会另行强制轮数。

两档的含义、matched-MSE 补充对照与实际 `y_w` 二阶矩对象见 [六组 runbook](MP1_ablation_runbook_zh.md) 和 [理论审核](mp1_theory_audit_zh.md)。默认 `mp1` 档保留 w=2 和 dispersive=0.5；头组替换 adaptive L2。`theory` 档统一 w=1、dispersive=0、无头组 MSE。

长任务可在已经安装的 `tmux` 或 `screen` 会话中运行，避免 SSH 断线终止进程；不要在训练过程中覆盖源码。

## 6. 输出与显式续训

每组输出为：

```text
runs/MP1_baseline_mp1_kitchen_d1_demo200_s20/
  console.log
  metrics.jsonl
  .hydra/
  checkpoints/latest.ckpt
  media/ 或 runner 生成的视频目录
```

其它组/profile/task/seed 使用不同目录。`MP1_RUN_ROOT` 可改输出根；它与只读数据根分开。脚本默认 `MP1_RESUME=false`，在任何所选组的输出已非空时，先拒绝启动整批，避免覆盖旧 checkpoint 或把新结果混入旧日志。需要新实验时选择新的 `MP1_RUN_ROOT`。

明确恢复某一组时，保持原 task、数据文件、seed、总 epoch、batch、profile 与结构配置：

```bash
MP1_TASK=kitchen_d1 MP1_PROFILE=mp1 MP1_SEED=20 MP1_RESUME=true bash EfficientFlow/scripts/run_mp1_ablation_server.sh MP1_full
```

这里仅适用于当前标准 250 epoch、每 5 epoch 评估/保存的同一实验。**新标准实验不要使用旧 3000 epoch / 每 200 epoch 配置生成的 checkpoint 续训**；即使模型尺寸相同，预算和学习率计划也已不同。若原输出目录已有旧结果，用新的 `MP1_RUN_ROOT` 从头启动，不修改或覆盖旧目录。

脚本要求该目录已有 `checkpoints/latest.ckpt`；workspace 还会校验续训签名。全六组 resume 时六组都必须存在 checkpoint，不自动从头补跑缺失组。延长总 epoch 会改变学习率计划，应另作实验处理。

本机已通过六组解包后的模块入口检查、训练回归与真实 MP4→GIF→SwanLab 离线记录。尚未执行远端部署、默认大模型 GPU 训练、真实数据或云端视频上传。默认超参尚无机器人成功率验证。
