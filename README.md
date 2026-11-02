# MyMP1

MP1 idea 的完整源码快照：原 MP1、MGAD、H 残差二阶矩统计头，以及六组受控消融。保留 `EfficientFlow.*` 导入路径，便于加载已有配置和 checkpoint。

## 包含内容

| 内容 | 路径 |
| --- | --- |
| 原 MP1 策略、配置、旧训练器 | `EfficientFlow/policy/mp1.py`、`config/MP1.yaml`、`workspace/train_mp1_workspace.py` |
| 六组独立实验配置 | `EfficientFlow/config/MP1_{baseline,mgad,diag,full,mgad_diag,mgad_full}.yaml` |
| 共享消融策略与训练器 | `EfficientFlow/policy/mp1_ablation.py`、`workspace/train_mp1_ablation_workspace.py` |
| MGAD 模块 | `EfficientFlow/model/meanflow_mgad.py` |
| H 统计头、NLL、有界矩阵损失 | `EfficientFlow/common/meanflow_statistics.py` |
| 训练入口与服务器启动脚本 | `EfficientFlow/train_mp1_ablation.py`、`scripts/run_mp1_ablation_server.sh` |
| 旧 checkpoint 评估、SwanLab、视频 | `EfficientFlow/scripts/eval_mp1_checkpoint.py`、`check_mp1_swanlab.py`、`common/swanlab_logging.py` |
| 多步采样、分支实验、方差与路由评估接口 | `EfficientFlow/common/meanflow_sampling.py`、`experiments/meanflow_eval/` |
| 数据转换、数据集、环境封装与 rollout | `EfficientFlow/scripts/robomimic_dataset_*.py`、`dataset/`、`env/`、`env_runner/`、`gym_util/` |
| MimicGen 环境与运行所需模型资源 | `EfficientFlow/mimicgen_environments/` |
| 共享底座、对照策略和早期 EfficientAdaFlow 实现 | `EfficientFlow/model/`、`policy/`、`common/` 及对应文档 |
| 测试、理论审核、运行说明 | `EfficientFlow/tests/`、`EfficientFlow/docs/` |

表内第二条路径起的简写均相对于 `EfficientFlow/`。`SOURCE_MANIFEST.json` 记录复制文件的源 SHA256、发布 SHA256 和来源版本。

数据集、训练权重、实验输出、缓存、旧压缩包、登录凭据及源仓库 Git 历史不包含在此快照中。MimicGen 的 XML、网格和纹理是仿真运行资源，已保留。

## 安装环境

从仓库根目录运行。正式仿真训练按 Linux + NVIDIA CUDA 环境准备；本次整理不代表已在全新 GPU 机器完成训练。

已有可运行的 EfficientFlow 环境：

```bash
conda activate EfficientFlow
python -m pip install -r requirements-mp1.txt
python -m pip install --no-deps -e .
python -m pip install --no-deps -e EfficientFlow/mimicgen_environments
```

新机器的环境起点：

```bash
conda env create -f conda_environment.yaml
conda activate EfficientFlow
python -m pip install --no-deps -e .
python -m pip install --no-deps -e EfficientFlow/mimicgen_environments
```

环境基于原项目 Python 3.9 / PyTorch 2.1 / CUDA 11.8 配置，补充 MP1 日志依赖，并采用本地 CPU 验证使用的 Diffusers 0.27.2 / Hugging Face Hub 0.25.2 组合。原环境文件另存于 `environment/conda_environment.upstream.yaml`。完整 Conda 环境尚未在干净 Linux 机器重建验证。

如果使用的 robomimic 版本仍在 `robomimic/envs/env_robosuite.py` 导入 `mimicgen_envs`，需改成 `import mimicgen`，与本快照的 MimicGen 包名一致。无显示器服务器可设置 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl`。

## 六组配置检查

```bash
for group in MP1_baseline MP1_mgad MP1_diag MP1_full MP1_mgad_diag MP1_mgad_full; do
  python EfficientFlow/train_mp1_ablation.py --verify --config-name "$group"
done
python EfficientFlow/train_mp1_ablation.py --verify --config-name MP1_mgad_full profile=theory
```

`--verify` 不读取真实数据或启动仿真。原版 MP1 的旧入口也保留：`python EfficientFlow/train.py --verify --config-name MP1`。新消融请使用独立的 `train_mp1_ablation.py`；旧训练器含历史 epoch/恢复行为，不用于六组公平对比。

## 数据和训练

默认数据路径：

```text
data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5
```

这是需要自行准备的点云观测、绝对动作数据。可通过 `dataset_path=/absolute/path/file.hdf5` 或脚本的 `MP1_DATASET` 指定已有数据；原始下载和转换说明见 [原项目说明](EfficientFlow/README.md)。

单组训练：

```bash
swanlab login
python EfficientFlow/train_mp1_ablation.py --config-name MP1_mgad_full \
  task_name=kitchen_d1 n_demo=200 training.seed=20 profile=mp1
```

全部六组按相同预算串行执行（默认使用 SwanLab online，先准备数据并登录）：

```bash
bash EfficientFlow/scripts/run_mp1_ablation_server.sh all --dry-run
bash EfficientFlow/scripts/run_mp1_ablation_server.sh all
```

默认 200 demonstrations、seed 20、250 epochs，每 5 epochs rollout 和保存 checkpoint，1 NFE 推理。`profile=mp1` 保留原底座的 guided 标签与辅助损失；严格 H 隔离实验统一使用 `profile=theory`（批量脚本设置 `MP1_PROFILE=theory`）。两档分别比较，详细公平对照条件见 [六组运行说明](EfficientFlow/docs/MP1_ablation_runbook_zh.md)。

## 评估与接口

```bash
python EfficientFlow/scripts/meanflow_experiment.py doctor
python EfficientFlow/scripts/meanflow_experiment.py run --help
python EfficientFlow/scripts/eval_mp1_checkpoint.py --help
python EfficientFlow/scripts/check_mp1_swanlab.py --mode offline
```

原 MP1 采样接口及 fixed / branches / router 评估见 [MeanFlow 接口说明](EfficientFlow/experiments/meanflow_eval/README.zh-CN.md)。`run_meanflow_mp1.sh`、`install_meanflow_bundle.py` 是旧 overlay 压缩包部署工具，依赖对应压缩包布局；Git clone 后直接使用上面的 Python CLI。

## 验证与更多说明

在仓库根目录执行：

```bash
cd EfficientFlow
python -m unittest discover -s tests -p 'test_mp1_*.py' -v
python -m unittest discover -s tests -p 'test_meanflow_*.py' -v
cd ..
```

部分历史测试直接导入 `model` / `experiments`，因此上面的测试命令进入 `EfficientFlow/` 执行。

- [理论与 H11 审核](EfficientFlow/docs/mp1_theory_audit_zh.md)
- [六组 YAML 审核](EfficientFlow/docs/MP1_yaml_audit_zh.md)
- [训练全零、归一化与录像修复](EfficientFlow/docs/MP1_zero_score_fix_20261101_zh.md)
- [服务器启动与 smoke 命令](EfficientFlow/docs/MP1_server_commands_zh.md)
- [本次快照验证记录](VALIDATION.md)

历史文档中的服务器地址已替换为占位符；旧输出目录、压缩包名称和历次测试结论仅为记录。当前 clone 的入口以本 README 为准。仓库没有附带六组训练结果，也不声称算法已经提高任务成功率。

## 来源与许可证

基于 [EfficientFlow](https://github.com/Mirage627/PROJECT_EF)、[MP1](https://github.com/LogSSim/MP1) 及原项目列出的工作扩展。原作者、引用及致谢见 [保留的原 README](EfficientFlow/README.md)。主项目保留 [MIT LICENSE](LICENSE)；MimicGen 按其 [NVIDIA License](EfficientFlow/mimicgen_environments/LICENSE)，其他第三方组件遵循各自许可证。
