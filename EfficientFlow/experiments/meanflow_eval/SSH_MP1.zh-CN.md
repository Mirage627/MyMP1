# MP1：安装后直接启动实验

本包只依据你指定的 `/path/to/EfficientFlow\policy\mp1.py` 制作。安装器将先备份服务器根目录的 `policy/mp1.py`，再复制加入多步采样后的完整文件，避免文本补丁上下文或换行符不一致。评测入口固定导入项目根目录源码。

使用原来的 `EfficientFlow` Conda 环境。把 **meanflow_mp1_ready.tar.gz** 上传到 `/home/YOUR_USER/PROJECT_EfficientFlow/`，在 SSH 终端执行：

```bash
cd /home/YOUR_USER/PROJECT_EfficientFlow
tar -xzf meanflow_mp1_ready.tar.gz
bash meanflow_mp1_ready/run.sh smoke
```

这一条脚本会依次安装、寻找 checkpoint、检查模型配置、运行 2 个初态的固定 1/2/4 步与 oracle 小实验，最后打印报告路径。使用已有训练环境，不安装依赖、不启动训练。被替换文件保存在 `data/meanflow_install_backup/`。

通过小实验后，开始正式预实验：

```bash
bash meanflow_mp1_ready/run.sh pilot
```

pilot 使用 20 个环境初态、每个初态 3 个噪声 trial，比较 1/2/4 步及 uniform/noise_first/data_first 网格，同时收集同状态分支。每次运行创建新的结果目录；不会追加旧结果。

脚本默认查找 `data/outputs/` 下 `kitchen_d1 / n_demo=200 / training.seed=20 / MP1` 对应的最新 `latest.ckpt`，根据 `.hydra/config.yaml` 匹配。如果 checkpoint 在其他位置，可直接指定：

```bash
MF_CHECKPOINT=/你的实际路径/latest.ckpt bash meanflow_mp1_ready/run.sh smoke
```

默认数据为 `data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5`。需要时用 `MF_DATASET=/实际路径/data.hdf5` 覆盖；GPU 默认 `cuda:0`，可用 `MF_DEVICE=cuda:1` 指定逻辑设备，或沿用集群设置的 `CUDA_VISIBLE_DEVICES`。

执行长度自动匹配原策略实际输出的动作切片：例如 horizon=4、n_obs_steps=2、n_action_steps=4 时实际返回 3 个动作，因此所有候选统一执行 3 个。horizon=8、n_action_steps=4 时执行 4 个。不会修改 checkpoint 的预测时域。

结果先看打印目录中的 `report.md`：

- 固定 2/4 步相对 1 步的闭环成功率配对差、置信区间及真实推理耗时。
- 同状态升级的 `win / hurt`，判断额外步骤的收益是否集中。
- `oracle_gain_over_matched_all_grid_nonadaptive`，判断相同算力下是否仍有自适应分配空间。

如果查不到 checkpoint，先检查已有权重。根目录 `config/MP1.yaml` 默认 `checkpoint.save_ckpt: False`；从未保存的已结束训练无法事后恢复权重。脚本会明确报错，不会自动重训。

旧版 `patch does not apply` 发生在修改前，无需回滚。此次使用新包的 `run.sh`，它不调用旧安装器。
