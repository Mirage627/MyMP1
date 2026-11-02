# MyMP1 快照验证（2026-11-02）

验证对象是本仓库独立导出的文件，而非原工作目录的 Git 索引。原工作目录尚有未解决的索引合并状态；这里直接读取当前工作文件建立全新仓库，没有修改原索引或复制旧 Git 历史。

## 完成的检查

- 六组配置分别在 `profile=mp1` 和 `profile=theory` 下执行 `--verify`：12/12 通过。
- 原 `MP1` 配置经 `train.py --verify --config-name MP1` 检查通过。
- MP1 测试集：39 项，主 CPU 环境中 38 项通过、1 项因缺少 PyAV 跳过。
- MeanFlow 采样、评估、路由、方差与部署接口测试：57/57 通过。
- 独立视频环境执行全部 6 项 SwanLab 测试：6/6 通过，包含上述跳过的真实 MP4 → GIF 转换、源文件保持和时长检查。两个环境合计覆盖全部 96 项不同的 MP1 / MeanFlow 测试。
- 保留的 252 个 Python 文件通过语法编译检查。
- 六组服务器启动脚本通过 `bash -n` 语法检查。
- 798 个复制文件与 `SOURCE_MANIFEST.json` 的发布 SHA256 一致；检查未发现私钥或常见访问令牌格式，个人服务器地址已替换为占位符。
- 根目录 `setup.py --name` 返回 `MyMP1`；MeanFlow `doctor` 选择本快照内的源包，并确认入口和环境 YAML 无冲突标记。

## 验证环境及范围

主 CPU 验证使用已有 Python 3.9.15 / PyTorch 2.1.0 环境，进程内通过 `PYTHONPATH` 使用本机现成的 Diffusers 0.27.2 / Hugging Face Hub 0.25.2。该旧环境已有不匹配的 Transformers，因此仅测试进程设置 `USE_TORCH=0`；实际 PyTorch、autograd、JVP 与 AdamW 继续正常运行。详情见六组运行说明中的历史兼容性记录。

视频补测使用独立 Python 运行时及已有 PyAV 19.0.2 依赖。测试使用 SDK 替身验证日志参数和载荷，不代表云端登录、真实服务器视频上传或完整 MuJoCo rollout 已完成。

本次未在干净 Linux GPU 机器重建完整 Conda 环境，未运行真实 HDF5 / MuJoCo 任务训练，未检验六组任务成功率。`doctor` 报告此 Windows 验证环境缺少 robomimic、robosuite 和 PyTorch3D。

## 发布整理的改动

- 保留 `EfficientFlow/` 目录和历史包名，新增 MyMP1 README、根目录可编辑安装入口、MP1 环境补充文件与逐文件来源清单。
- 文本统一为 UTF-8 / LF；历史服务器账号、主机和本机绝对路径替换为可移植占位符。
- 复制的 `test_meanflow_sampling.py` 补上实际 `_normalize_observations` 方法的 AST 提取，修复该旧测试在观测归一化更新后遗漏辅助方法的问题；未修改策略训练或推理算法。
- 排除与 MP1 无关、包含截断 Python 的 RDT 草稿目录 `model/rdtmodels/` 及其唯一使用方 `model/equi/equi_conditional_dit.py`。
- 保留原主项目、MimicGen 和其他组件的许可证及作者信息。

数据、权重、实验运行记录、旧压缩包、缓存和凭据均不属于本次发布内容。
