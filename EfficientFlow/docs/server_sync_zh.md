# Windows 到 Linux 的代码同步

## Kitchen D1 的 EfficientAdaFlow 命令

在服务器的项目根目录、已激活的训练环境中运行：

```bash
cd /home/YOUR_USER/PROJECT_EfficientFlow
python train_efficientadaflow.py --verify task_name=kitchen_d1 n_demo=200 training.seed=20
python train_efficientadaflow.py task_name=kitchen_d1 n_demo=200 logging.project=EfficientAdaFlow logging.name=kitchen_d1_200_s20 training.seed=20
```

这会从零训练，不需要旧 checkpoint。配置默认不续训，200 demonstrations 对应默认 250 epoch，每 5 epoch 评估和保存。训练/数据依赖须已安装，默认数据为 `data/robomimic/datasets/kitchen_d1/kitchen_d1_abs.hdf5`；路径不同时加 `dataset_path=/actual/path/kitchen_d1_abs.hdf5`。

`--verify` 仅校验配置和源码文件是否齐全，不构造策略、不读取 HDF5。确认输出含 `ENTRY_CHECK_OK`、`WORKSPACE_TARGET=EfficientFlow.workspace.train_equi_workspace.TrainEquiWorkspace`。新入口独立于旧 `train.py`，显式绑定当前目录源码，避免服务器旧 MP1 入口或嵌套旧包被误用。

命令末尾无需 `; \`。Linux Bash 的 `\` 只用于换行，后面必须还有命令内容。

## 一次性准备

推荐 [rsync](https://download.samba.org/pub/rsync/rsync.1)：按路径更新，配合 checksum 可跳过内容相同但时间不同的文件。它需要在本机和服务器两端安装。

本机已发现 `D:\Software\MSYS2\usr\bin\bash.exe`，但未发现 rsync。打开 **MSYS2 MSYS** 终端，按[官方包说明](https://packages.msys2.org/packages/rsync)安装：

```bash
pacman -S --needed rsync openssh
```

服务器登录后检查 `rsync --version`。Ubuntu/Debian 若未安装，用 `sudo apt update` 和 `sudo apt install rsync`；无 sudo 权限时由管理员安装。

MSYS2 的 SSH 配置目录可能与 Windows OpenSSH 不同；第一次连接可能提示核对服务器指纹、重新输入登录密码。保持主机指纹校验；不要为了省事关闭它。

## 每次修改后执行

在本机 **MSYS2** 终端：

```bash
cd /d/GithubRepo/EfficientFlow
bash scripts/sync_code.sh
```

默认只预览。检查列出的目标和文件后正式传输：

```bash
bash scripts/sync_code.sh --apply
```

也可在本机 PowerShell 调用同一个脚本：

```powershell
& 'D:\Software\MSYS2\usr\bin\bash.exe' -lc 'bash /d/GithubRepo/EfficientFlow/scripts/sync_code.sh'
& 'D:\Software\MSYS2\usr\bin\bash.exe' -lc 'bash /d/GithubRepo/EfficientFlow/scripts/sync_code.sh --apply'
```

这是本机到服务器的单向同步。同路径文件更新、新文件补上、未变化文件跳过；不会每次生成新文件夹或备份副本。源目录末尾 `/` 表示复制目录内容，所以本机 `policy/EfficientAdaFlow.py` 对应服务器 `/home/YOUR_USER/PROJECT_EfficientFlow/policy/EfficientAdaFlow.py`。

默认不删除服务器独有文件，也不上传 `.rsyncignore` 排除的内容：数据、输出、checkpoint、模型权重、日志、缓存、Git 和本地环境配置。若将来要同步权重或数据，应另开传输任务或明确调整排除规则。其他命名的服务器输出目录应补入排除文件。

## 改名、删除与镜像

默认同步不会新增未改动文件的副本，但本地删除或改名后的旧路径会留在服务器。确实需要使服务器源码与本地一致时：

```bash
bash scripts/sync_code.sh --mirror
# 确认预览中的 *deleting 项确实可删后：
bash scripts/sync_code.sh --mirror --apply
```

mirror 会删除服务器上本地不存在的、未排除的文件；服务器单独写过的源码也在其中。普通 exclude 同时保护相应远端文件免于此类删除；不要再加 `--delete-excluded`。参见 [rsync 删除语义](https://download.samba.org/pub/rsync/rsync.1#opt--delete)。

若服务器仍是旧的 `PROJECT_EfficientFlow/EfficientFlow/policy/` 嵌套布局，本次平铺源码同步不会自动移走或删除旧副本。先确认训练使用的是根目录的新 `train_efficientadaflow.py`，并检查旧目录中的独有内容，再单独处理旧副本。不要直接把旧目录当临时文件删掉。

训练前完成同步；运行中的训练不宜同步源码，以免同一次实验混用不同代码版本。

## 复用到其他项目/服务器

脚本自动使用它所在项目的根目录，目标可覆盖：

```bash
bash scripts/sync_code.sh --dest username@server:/absolute/path/project/
bash scripts/sync_code.sh --dest username@server:/absolute/path/project/ --apply
```

换 SSH 端口：

```bash
RSYNC_RSH='ssh -p 2222' bash scripts/sync_code.sh --apply
```

把脚本与 `.rsyncignore` 一起保留在项目中，日常流程就是“本机修改 → 预览 → 同步 → 服务器训练”。本次只创建了工具和说明，没有连接或修改目标服务器，也没有代装 rsync。
