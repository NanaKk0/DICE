# DICE 环境依赖说明

`DICE (Distillation-driven Co-Evolution)` 是一个面向机器学习势函数的闭环主动学习工作流。当前 `v0.1.0` release 主线使用 MACE / MACE-LES teacher committee 做高质量标注与候选分流，使用 DeepMD student committee 和 LAMMPS 做采样、不确定度评估与持续训练。

这份文档说明当前仓库运行所需要的环境能力。它不是一键安装脚本，也不要求用户复刻历史服务器环境。它的目标是说明：

- 当前正式主线必须具备哪些软件能力；
- 哪些能力只在特定后端或扩展路径下需要；
- 直接 shell 环境和 `environment_profile` 两种环境组织方式各自承担什么角色。

---

## 1. 适用范围

当前仓库的正式主线是：

- 顶层 [`workflow.py`](./workflow.py)
  - `committee_cycle`
    - MACE teacher committee finetune
    - DeepMD student pipeline stages
    - LAMMPS sampling

因此，这份依赖说明主要覆盖：

- workflow 顶层调度；
- MACE / MACE-LES teacher finetune 与 relabel；
- DeepMD student train / evaluate；
- LAMMPS + DeepMD model deviation sampling；
- 可选的 NEP / GPUMD student 路径；
- 可选的 CP2K / first-principles 内部执行路径。

当前 release 的默认 shipped schedule 没有把 CP2K 或 DFT screening 作为必经阶段；外部 DFT 回灌是常规使用方式。

---

## 2. 推荐操作系统

正式工作流运行推荐使用 Linux。

原因是当前主线中的这些部分都默认按 Linux 运行设计：

- MACE / MACE-LES 训练与推理；
- DeepMD 训练与 freeze；
- LAMMPS + DeepMD plugin 采样；
- 可选 CP2K 执行；
- 多进程 / shell 脚本 / 后台任务管理。

Windows 可以用于编辑代码、整理文档和 Git 操作，但正式 workflow 运行环境应优先视为 Linux。WSL 可以用于开发和局部运行，但生产运行仍取决于 CUDA、PyTorch、DeepMD、LAMMPS plugin 等组件是否在该 WSL 环境中正确工作。

---

## 3. 环境组织方式

当前仓库支持两种环境组织方式。

### 3.1 直接 shell 环境方式

这种方式由用户在启动 `workflow.py` 前自行完成：

- source conda；
- 激活包含主线依赖的 Python 环境；
- 设置 CUDA 和线程相关环境变量；
- 按需设置 `CP2K_EXEC`、`LAMMPS_PLUGIN_PATH`、`LD_LIBRARY_PATH` 等本机路径；
- 执行 `python workflow.py <workflow.yaml>`。

这种方式适合“所有主要能力都在同一个 Python 环境里可用”的部署。

常见变量包括：

- `CP2K_EXEC`
- `CUDA_VISIBLE_DEVICES`
- `OMP_NUM_THREADS`
- `TF_INTRA_OP_PARALLELISM_THREADS`
- `TF_INTER_OP_PARALLELISM_THREADS`

### 3.2 environment_profile v3 方式

当前代码还支持 `environment_profile` schema v3。示例文件是：

- [`configs/environments/local_wsl.example.json`](./configs/environments/local_wsl.example.json)

它把软件环境拆成两层：

- `software_profiles`
  - 每个 profile 声明需要 source 的 shell 脚本和可执行文件；
- `bindings`
  - 把 workflow role 绑定到某个 profile。

当前支持的典型 role 包括：

- `teacher_backend.mace.finetune`
- `teacher_backend.mace.relabel`
- `student_backend.deepmd.train`
- `sampling_backend.lammps.sample`
- `student_backend.nep.train`
- `sampling_backend.gpumd.sample`

这种方式适合把 MACE、DeepMD/LAMMPS、GPUMD/NEP 拆在不同环境中运行。每个 profile 的 `source_list` 脚本可以负责 conda activate、设置 `PATH`、`LD_LIBRARY_PATH`、cache 目录、plugin 路径等。

示例 source 脚本位于：

- [`configs/environments/scripts/mace-les.env.sh`](./configs/environments/scripts/mace-les.env.sh)
- [`configs/environments/scripts/deepmd-lammps-dpa4.env.sh`](./configs/environments/scripts/deepmd-lammps-dpa4.env.sh)
- [`configs/environments/scripts/gpumd-nep.env.sh`](./configs/environments/scripts/gpumd-nep.env.sh)

这些脚本包含本地路径，只能作为模板，不能直接假定适用于其他机器。

---

## 4. 核心 Python 依赖

运行 workflow 调度和通用结构处理的 Python 环境至少需要：

- Python 3.10 或兼容版本；
- `PyYAML`；
- `numpy`；
- `ase`；
- `tqdm`。

这些依赖覆盖：

- 顶层 YAML 读取；
- JSON / extxyz 读写；
- 结构合并与格式转换；
- teacher / student 不确定度计算；
- seed pool metadata 和筛选；
- stage orchestration。

推荐安装：

- `scikit-learn`
  - direct selection / clustering 相关策略会用到；
- `dscribe`
  - SOAP descriptor 相关 direct selection 会用到；
- `matplotlib`
  - 只在启用绘图或分析输出时需要。

不建议把历史分析脚本依赖误认为正式主线硬依赖。是否安装 `pandas`、`dpdata` 等，应根据本地分析脚本或额外后处理需求决定。

---

## 5. MACE / MACE-LES 依赖

teacher committee finetune 和 teacher relabel 需要 MACE 环境。

必须具备：

- 可导入 `mace`；
- 可导入 `from mace.calculators import MACECalculator`；
- 可执行 `mace_run_train`；
- 可用 PyTorch；
- PyTorch 与目标 CUDA 运行时兼容。

当前 teacher 配置使用：

- `workflow.model_family = "les"`

因此 MACE-LES 路径还需要：

- 可导入 `les`；
- 当前 MACE 安装能暴露 `mace.modules.models.MACELES`；
- foundation model 与 `les` backend 兼容。

如果改成 standard MACE，则不需要 LES 相关能力，但仍需要普通 MACE + PyTorch 环境。

---

## 6. DeepMD 依赖

当前默认 student backend 是：

- `student_backend = "deepmd"`

DeepMD 环境必须提供：

- `deepmd-kit`；
- Python 推理能力，例如 `from deepmd.calculator import DP` 或当前 DeepMD 版本对应的推理 API；
- `dp` 命令；
- `dp train`；
- `dp freeze`。

这些能力用于：

- student committee 训练；
- checkpoint freeze / `graph.pb` 导出；
- committee evaluation；
- LAMMPS sampling 前后的 DeepMD 模型组织。

DeepMD 通常依赖 TensorFlow 或对应版本的运行时组件。用户需要保证 DeepMD、CUDA、驱动和 LAMMPS DeepMD plugin 之间版本兼容。

---

## 7. LAMMPS + DeepMD 采样依赖

当前默认 sampling backend 是：

- `sampling_backend = "lammps"`

正式采样必须具备：

- 一个可执行的 LAMMPS 程序；
- 对 `pair_style deepmd` 的支持；
- 能生成 trajectory dump；
- 能生成 `model_devi.out`；
- 能在当前 shell 环境中找到 DeepMD plugin 及相关动态库。

这种支持通常来自两种方式之一：

1. LAMMPS 已经编译进 DeepMD 支持；
2. LAMMPS 通过兼容的 plugin 获得支持，例如 `libdeepmd_lmp.so`。

在 `environment_profile` 方式下，`sampling_backend.lammps.sample` 对应的 profile 应声明：

- executable key: `lmp`

并在 source 脚本中设置必要的：

- `LAMMPS_PLUGIN_PATH`
- `LD_LIBRARY_PATH`
- `DP_INTERFACE_PREC`
- `DP_INFER_BATCH_SIZE`
- 线程相关变量

具体变量名和路径取决于本地 DeepMD/LAMMPS 构建。

---

## 8. 可选 NEP / GPUMD 依赖

代码库包含 NEP student backend 和 GPUMD sampling backend，但当前默认 shipped DeepMD 主线不依赖它们。

只有当配置切换为：

- `student_backend = "nep"`
- `sampling_backend = "gpumd"`

才需要安装：

- `nep` 可执行文件；
- `gpumd` 可执行文件；
- GPUMD/NEP 运行所需的 GPU 和动态库环境。

在 `environment_profile` 方式下，相关 role 是：

- `student_backend.nep.train`
- `student_backend.nep.evaluate`
- `sampling_backend.gpumd.sample`

对应 profile 应声明 executable key：

- `nep`
- `gpumd`

---

## 9. 可选 CP2K / First-Principles 依赖

CP2K 只在用户希望 workflow 内部直接执行 first-principles DFT 时才是必须依赖。

如果采用当前常见的外部 DFT 回灌模式：

1. workflow 产生 teacher DFT candidate；
2. 用户从 relabel stage 的 `active_learning/dft.xyz` 整理 DFT 输入；
3. 用户在 workflow 外完成 DFT；
4. 用户把结果放回 `user_assets/dft/rX/labeled.xyz`；

那么 CP2K 不是日常运行的硬性依赖。

如果启用 workflow 内部 CP2K，则必须提供：

- 可用的 CP2K 可执行文件；
- CP2K basis / potential 数据文件；
- 与配置匹配的提交或本地运行命令；
- 可写的任务目录和输出目录。

当前 CP2K 路径可通过以下方式解析：

- 环境变量 `CP2K_EXEC`；
- first-principles 配置中的 `cp2k_exec` 或 runtime command；
- 代码中的候选可执行文件搜索。

---

## 10. GPU / CPU 预期

当前 workflow 可以在 CPU 上做轻量导入检查、配置解析、部分小规模 smoke，但正式生产运行默认面向 CUDA GPU。

典型预期：

- MACE / MACE-LES finetune
  - 通常需要 GPU；
- MACE relabel
  - 小批量可 CPU，但正式运行建议 GPU；
- DeepMD student train
  - 通常需要 GPU；
- LAMMPS + DeepMD model deviation sampling
  - 通常需要 GPU；
- CP2K
  - 取决于本地 CP2K 构建，可 CPU 或 GPU。

仓库本身不限定 GPU 型号，但必须保证以下组件彼此兼容：

- NVIDIA driver；
- CUDA runtime；
- PyTorch；
- MACE / LES；
- DeepMD；
- LAMMPS DeepMD plugin。

---

## 11. 环境变量与运行时约定

这些变量属于运行时约定，不是 Python 包安装清单。

### 11.1 启动层变量

直接 shell 环境或 `environment_profile` source 脚本中常见需要设置：

- `CUDA_VISIBLE_DEVICES`
- `OMP_NUM_THREADS`
- `TF_INTRA_OP_PARALLELISM_THREADS`
- `TF_INTER_OP_PARALLELISM_THREADS`
- `TF_CPP_MIN_LOG_LEVEL`
- `ABSL_LOG_LEVEL`
- `CP2K_EXEC`

### 11.2 MACE 相关变量

MACE 环境中常见需要设置：

- `PYTHONPATH`
- `MPLCONFIGDIR`
- `XDG_CACHE_HOME`
- `TORCHINDUCTOR_CACHE_DIR`
- PyTorch / CUDA 相关变量

### 11.3 DeepMD / LAMMPS 相关变量

DeepMD + LAMMPS 环境中常见需要设置：

- `PYTHONPATH`
- `DP_INTERFACE_PREC`
- `DP_INFER_BATCH_SIZE`
- `OMP_NUM_THREADS`
- `NUM_WORKERS`
- `LAMMPS_PLUGIN_PATH`
- `LD_LIBRARY_PATH`
- cache 目录变量

具体值应按本地安装位置设置。示例脚本中的 `/mnt/c/Users/...` 路径只是本地模板。

---

## 12. 最小能力清单

如果用户运行当前默认 MACE teacher + DeepMD student + LAMMPS sampling 主线，建议至少具备下面能力。

### 必需

- Linux 运行环境；
- Python 3.10 或兼容版本；
- `PyYAML`；
- `numpy`；
- `ase`；
- `tqdm`；
- `mace`；
- `mace_run_train`；
- PyTorch；
- `deepmd-kit`；
- `dp` 命令；
- 支持 DeepMD 的 LAMMPS；
- `lmp` 可执行文件；
- 正确的 `PYTHONPATH`，使 workflow 代码可被各 backend 子进程导入。

### 当前 LES teacher 配置额外必需

- `les`；
- 能暴露 `MACELES` 的 MACE 安装；
- 与 LES backend 兼容的 foundation model。

### 推荐

- `scikit-learn`；
- `dscribe`；
- `matplotlib`。

### 可选

- `nep` 和 `gpumd`
  - 仅当启用 NEP / GPUMD 路径时需要；
- CP2K 可执行文件与 CP2K 数据目录
  - 仅当启用 workflow 内部 first-principles 执行时需要；
- `pandas`、`dpdata` 等分析依赖
  - 仅当本地后处理脚本需要时安装。

---

## 13. 这份文档不要求什么

这份仓库不要求用户：

- 机械照搬旧的 `build_env.sh`；
- 使用固定的历史 conda 环境名；
- 把 MACE、DeepMD、LAMMPS、GPUMD 全装进同一个环境；
- 安装所有开发期辅助工具；
- 在默认主线中安装 CP2K；
- 在默认 DeepMD 主线中安装 NEP / GPUMD。

用户真正需要提供的是：让当前配置选中的 backend role 能找到自己的 Python 包、动态库和可执行文件。
