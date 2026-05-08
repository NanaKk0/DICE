# 环境依赖说明

这份文档说明当前仓库运行所需要的环境能力。它不是一份一键安装脚本，用户也不需要完全照搬旧的 `build_env.sh`。本文档的目标是说明：

- 当前上传的工作流代码依赖什么；
- 哪些组件是必须的；
- 哪些组件只在特定场景下需要。

---

## 1. 适用范围

当前仓库的正式主线是：

- 顶层 `workflow.py`
  - `committee_cycle`
    - MACE teacher committee finetune
    - DP active-learning subworkflow

因此，这份依赖说明主要覆盖：

- workflow 顶层调度
- MACE teacher committee
- DeepMD student committee
- LAMMPS 采样
- 可选的 CP2K 内部 DFT

它不要求用户安装所有开发期辅助工具，也不要求用户复刻历史环境脚本的全部内容。

---

## 2. 推荐操作系统

对于正式工作流运行，推荐使用 Linux。

原因是当前主线中的以下部分都默认按 Linux 运行设计：

- MACE 训练
- DeepMD 训练
- LAMMPS + DeepMD 采样
- CP2K 执行

Windows 仍然适合用于：

- 编辑代码
- 编写文档
- 整理仓库
- 上传 GitHub

但正式 workflow 运行环境应优先视为 Linux。

---

## 3. 环境分层

当前仓库默认存在两层运行环境：

1. **主 Python 环境**
   - 运行 `workflow.py`
   - 运行 MACE finetune
   - 运行 DeepMD 相关 Python 逻辑
2. **可选的 CP2K 执行环境**
   - 仅当 workflow 内部直接执行 CP2K DFT 时使用

用户不需要和历史服务器完全使用相同的环境名字，但必须满足下面这两条能力要求：

- 至少有一个 Python 环境可以导入仓库代码并运行 MACE / DeepMD 主逻辑；
- 如果需要 workflow 内部 CP2K，则必须有一个可解析的 CP2K 可执行文件路径。

当前服务器启动脚本使用的是：

- 主环境：`deepmd`
- CP2K 可执行文件：通过 `CP2K_EXEC` 注入

但这只是当前部署方案，不是唯一合法方案。

---

## 4. 核心 Python 依赖

主 Python 环境应提供：

- Python `3.10`
- `PyYAML`
- `numpy`
- `ase`
- `tqdm`
- `scikit-learn`

这些依赖覆盖了当前正式主线的核心需求：

- 顶层 YAML 读取
- `extxyz` / LAMMPS 结构格式转换
- 不确定度计算
- teacher labeling
- seed pool 更新
- 数据分布投影与绘图配置支持

说明：

- `matplotlib`
  - 推荐安装
  - 当前代码是惰性导入，正常运行不一定每次都会用到，但如果启用数据分布图输出，就需要它
- `pandas`、`dpdata`、`dscribe`
  - 不属于当前正式主线运行的硬性依赖
  - 它们在历史脚本或分析脚本里曾经有用，但不是当前正式主线路径的必要条件

---

## 5. MACE 依赖

teacher committee 和 teacher labeling 需要一个可用的 MACE Python 环境。

主 Python 环境必须能够导入：

- `mace`
- `from mace.calculators import MACECalculator`

当前代码支持两种 `mace_backend`：

- `standard`
- `les`

### 5.1 当 `mace_backend = "standard"` 时

环境必须支持：

- 正常导入 `mace`
- 正常构造 `MACECalculator`

### 5.2 当 `mace_backend = "les"` 时

环境除了要满足普通 MACE 依赖外，还必须支持：

- `les`
- `mace.modules.models.MACELES`

也就是说，如果使用 `les` 模式，则当前 Python 环境中的 MACE / LES 安装组合必须能够正确暴露 `MACELES` 类。

### 5.3 MACE 运行时对 PyTorch 的要求

MACE 运行还依赖：

- PyTorch

用户不需要完全照搬旧脚本中的 PyTorch 安装命令，但必须保证当前安装的 PyTorch 与下面三者兼容：

- 当前安装的 `mace`
- 如果使用 `les`，则还要兼容 `les`
- 目标机器的 CUDA 运行时

---

## 6. DeepMD 依赖

student committee 需要一个可用的 DeepMD 环境。

主 Python 环境必须提供：

- `deepmd-kit`
- Python 导入：
  - `from deepmd.calculator import DP`
- 命令行工具：
  - `dp`

这些能力用于：

- student committee 训练
- student committee freeze / 导出
- committee uncertainty 推理

当前 workflow 默认假定：

- `dp train`
- `dp freeze`

都可以从主 Python 环境中直接调用。

---

## 7. LAMMPS 采样依赖

当前正式采样后端是：

- `backend = "lammps"`

因此，真实采样运行必须具备一个支持 DeepMD 的 LAMMPS 运行时。

环境必须提供：

- 一个可执行的 LAMMPS 程序
- 对 `pair_style deepmd` 的支持

这种支持可以来自两种方式之一：

1. LAMMPS 已经编译进 DeepMD 支持；
2. LAMMPS 通过兼容的 DeepMD plugin 获得支持。

当前代码会尝试自动发现 plugin，但部署环境本质上仍然必须满足下面之一：

- LAMMPS 自带 DeepMD 支持；
- 或者提供一个可用的 plugin，例如 `libdeepmd_lmp.so`。

如果 plugin 无法自动解析，用户就需要在 workflow 配置里显式提供 plugin 路径。

当前 LAMMPS 运行时还需要支持：

- 轨迹 dump 输出
- `model_devi.out`
- inline 或 grouped-shell 的任务执行模式

---

## 8. CP2K 依赖

CP2K 只在用户希望 workflow 内部直接执行 DFT 时才是必须依赖。

如果用户采用当前常见的手动模式：

- workflow 负责采样与筛选
- 用户导出 DFT 候选
- 用户在 workflow 外完成 DFT
- 用户再把 `labeled.xyz` 放回 `user_assets/dft/rX/`

那么 CP2K 不是日常运行的硬性依赖。

如果用户希望 workflow 内部直接执行 DFT，则必须提供：

- 可用的 CP2K 可执行文件
- CP2K 数据文件
  - 例如 `BASIS_MOLOPT`
  - 以及 `POTENTIAL`

当前仓库会尝试从以下来源解析 CP2K：

- 环境变量 `CP2K_EXEC`
- workflow 配置
- 自动搜索可执行文件

因此，这里的要求不是固定环境名，而是：

- workflow 能够解析到一个可执行的 CP2K 路径；
- 并且能够找到对应的数据目录。

---

## 9. GPU / CPU 预期

当前 workflow 可以在 CPU 上做轻量检查或局部 smoke，但正式生产运行默认是面向 GPU 的。

典型预期如下：

- MACE finetune
  - 通常使用 GPU
- DeepMD student committee 训练
  - 通常使用 GPU
- LAMMPS + DeepMD model deviation 采样
  - 通常使用 GPU

因此：

- 纯 CPU 环境可以用于部分轻量验证；
- 正式规模运行应默认需要 CUDA GPU。

仓库本身不限定 GPU 型号，但必须保证：

- PyTorch
- DeepMD
- LAMMPS 的 DeepMD 支持

与当前 CUDA 运行时兼容。

---

## 10. 环境变量与运行时约定

仓库本身并不强制要求用户复刻历史 shell 脚本，但当前正式运行仍然有几类运行时约定。

### 10.1 workflow 主进程环境

启动层通常会设置：

- `CUDA_VISIBLE_DEVICES`
- `OMP_NUM_THREADS`
- `TF_INTRA_OP_PARALLELISM_THREADS`
- `TF_INTER_OP_PARALLELISM_THREADS`
- `TF_CPP_MIN_LOG_LEVEL`
- `ABSL_LOG_LEVEL`

这些属于运行时行为，不是安装依赖本身。

### 10.2 DeepMD 训练环境变量

student committee 训练时，配置文件可能还会额外注入：

- `DP_INTERFACE_PREC`
- `DP_INFER_BATCH_SIZE`

这些同样属于运行时行为，不属于环境安装清单的一部分。

### 10.3 CP2K 执行路径

如果启用 workflow 内部 CP2K，则应确保：

- `CP2K_EXEC`

可被 workflow 正确解析，或者等价配置字段已经写入配置文件。

---

## 11. 最小能力清单

如果用户希望运行当前上传的正式主线，建议至少具备下面这些能力。

### 必需

- Linux 运行环境
- Python `3.10`
- `PyYAML`
- `numpy`
- `ase`
- `tqdm`
- `scikit-learn`
- `mace`
- `torch`
- `deepmd-kit`
- `dp` 命令可用
- 支持 DeepMD 的 LAMMPS

### 当 `mace_backend = "les"` 时额外必需

- `les`
- 能暴露 `MACELES` 的 MACE 安装

### 推荐

- `matplotlib`

### 可选

- CP2K 可执行文件与数据目录
  - 仅当用户希望 workflow 内部直接执行 DFT 时需要

---

## 12. 这份文档不要求什么

这份仓库不要求用户：

- 机械照搬旧的 `build_env.sh`
- 使用历史环境名
- 安装所有开发期辅助工具
- 安装所有分析期附加包

用户真正需要提供的，只是满足当前正式主线运行能力的环境。
