# 社交网络挖掘 - 假新闻检测项目

基于图神经网络（GNN）的假新闻检测项目，参考 [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews) 实现。

## 项目概述

This is a description.

## 文件结构

```
pj/
├── code/                    # 核心代码（来自参考仓库）
│   ├── data_loader.py      # 数据加载和预处理
│   ├── gnn.py              # UPFD-GCN/GAT/GraphSAGE 模型
│   ├── gcnfn.py            # GCNFN 模型
│   ├── gnncl.py            # GNN-CL 模型
│   ├── bigcn.py            # Bi-GCN 模型
│   ├── eval_helper.py      # 模型评估工具
│   └── run                 # 模型训练脚本
│
├── visualization/           # 可视化模块（已实现）
│   ├── graph_visualizer.py # 单个传播图可视化
│   ├── node_distribution.py # 节点数分布统计分析
│   ├── utils.py            # 可视化工具函数
│   └── README.md           # 可视化模块文档
│
├── utils/                  # 工具脚本
│   ├── reprocess_all_features.py  # 数据重处理工具
│   └── profile_feature.py         # 用户特征生成
│
├── data/                   # 数据集（需自行下载，见下方说明）
│   └── politifact/        # Politifact 数据集
│       ├── raw/           # 原始数据文件
│       └── processed/    # 处理后的 PyG 数据文件
│
├── figures/               # 可视化结果输出目录
├── references/           # 参考资料和项目需求文档
├── graph_info.txt        # 图信息索引文件（用于选择可视化目标）
│
├── pyproject.toml        # 项目配置文件
├── requirements.txt      # Python 依赖
└── main.py              # 项目入口文件
```

## 已实现功能

### 1. 数据加载和预处理（原有） ✅
- **数据加载器** (`code/data_loader.py`)
  - 支持多种节点特征类型：`bert`, `spacy`, `profile`, `content`
  - 自动处理原始数据并转换为 PyG 格式
  - 修复了 PyTorch 2.6+ 兼容性问题

### 2. 网络图可视化 ✅
- **单个传播图可视化** (`visualization/graph_visualizer.py`)
  - 支持多种布局算法：spring, kamada_kawai, hierarchical, circular, shell, spectral
  - 自动区分根节点（新闻）和用户节点
  - 根据节点度动态调整节点大小
  - 自动移除自环边，使可视化更清晰

- **节点数分布分析** (`visualization/node_distribution.py`)
  - 真假新闻节点数统计对比
  - 生成均值、方差、中位数等统计指标
  - 可视化条形图和分布直方图

### 3. 图信息索引 ✅
- **图信息文件** (`graph_info.txt`)
  - 包含每个图的索引、节点数、标签信息
  - 方便选择代表性图进行可视化

### 4. 工具脚本 ✅
- **数据重处理工具** (`utils/reprocess_all_features.py`)
  - 用于解决版本兼容性问题
  - 支持批量处理所有特征类型

## 快速开始

### 环境设置

项目使用 `uv` 作为包管理器。确保已安装 `uv`：

**使用 uv（推荐）：**
```bash
# 安装依赖（uv 会自动处理依赖顺序）
uv sync
```

**使用 pip：**
```bash
# 安装依赖
pip install -r requirements.txt
```

**注意：** PyTorch 扩展包（torch-sparse, torch-scatter 等）需要先安装 torch。如果遇到构建错误，可以：
1. 先安装 torch: `uv pip install torch` 或 `pip install torch`
2. 然后运行 `uv sync` 或 `pip install -r requirements.txt`

主要依赖：
- `torch` 和 `torch_geometric` (图神经网络框架)
- `torch-sparse`, `torch-scatter`, `torch-cluster`, `torch-spline-conv` (PyG 扩展)
- `networkx` (可视化)
- `matplotlib` (绘图)
- `numpy`, `scipy` (数据处理)
- `scikit-learn` (评估工具)

### 数据准备

1. **下载数据集**：
   - 从 [Code Ocean](https://codeocean.com/capsule/7305473/tree/v1) 下载数据
   - 将数据放在 `data/politifact/raw/` 目录下

2. **数据预处理**（首次使用或遇到兼容性问题时）：
   ```bash
   uv run python utils/reprocess_all_features.py
   ```

### 使用可视化功能

请参考 [`visualization/README.md`](visualization/README.md)。

## 模型训练（待进行）

项目包含多个 GNN 模型实现，训练脚本位于 `code/run`。模型包括：
- UPFD-GCN / UPFD-GAT / UPFD-GraphSAGE
- GCNFN
- GNN-CL
- Bi-GCN

**模型训练功能已经包含在原仓库的实现中。需要考虑在其基础上进行复现或增强。**

## 项目状态

- ✅ 数据加载和预处理
- ✅ 网络图可视化
- ✅ 统计分析功能
- ⏳ 模型训练和评估（待完善）
- ⏳ 更多可视化功能（待扩展）

## 引用与致谢

本项目基于 [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews) 项目实现，参考了其代码实现和数据处理方法。

### 原论文引用

如果使用本项目或参考代码，请引用以下论文：

```bibtex
@inproceedings{dou2021user,
  title={User Preference-aware Fake News Detection},
  author={Dou, Yingtong and Shu, Kai and Xia, Congying and Yu, Philip S. and Sun, Lichao},
  booktitle={Proceedings of the 44nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```

### 代码来源

- **核心代码**：本项目中的 `code/` 目录下的代码来自 [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews) 仓库
- **数据集**：使用 UPFD (User Preference-aware Fake News Detection) 框架提供的数据集
- **可视化模块**：本项目新增的可视化和统计分析功能

## 参考资料

- 项目需求文档：`references/ref.md`
- 参考仓库：https://github.com/safe-graph/GNN-FakeNews
- 数据集来源：https://codeocean.com/capsule/7305473/tree/v1
- 原论文：https://arxiv.org/pdf/2005.00625.pdf


