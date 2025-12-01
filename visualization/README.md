# 可视化模块说明

本模块提供了假新闻检测数据集的网络图可视化功能，包括单个传播图的可视化和节点数分布统计分析。

## 使用前准备

### 数据预处理

在首次使用可视化功能之前，请确保数据已经正确预处理。如果遇到以下错误：
- `RuntimeError: The 'data' object was created by an older version of PyG...`
- `ModuleNotFoundError` 或数据加载失败

请运行数据重处理脚本：

```bash
# 重新处理所有特征类型的数据（推荐）
uv run python utils/reprocess_all_features.py

# 或者只处理特定特征（可视化默认使用 spacy）
# 直接运行可视化脚本会自动处理，但重新处理可以确保兼容性
```

**注意：** 如果数据已经正确处理过，通常不需要重新运行。只有在遇到版本兼容性问题时才需要。

## 文件结构

- `graph_visualizer.py`: 单个传播网络图可视化（支持多种布局算法）
- `node_distribution.py`: 真假新闻节点数分布对比分析
- `depth_distribution.py`: 传播深度（最大层数）对比分析
- `nonroot_centralization.py`: 非根节点传播集中度分析（大V vs 水军矩阵）
- `utils.py`: 工具函数（PyG 数据转换、度计算等）

## 核心功能

### 1. 可视化单个传播图

使用 `graph_visualizer.py` 可以可视化数据集中任意一个传播网络图。

**命令行使用：**

```bash
# 可视化索引为 5 的图，使用层次化布局
uv run python visualization/graph_visualizer.py --idx 5 --layout hierarchical

# 可视化索引为 0 的图，使用 spring 布局，并显示节点标签
uv run python visualization/graph_visualizer.py --idx 0 --layout spring --show_labels

# 指定保存路径
uv run python visualization/graph_visualizer.py --idx 10 --layout kamada_kawai --save_path figures/my_graph.png
```

**支持的布局算法：**
- `spring`: 弹簧布局（默认）
- `kamada_kawai`: Kamada-Kawai 布局
- `hierarchical`: 层次化布局（根节点在顶部，按 BFS 层次排列）
- `circular`: 圆形布局
- `shell`: 壳形布局
- `spectral`: 谱布局

**Python 代码使用：**

```python
from visualization.graph_visualizer import visualize_single_graph
from data_loader import FNNDataset, ToUndirected

# 加载数据集
dataset = FNNDataset(root='data', feature='spacy', name='politifact', transform=ToUndirected())

# 可视化第 0 个图
data = dataset[0]
visualize_single_graph(
    data,
    title="Example Propagation Network",
    save_path='figures/graph_0.png',
    layout='hierarchical',
    show_labels=False
)
```

### 2. 节点数分布对比分析

使用 `node_distribution.py` 可以分析并可视化真假新闻图的节点数分布差异。

**命令行使用：**

```bash
# 分析 politifact 数据集的节点数分布
uv run python visualization/node_distribution.py --dataset politifact

# 指定保存路径
uv run python visualization/node_distribution.py --dataset politifact --save_path figures/my_distribution.png
```

**输出内容：**
- 控制台输出：详细的统计信息（均值、标准差、最小值、最大值、中位数）
- 图片文件：包含两个子图
  - 条形图：真假新闻平均节点数对比（含标准差误差棒）
  - 直方图：节点数分布对比

### 3. 传播深度对比分析

使用 `depth_distribution.py` 统计真假新闻传播树的最大深度，并输出柱状图 + 直方图：

```bash
# 生成 politifact 深度分布图
uv run python visualization/depth_distribution.py --dataset politifact

# 指定特征或保存路径
uv run python visualization/depth_distribution.py --dataset politifact --feature bert --save_path figures/depth_custom.png
```

**输出内容：**
- 控制台：真假新闻传播深度的统计信息（均值、标准差、中位数等）
- 图片：左侧柱状图对比平均最大深度，右侧直方图展示深度分布

### 4. 非根节点集中度（大V vs 水军矩阵）

`nonroot_centralization.py` 用于比较真假新闻在「非根节点」上的传播集中度：

```bash
uv run python visualization/nonroot_centralization.py --dataset politifact
```

主要指标：
- **Top-k 非根出度占比**：少数非根节点是否贡献了大部分传播？
- **Gini 系数**：非根出度是否高度不均衡？
- **中度出度节点占比**（例如 3~10）：是否存在大量中度活跃账号，更像“水军矩阵”。

## 选择要可视化的图

在可视化之前，建议先查看 `graph_info.txt` 文件来选择感兴趣的图。该文件包含每个图的基本信息：

```
索引    节点数    标签
0       497       True
1       21        True
...
157     109       Fake
```

**使用建议：**
1. 查看 `graph_info.txt` 了解所有图的基本信息
2. 根据节点数范围选择代表性图（例如：小图、中图、大图）
3. 平衡选择真新闻和假新闻图进行对比
4. 使用命令行工具快速生成可视化


## 可视化特性

- **节点颜色**：根节点（新闻）为红色，用户节点为蓝色
- **节点大小**：根据节点度动态调整
- **自环处理**：可视化时自动移除自环边，使图更清晰
- **中文支持**：图表标题和标签支持中文显示

## 输出目录

所有可视化结果默认保存在项目根目录下的 `figures/` 目录中。

