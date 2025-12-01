"""
分析真假新闻传播图的深度分布对比
"""
import os
import sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径，便于直接运行脚本
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'code'))

from data_loader import FNNDataset, ToUndirected

# 设置 matplotlib 支持中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def compute_tree_depth(data):
    """
    计算传播树的最大深度与平均深度。
    默认假设 PyG Data 中的节点 0 是根节点（新闻本体）。
    """
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes

    # 构建无向邻接表，忽略自环
    adj = [[] for _ in range(num_nodes)]
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src == dst:
            continue
        adj[src].append(dst)
        adj[dst].append(src)

    root = 0
    visited = [False] * num_nodes
    queue = deque([(root, 0)])
    visited[root] = True

    max_depth = 0
    depth_sum = 0

    while queue:
        node, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        depth_sum += depth
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append((nb, depth + 1))

    avg_depth = depth_sum / num_nodes if num_nodes > 0 else 0
    return max_depth, avg_depth


def analyze_depth_distribution(dataset_name='politifact', feature='spacy', save_path=None):
    """
    计算真假新闻传播图的深度分布并绘图。
    """
    data_root = os.path.join(project_root, 'data')
    dataset = FNNDataset(
        root=data_root,
        feature=feature,
        empty=False,
        name=dataset_name,
        transform=ToUndirected()
    )

    true_max_depths, fake_max_depths = [], []
    true_avg_depths, fake_avg_depths = [], []

    print(f"共 {len(dataset)} 个传播图，开始计算深度…")

    for idx in range(len(dataset)):
        data = dataset[idx]
        max_depth, avg_depth = compute_tree_depth(data)
        if data.y.item() == 0:  # True news
            true_max_depths.append(max_depth)
            true_avg_depths.append(avg_depth)
        else:  # Fake news
            fake_max_depths.append(max_depth)
            fake_avg_depths.append(avg_depth)

    def summarize(values, label):
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        print(f"\n{label}:")
        print(f"  数量: {len(values)}")
        print(f"  平均值: {mean_val:.2f}")
        print(f"  标准差: {std_val:.2f}")
        print(f"  中位数: {median_val:.2f}")
        print(f"  最小值: {min(values):.0f}，最大值: {max(values):.0f}")
        return mean_val, std_val

    print("=" * 60)
    print(f"传播深度分析 - {dataset_name} 数据集")
    print("=" * 60)
    true_mean, true_std = summarize(true_max_depths, "真新闻最大深度")
    fake_mean, fake_std = summarize(fake_max_depths, "假新闻最大深度")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：最大深度对比柱状图
    ax1 = axes[0]
    categories = ['True News', 'Fake News']
    means = [true_mean, fake_mean]
    stds = [true_std, fake_std]
    colors = ['#4472C4', '#FF4444']

    bars = ax1.bar(
        categories,
        means,
        yerr=stds,
        capsize=10,
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    ax1.set_ylabel('Maximum Depth', fontsize=12, fontweight='bold')
    ax1.set_title('Maximum Depth Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + std + 0.5,
            f'Mean: {mean:.1f}\nStd: {std:.1f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # 右图：最大深度分布直方图
    ax2 = axes[1]
    max_depth_all = true_max_depths + fake_max_depths
    bins = np.linspace(0, max(max_depth_all) + 2, 25)
    ax2.hist(
        true_max_depths,
        bins=bins,
        alpha=0.6,
        label='True News',
        color='#4472C4',
        edgecolor='black',
        linewidth=0.5
    )
    ax2.hist(
        fake_max_depths,
        bins=bins,
        alpha=0.6,
        label='Fake News',
        color='#FF4444',
        edgecolor='black',
        linewidth=0.5
    )
    ax2.set_xlabel('Maximum Depth', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Graph Count', fontsize=12, fontweight='bold')
    ax2.set_title('Depth Distribution Histogram', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(project_root, 'figures', f'depth_distribution_{dataset_name}.png')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    rel_path = os.path.relpath(save_path, start=os.getcwd())
    print(f"\n✓ 深度分布可视化已保存到: {rel_path}")
    plt.close()

    return {
        'true': {'max_depths': true_max_depths, 'avg_depths': true_avg_depths},
        'fake': {'max_depths': fake_max_depths, 'avg_depths': fake_avg_depths},
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="分析传播深度分布")
    parser.add_argument('--dataset', type=str, default='politifact', help='数据集名称')
    parser.add_argument('--feature', type=str, default='spacy', help='特征类型')
    parser.add_argument('--save_path', type=str, default=None, help='保存路径')

    args = parser.parse_args()
    analyze_depth_distribution(args.dataset, args.feature, args.save_path)

