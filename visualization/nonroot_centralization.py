"""
分析真假新闻在「非根节点」上的传播集中度（是否依赖少数大V）
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'code'))

from data_loader import FNNDataset

# 设置 matplotlib 支持中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def gini_coefficient(x: np.ndarray) -> float:
    """
    计算一维非负数组的 Gini 系数，反映出度是否集中在少数节点上。
    """
    x = x.astype(float)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # G = 1 + 1/n - 2 * sum(cum) / (n * cum[-1])
    g = 1.0 + 1.0 / n - 2.0 * np.sum(cum) / (n * cum[-1])
    return float(g)


def analyze_nonroot_centralization(
    dataset_name='politifact',
    feature='spacy',
    save_path=None,
    top_k=3,
    mid_low=3,
    mid_high=10,
):
    """
    只考虑非根节点，衡量传播是集中（依赖少数大V）还是更去中心化（水军矩阵）。

    指标：
    - top_k_share: 非根出度最高的 top_k 节点占非根总出度的比例（越高越集中）
    - gini: 非根出度的 Gini 系数（越高越集中）
    - mid_degree_ratio: 出度在 [mid_low, mid_high] 内的非根节点占比（中度活跃群体，多则更像“水军矩阵”）
    """
    data_root = os.path.join(project_root, 'data')
    dataset = FNNDataset(root=data_root, feature=feature, empty=False, name=dataset_name)

    true_topk, fake_topk = [], []
    true_gini, fake_gini = [], []
    true_mid_ratio, fake_mid_ratio = [], []

    print(f"共 {len(dataset)} 个传播图，开始统计非根节点集中度…")

    for data in dataset:
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.num_nodes
        if num_nodes <= 1:
            continue

        # 计算出度并去掉自环
        out_deg = np.zeros(num_nodes, dtype=int)
        for src, dst in zip(edge_index[0], edge_index[1]):
            if src == dst:
                continue
            out_deg[src] += 1

        # 只看非根节点（节点 1..n-1）
        nonroot_deg = out_deg[1:]
        nonroot_total = int(nonroot_deg.sum())
        nonroot_count = nonroot_deg.size

        if nonroot_total == 0 or nonroot_count == 0:
            topk_share = 0.0
            gini = 0.0
            mid_ratio = 0.0
        else:
            # Top-k 出度占比
            topk = min(top_k, nonroot_count)
            topk_vals = np.sort(nonroot_deg)[-topk:]
            topk_share = float(topk_vals.sum() / nonroot_total)
            # Gini 系数
            gini = gini_coefficient(nonroot_deg)
            # 中度出度节点占比
            mid_mask = (nonroot_deg >= mid_low) & (nonroot_deg <= mid_high)
            mid_ratio = float(mid_mask.sum() / nonroot_count)

        if data.y.item() == 0:  # True
            true_topk.append(topk_share)
            true_gini.append(gini)
            true_mid_ratio.append(mid_ratio)
        else:  # Fake
            fake_topk.append(topk_share)
            fake_gini.append(gini)
            fake_mid_ratio.append(mid_ratio)

    def summarize(values, label):
        values = np.array(values, dtype=float)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        median_val = float(np.median(values))
        print(f"\n{label}:")
        print(f"  样本数: {len(values)}")
        print(f"  均值: {mean_val:.3f}")
        print(f"  标准差: {std_val:.3f}")
        print(f"  中位数: {median_val:.3f}")
        print(f"  最小值: {values.min():.3f}，最大值: {values.max():.3f}")
        return mean_val, std_val

    print("=" * 60)
    print(f"非根节点集中度分析 - {dataset_name} 数据集")
    print("=" * 60)

    t_topk_mean, t_topk_std = summarize(true_topk, f"真新闻 Top-{top_k} 非根出度占比")
    f_topk_mean, f_topk_std = summarize(fake_topk, f"假新闻 Top-{top_k} 非根出度占比")

    t_gini_mean, t_gini_std = summarize(true_gini, "真新闻非根出度 Gini")
    f_gini_mean, f_gini_std = summarize(fake_gini, "假新闻非根出度 Gini")

    t_mid_mean, t_mid_std = summarize(true_mid_ratio, f"真新闻中度出度({mid_low}~{mid_high})占比")
    f_mid_mean, f_mid_std = summarize(fake_mid_ratio, f"假新闻中度出度({mid_low}~{mid_high})占比")

    # 可视化：参考 node_distribution，画 3 组柱状图（Top-k占比 / Gini / 中度占比）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    categories = ['True', 'Fake']
    colors = ['#4472C4', '#FF4444']

    def bar_plot(ax, true_mean, true_std, fake_mean, fake_std, ylabel, title):
        means = [true_mean, fake_mean]
        stds = [true_std, fake_std]
        bars = ax.bar(
            categories,
            means,
            yerr=stds,
            capsize=8,
            color=colors,
            alpha=0.75,
            edgecolor='black',
            linewidth=1.2,
        )
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, mean, std in zip(bars, means, stds):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                h + std,
                f"Mean: {mean:.2f}\nStd: {std:.2f}",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
            )

    bar_plot(
        axes[0],
        t_topk_mean,
        t_topk_std,
        f_topk_mean,
        f_topk_std,
        ylabel=f"Top-{top_k} Non-root Share",
        title="Top-k Non-root Out-degree Share",
    )
    bar_plot(
        axes[1],
        t_gini_mean,
        t_gini_std,
        f_gini_mean,
        f_gini_std,
        ylabel="Gini (Non-root Out-degree)",
        title="Non-root Out-degree Gini",
    )
    bar_plot(
        axes[2],
        t_mid_mean,
        t_mid_std,
        f_mid_mean,
        f_mid_std,
        ylabel=f"Ratio (deg {mid_low}-{mid_high})",
        title="Medium-degree Node Ratio",
    )

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            project_root,
            'figures',
            f'nonroot_centralization_{dataset_name}.png',
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    rel_save = os.path.relpath(save_path, start=os.getcwd())
    print(f"\n✓ 非根集中度可视化已保存到: {rel_save}")
    plt.close()

    return {
        'true': {
            'topk_share': true_topk,
            'gini': true_gini,
            'mid_ratio': true_mid_ratio,
        },
        'fake': {
            'topk_share': fake_topk,
            'gini': fake_gini,
            'mid_ratio': fake_mid_ratio,
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="分析非根节点传播集中度")
    parser.add_argument('--dataset', type=str, default='politifact', help='数据集名称')
    parser.add_argument('--feature', type=str, default='spacy', help='特征类型')
    parser.add_argument('--save_path', type=str, default=None, help='保存路径')
    parser.add_argument('--top_k', type=int, default=3, help='Top-k 高出度节点个数')
    parser.add_argument('--mid_low', type=int, default=3, help='中度出度下界（含）')
    parser.add_argument('--mid_high', type=int, default=10, help='中度出度上界（含）')

    args = parser.parse_args()
    analyze_nonroot_centralization(
        dataset_name=args.dataset,
        feature=args.feature,
        save_path=args.save_path,
        top_k=args.top_k,
        mid_low=args.mid_low,
        mid_high=args.mid_high,
    )


