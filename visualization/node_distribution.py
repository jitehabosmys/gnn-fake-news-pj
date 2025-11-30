"""
分析并可视化真假新闻图的节点数分布对比
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'code'))

from data_loader import FNNDataset, ToUndirected

# 设置matplotlib支持中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def analyze_node_distribution(dataset_name='politifact', feature='spacy', save_path=None):
	"""
	分析并可视化节点数分布
	
	Args:
		dataset_name: 数据集名称
		feature: 特征类型
		save_path: 保存路径
	"""
	data_root = os.path.join(project_root, 'data')
	dataset = FNNDataset(root=data_root, feature=feature, empty=False, 
						name=dataset_name, transform=ToUndirected())
	
	# 收集节点数
	true_node_counts = []
	fake_node_counts = []
	
	for i in range(len(dataset)):
		data = dataset[i]
		num_nodes = data.num_nodes
		if data.y.item() == 0:  # True
			true_node_counts.append(num_nodes)
		else:  # Fake
			fake_node_counts.append(num_nodes)
	
	# 计算统计量
	true_mean = np.mean(true_node_counts)
	true_std = np.std(true_node_counts)
	fake_mean = np.mean(fake_node_counts)
	fake_std = np.std(fake_node_counts)
	
	print("=" * 60)
	print(f"节点数分布分析 - {dataset_name} 数据集")
	print("=" * 60)
	print(f"\n真新闻图:")
	print(f"  数量: {len(true_node_counts)}")
	print(f"  平均节点数: {true_mean:.2f}")
	print(f"  标准差: {true_std:.2f}")
	print(f"  最小值: {min(true_node_counts)}")
	print(f"  最大值: {max(true_node_counts)}")
	print(f"  中位数: {np.median(true_node_counts):.2f}")
	
	print(f"\n假新闻图:")
	print(f"  数量: {len(fake_node_counts)}")
	print(f"  平均节点数: {fake_mean:.2f}")
	print(f"  标准差: {fake_std:.2f}")
	print(f"  最小值: {min(fake_node_counts)}")
	print(f"  最大值: {max(fake_node_counts)}")
	print(f"  中位数: {np.median(fake_node_counts):.2f}")
	
	# 可视化
	fig, axes = plt.subplots(1, 2, figsize=(14, 6))
	
	# 子图1: 条形图对比
	ax1 = axes[0]
	categories = ['True News', 'Fake News']
	means = [true_mean, fake_mean]
	stds = [true_std, fake_std]
	colors = ['#4472C4', '#FF4444']
	
	bars = ax1.bar(categories, means, yerr=stds, capsize=10, 
				   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
	ax1.set_ylabel('Average Node Count', fontsize=12, fontweight='bold')
	ax1.set_title('Average Node Count Comparison', fontsize=13, fontweight='bold')
	ax1.grid(axis='y', alpha=0.3, linestyle='--')
	
	# 在柱状图上标注数值
	for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
		height = bar.get_height()
		ax1.text(bar.get_x() + bar.get_width()/2., height + std + 5,
				f'Mean: {mean:.1f}\nStd: {std:.1f}',
				ha='center', va='bottom', fontsize=10, fontweight='bold')
	
	# 子图2: 分布直方图
	ax2 = axes[1]
	bins = np.linspace(0, max(max(true_node_counts), max(fake_node_counts)) + 20, 30)
	ax2.hist(true_node_counts, bins=bins, alpha=0.6, label='True News', 
			 color='#4472C4', edgecolor='black', linewidth=0.5)
	ax2.hist(fake_node_counts, bins=bins, alpha=0.6, label='Fake News', 
			 color='#FF4444', edgecolor='black', linewidth=0.5)
	ax2.set_xlabel('Node Count', fontsize=12, fontweight='bold')
	ax2.set_ylabel('Graph Count', fontsize=12, fontweight='bold')
	ax2.set_title('Node Count Distribution', fontsize=13, fontweight='bold')
	ax2.legend(fontsize=11)
	ax2.grid(axis='y', alpha=0.3, linestyle='--')
	
	plt.tight_layout()
	
	# 保存
	if save_path is None:
		save_path = os.path.join(project_root, 'figures', f'node_distribution_{dataset_name}.png')
	
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	project_cwd = os.getcwd()
	rel_save_path = os.path.relpath(save_path, start=project_cwd)
	print(f"\n✓ 可视化已保存到: {rel_save_path}")
	plt.close()
	
	# 返回统计结果
	return {
		'true': {'counts': true_node_counts, 'mean': true_mean, 'std': true_std},
		'fake': {'counts': fake_node_counts, 'mean': fake_mean, 'std': fake_std}
	}

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="分析节点数分布")
	parser.add_argument('--dataset', type=str, default='politifact', help='数据集名称')
	parser.add_argument('--feature', type=str, default='spacy', help='特征类型')
	parser.add_argument('--save_path', type=str, default=None, help='保存路径')
	
	args = parser.parse_args()
	analyze_node_distribution(args.dataset, args.feature, args.save_path)

