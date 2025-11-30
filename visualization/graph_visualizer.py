"""
基础网络图可视化模块
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.data import Data

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False   

# --------
from utils import pyg_to_networkx, get_node_degrees
# --------

def visualize_single_graph(data: Data, 
						   title: str = None,
						   save_path: str = None,
						   figsize: tuple = (12, 10),
						   node_size_scale: float = 100,
						   layout: str = 'spring',
						   show_labels: bool = False):
	"""
	可视化单个传播图
	
	Args:
		data: PyG Data 对象（单个图）
		title: 图标题（如果为None，会根据标签自动生成）
		save_path: 保存路径（如果为None，则显示而不保存）
		figsize: 图片大小
		node_size_scale: 节点大小缩放因子
		layout: 布局算法 ('spring', 'kamada_kawai', 'hierarchical')
		show_labels: 是否显示节点标签
	"""
	# 转换为 NetworkX 图（移除自环，避免可视化时出现"空心圆"效果）
	G = pyg_to_networkx(data, to_undirected=True, remove_self_loops=True)
	
	# 获取根节点索引 0
	root_idx = 0
	
	# 计算节点度
	degrees = get_node_degrees(data)
	
	# 设置布局
	if layout == 'spring':
		pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
	elif layout == 'kamada_kawai':
		pos = nx.kamada_kawai_layout(G)
	elif layout == 'circular':
		# 圆形布局：所有节点均匀分布在圆周上
		pos = nx.circular_layout(G)
	elif layout == 'shell':
		# 壳形布局：节点分布在同心圆上
		pos = nx.shell_layout(G)
	elif layout == 'spectral':
		# 谱布局：基于图的拉普拉斯矩阵特征向量
		pos = nx.spectral_layout(G)
	elif layout == 'hierarchical':
		# 层次化布局：根节点在顶部，按BFS层次排列
		pos = {}
		# 使用 BFS 计算层次
		levels = {root_idx: 0}
		queue = [root_idx]
		while queue:
			node = queue.pop(0)
			for neighbor in G.neighbors(node):
				if neighbor not in levels:
					levels[neighbor] = levels[node] + 1
					queue.append(neighbor)
		
		# 按层次排列节点
		for node, level in levels.items():
			# 计算同一层的节点位置
			same_level_nodes = [n for n, l in levels.items() if l == level]
			idx_in_level = same_level_nodes.index(node)
			# x 和 y 是节点的平面可视化坐标，用于在层次化布局下生成节点位置
			# - y 表示第 level 层，层数越大，y 越小（顶层为 0，依次向下），每层节点放在同一水平线上
			# - x 用于将同一层的节点均匀分布在水平方向，中心对齐
			x = (idx_in_level - (len(same_level_nodes) - 1) / 2) * 2  # 在该层内的索引减去中心索引再乘以间隔，左右均匀分布
			y = -level * 2  # 层数向下，纵坐标依次减小
			pos[node] = (x, y)
	else:
		# 未知布局类型，采用 spring_layout 作为默认布局
		pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
	
	# 准备节点颜色和大小
	node_colors = []
	node_sizes = []
	
	for node in G.nodes():
		# 根节点（新闻）用红色，其他节点（用户）用蓝色
		if node == root_idx:
			node_colors.append('#FF4444')  # 红色
			node_sizes.append(node_size_scale * 2)  # 根节点更大
		else:
			node_colors.append('#4472C4')  # 蓝色
			# 节点大小根据度调整
			degree = degrees[node] if node < len(degrees) else 1
			node_sizes.append(node_size_scale * (1 + degree * 0.3))
	
	# 绘制
	plt.figure(figsize=figsize)
	
	# 绘制边
	nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
	
	# 绘制节点
	nx.draw_networkx_nodes(G, pos, 
						   node_color=node_colors,
						   node_size=node_sizes,
						   alpha=0.9,  
						   edgecolors='none', 
						   linewidths=0)  
	
	# 可选：显示节点标签
	if show_labels:
		labels = {node: str(node) for node in G.nodes()}
		nx.draw_networkx_labels(G, pos, labels, font_size=8)
	
	# 设置标题
	if title is None:
		label = 'Fake' if data.y.item() == 1 else 'True'
		title = f"News Propagation Network (Label: {label}, Nodes: {data.num_nodes})"
	
	plt.title(title, fontsize=14, fontweight='bold')
	plt.axis('off')
	plt.tight_layout()
	
	# 添加图例
	from matplotlib.patches import Patch
	legend_elements = [
		Patch(facecolor='#FF4444', label='News Node (Root)'),
		Patch(facecolor='#4472C4', label='User Node')
	]
	plt.legend(handles=legend_elements, loc='upper right')
	
	# 保存或显示
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		project_cwd = os.getcwd()
		rel_save_path = os.path.relpath(save_path, start=project_cwd)
		print(f"完成。图已保存到: {rel_save_path}")
	else:
		plt.show()
	
	plt.close()

if __name__ == '__main__':
	import argparse
	import os
	import sys

	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	sys.path.insert(0, os.path.join(project_root, 'code'))
	sys.path.insert(0, project_root)

	# 以后全部用本地（相对根目录）的方式导入 data_loader 和 utils
	from data_loader import FNNDataset, ToUndirected

	parser = argparse.ArgumentParser(description="可视化单个传播图")
	parser.add_argument('--idx', type=int, required=True, help='要可视化的图在数据集中的索引（从0开始）')
	parser.add_argument('--dataset', type=str, default='politifact', help='数据集名称')
	parser.add_argument('--title', type=str, default=None, help='图标题')
	parser.add_argument('--layout', type=str, default='spring', choices=['spring', 'kamada_kawai', 'circular', 'shell', 'spectral', 'hierarchical'], help='布局算法')
	parser.add_argument('--show_labels', action='store_true', help='是否显示节点标签')
	parser.add_argument('--save_path', type=str, default=None, help='图片保存路径（默认为figures/graph_{idx}.png）')

	args = parser.parse_args()

	data_root = os.path.join(project_root, 'data')

	dataset = FNNDataset(
		root=data_root, 
		feature='spacy', 
		empty=False, 
		name=args.dataset, 
		transform=ToUndirected()
	)

	if args.idx < 0 or args.idx >= len(dataset):
		print(f"错误: 请求索引 {args.idx} 超出数据集范围 (0 ~ {len(dataset)-1})")
		sys.exit(1)

	data = dataset[args.idx]
	label = 'Fake' if data.y.item() == 1 else 'True'
	title = args.title if args.title is not None else f"News Propagation Network (Label: {label}, Nodes: {data.num_nodes})"

	# 处理保存路径
	if args.save_path is not None:
		save_path = args.save_path
	else:
		# 默认保存到根目录下figures/
		save_dir = os.path.join(project_root, 'figures')
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f'graph_{args.idx}_{label.lower()}.png')

	visualize_single_graph(
		data,
		title=title,
		save_path=save_path,
		layout=args.layout,
		show_labels=args.show_labels
	)


