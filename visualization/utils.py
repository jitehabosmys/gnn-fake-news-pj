"""
工具函数：PyG Data 与 NetworkX 之间的转换
"""
import torch
import networkx as nx
from torch_geometric.data import Data


def pyg_to_networkx(data: Data, to_undirected: bool = True, remove_self_loops: bool = False):
	"""
	将 PyTorch Geometric 的 Data 对象转换为 NetworkX 图
	
	Args:
		data: PyG Data 对象
		to_undirected: 是否转换为无向图（默认True）
		remove_self_loops: 是否移除自环（默认False）
	
	Returns:
		G: NetworkX 图对象
	"""
	G = nx.DiGraph() if not to_undirected else nx.Graph()
	
	# 添加节点
	num_nodes = data.x.size(0) if data.x is not None else (data.edge_index.max().item() + 1)
	G.add_nodes_from(range(num_nodes))
	
	# 添加边
	edge_index = data.edge_index.cpu().numpy()
	for i in range(edge_index.shape[1]):
		src, dst = edge_index[0, i], edge_index[1, i]
		if remove_self_loops and src == dst:
			continue
		G.add_edge(int(src), int(dst))
	
	return G


def get_node_degrees(data: Data):
	"""
	计算每个节点的度（连接数）
	
	Args:
		data: PyG Data 对象
	
	Returns:
		degrees: 节点度数组
	"""
	edge_index = data.edge_index.cpu().numpy()
	num_nodes = data.x.size(0) if data.x is not None else (edge_index.max().item() + 1)
	
	degrees = torch.zeros(num_nodes, dtype=torch.long)
	for i in range(edge_index.shape[1]):
		src, dst = edge_index[0, i], edge_index[1, i]
		degrees[src] += 1  # 出度
		degrees[dst] += 1   # 入度
	
	return degrees.numpy()
	
