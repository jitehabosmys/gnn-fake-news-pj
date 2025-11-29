import os.path as osp
import warnings
warnings.filterwarnings("ignore")

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array

import random
import numpy as np
import scipy.sparse as sp

"""
	Functions to help load the graph data
"""

def read_file(folder, name, dtype=None):
	path = osp.join(folder, '{}.txt'.format(name))
	return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
	"""
	PyG util code to create graph batches

	# Example usage of the split function:
	#
	# batch = torch.tensor([0, 0, 1, 1, 1])  # 2 graphs, 5 nodes in total
	# data.x = torch.randn(5, 10)  # Node features
	# data.edge_index = torch.tensor([[0,1,2,3,3,4], [1,0,3,2,4,3]]).t()
	# data.y = torch.tensor([0, 1])  # Graph-level labels: graph 1 = fake, graph 2 = real
	#
	# After processing with the split function, the slices dict will be:
	# slices = {
	#     'edge_index': torch.tensor([0, 2, 6]),    # Edge slices for each graph
	#     'x': torch.tensor([0, 2, 5]),             # Node feature slices per graph
	#     'y': torch.tensor([0, 1, 2])              # Label slices (graph-level)
	# }
	"""

	node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
	node_slice = torch.cat([torch.tensor([0]), node_slice])

	row, _ = data.edge_index
	edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
	edge_slice = torch.cat([torch.tensor([0]), edge_slice])

	# Edge indices should start at zero for every graph.
	data.edge_index -= node_slice[batch[row]].unsqueeze(0)
	data.__num_nodes__ = torch.bincount(batch).tolist()

	slices = {'edge_index': edge_slice}
	if data.x is not None:
		slices['x'] = node_slice
	if data.edge_attr is not None:
		slices['edge_attr'] = edge_slice
	if data.y is not None:
		if data.y.size(0) == batch.size(0):
			slices['y'] = node_slice
		else:
			slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

	return data, slices


def read_graph_data(folder, feature):
	"""
	PyG util code to create PyG data instance from raw graph data

	Args:
		folder (str): Path to the folder containing raw data files.
		feature (str): Node feature type (e.g., 'content', 'profile', 'bert', etc.)

	Returns:
		data (Data): PyG Data object containing the graph data.
		slices (dict): Slices dictionary for PyG batching.
	"""

	# Load node feature matrix from .npz file (rows: nodes, cols: features)
	node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')

	# Load edge list from file, then transpose to shape [2, num_edges]
	edge_index = read_file(folder, 'A', torch.long).t()

	# Load array indicating which graph each node belongs to
	node_graph_id = np.load(folder + 'node_graph_id.npy')

	# Load graph labels (e.g., fake/real for each graph)
	graph_labels = np.load(folder + 'graph_labels.npy')

	edge_attr = None  # No edge attributes used

	# Convert sparse node features to dense tensor (float)
	x = torch.from_numpy(node_attributes.todense()).to(torch.float)
	node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
	y = torch.from_numpy(graph_labels).to(torch.long)

	# Remap graph labels to a contiguous range starting from zero
	_, y = y.unique(sorted=True, return_inverse=True)

	# Get the number of nodes (from feature matrix or edge_index)
	num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

	# Add self-loops for all nodes
	edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

	# Coalesce edge_index (remove duplicates, sort, etc.)
	edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

	# Create a global Data object that concatenates all graphs
	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

	# Obtain slice indices for separating each graph within the batch
	data, slices = split(data, node_graph_id)

	return data, slices


class ToUndirected:
	def __init__(self):
		"""
		PyG util code to transform the graph to the undirected graph
		"""
		pass

	def __call__(self, data):
		edge_attr = None
		edge_index = to_undirected(data.edge_index, data.x.size(0))
		num_nodes = edge_index.max().item() + 1 if data.x is None else data.x.size(0)
		# edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
		edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
		data.edge_index = edge_index
		data.edge_attr = edge_attr
		return data


class DropEdge:
	def __init__(self, tddroprate, budroprate):
		"""
		Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
		1) Generate TD and BU edge indices
		2) Drop out edges
		Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
		"""
		self.tddroprate = tddroprate
		self.budroprate = budroprate

	def __call__(self, data):
		edge_index = data.edge_index

		if self.tddroprate > 0:
			row = list(edge_index[0])
			col = list(edge_index[1])
			length = len(row)
			poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
			poslist = sorted(poslist)
			row = list(np.array(row)[poslist])
			col = list(np.array(col)[poslist])
			new_edgeindex = [row, col]
		else:
			new_edgeindex = edge_index

		burow = list(edge_index[1])
		bucol = list(edge_index[0])
		if self.budroprate > 0:
			length = len(burow)
			poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
			poslist = sorted(poslist)
			row = list(np.array(burow)[poslist])
			col = list(np.array(bucol)[poslist])
			bunew_edgeindex = [row, col]
		else:
			bunew_edgeindex = [burow, bucol]

		data.edge_index = torch.LongTensor(new_edgeindex)
		data.BU_edge_index = torch.LongTensor(bunew_edgeindex)
		data.root = torch.FloatTensor(data.x[0])
		data.root_index = torch.LongTensor([0])

		return data


class FNNDataset(InMemoryDataset):
	r"""
		Graph dataset class based on FakeNewsNet data, for PyTorch Geometric.

		Args:
			root (string): Root directory where the dataset should be saved.
			name (string): The name of the dataset.
			feature (string, optional): Type of node features to use (default: 'spacy').
			empty (bool, optional): If True, create an empty dataset for special use (default: False).
			transform (callable, optional): Function/transform to apply to each data object before every access (default: None).
			pre_transform (callable, optional): Function/transform applied to each data object *before* saving to disk (default: None).
			pre_filter (callable, optional): Function that filters unwanted data objects before saving (default: None).
	"""

	def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
		# Dataset initialization: set properties and load from processed files if not empty
		self.name = name
		self.root = root
		self.feature = feature
		super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
		if not empty:
			# Load processed data from local file
			self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_dir(self):
		# Return the directory containing the raw, unprocessed data files
		name = 'raw/'
		return osp.join(self.root, self.name, name)

	@property
	def processed_dir(self):
		# Return the directory for processed PyG data objects
		name = 'processed/'
		return osp.join(self.root, self.name, name)

	@property
	def num_node_attributes(self):
		# Get number of node attributes (features). If empty, return 0.
		if self.data.x is None:
			return 0
		return self.data.x.size(1)

	@property
	def raw_file_names(self):
		# List of required raw .npy files to construct the dataset
		names = ['node_graph_id', 'graph_labels']
		return ['{}.npy'.format(name) for name in names]

	@property
	def processed_file_names(self):
		# Filenames for processed binary (pt) files. If pre_filter is applied, add tag to the name.
		if self.pre_filter is None:
			return f'{self.name[:3]}_data_{self.feature}.pt'
		else:
			return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

	def download(self):
		# Download is not implemented to avoid automatic network fetching
		raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

	def process(self):
		"""
		Processes raw data files to create graph objects and save them.
		Includes optional pre_filter and pre_transform for selective processing.
		"""
		# Read raw graph data based on the selected features
		self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

		# If a pre_filter is provided, only keep graphs that pass the filter
		if self.pre_filter is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [data for data in data_list if self.pre_filter(data)]
			self.data, self.slices = self.collate(data_list)

		# Apply any pre_transform to every data object (for example, data augmentation, etc.)
		if self.pre_transform is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [self.pre_transform(data) for data in data_list]
			self.data, self.slices = self.collate(data_list)

		# Save the final processed data in processed_dir
		torch.save((self.data, self.slices), self.processed_paths[0])

	def __repr__(self):
		# String representation: dataset name and number of graphs
		return '{}({})'.format(self.name, len(self))