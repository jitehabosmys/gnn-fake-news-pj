"""
重新处理所有特征类型的数据
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'code'))

from data_loader import FNNDataset, ToUndirected

def reprocess_all_features():
	"""重新处理所有特征类型的数据"""
	print("=" * 60)
	print("重新处理所有特征类型的数据")
	print("=" * 60)
	
	# 使用相对于项目根目录的路径
	data_root = os.path.join(project_root, 'data')
	
	# 所有需要处理的特征类型
	features = ['bert', 'spacy', 'profile', 'content']
	
	print(f"\n数据根目录: {data_root}")
	print(f"需要处理的特征类型: {features}\n")
	
	datasets = {}
	
	for feature in features:
		print(f"{'='*60}")
		print(f"处理 {feature.upper()} 特征数据...")
		print(f"{'='*60}")
		
		try:
			# 创建数据集（这会触发 process() 方法）
			dataset = FNNDataset(root=data_root, feature=feature, empty=False, 
							name='politifact', transform=ToUndirected())
			
			datasets[feature] = dataset
			
			print(f"✓ {feature.upper()} 特征处理完成！")
			print(f"  - 数据集大小: {len(dataset)} 个图")
			print(f"  - 节点特征维度: {dataset.num_node_attributes}")
			print(f"  - 类别数: {dataset.num_classes}")
			print()
			
		except Exception as e:
			print(f"✗ {feature.upper()} 特征处理失败: {e}")
			print()
			continue
	
	print("=" * 60)
	print("所有特征类型处理完成！")
	print("=" * 60)
	
	# 验证文件
	print("\n验证生成的文件:")
	processed_dir = os.path.join(data_root, 'politifact', 'processed')
	if os.path.exists(processed_dir):
		files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
		for f in sorted(files):
			file_path = os.path.join(processed_dir, f)
			size_mb = os.path.getsize(file_path) / (1024 * 1024)
			print(f"  - {f} ({size_mb:.2f} MB)")
	
	return datasets

if __name__ == '__main__':
	reprocess_all_features()

