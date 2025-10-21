"""
MOF 数据集类
PyTorch Dataset classes for MOF data
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import json


class MOFDataset(Dataset):
    """MOF 数据集基类"""
    
    def __init__(
        self,
        data: Union[pd.DataFrame, List, Dict],
        target_property: str = 'band_gap',
        transform=None
    ):
        """
        初始化数据集
        
        Args:
            data: 原始数据
            target_property: 目标属性名称
            transform: 数据转换函数
        """
        self.data = data
        self.target_property = target_property
        self.transform = transform
        self.samples = []
        self._prepare_data()
    
    def _prepare_data(self):
        """准备数据"""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class CGCNNDataset(MOFDataset):
    """用于 CGCNN 的数据集（基于晶体结构）"""
    
    def __init__(
        self,
        data: Union[pd.DataFrame, Dict],
        target_property: str = 'band_gap',
        structure_dir: Optional[Path] = None,
        max_num_neighbors: int = 12,
        radius: float = 8.0
    ):
        """
        初始化 CGCNN 数据集
        
        Args:
            data: 包含 MOF ID 和目标属性的数据
            target_property: 目标属性
            structure_dir: CIF 文件目录
            max_num_neighbors: 最大邻居数
            radius: 截断半径
        """
        self.structure_dir = Path(structure_dir) if structure_dir else None
        self.max_num_neighbors = max_num_neighbors
        self.radius = radius
        super().__init__(data, target_property)
    
    def _prepare_data(self):
        """准备 CGCNN 数据"""
        if isinstance(self.data, pd.DataFrame):
            for idx, row in self.data.iterrows():
                if self.target_property in row and not pd.isna(row[self.target_property]):
                    sample = {
                        'id': row.get('qmof_id', str(idx)),
                        'target': float(row[self.target_property]),
                        'structure': row.get('structure', None)
                    }
                    self.samples.append(sample)
        elif isinstance(self.data, dict):
            for mof_id, mof_data in self.data.items():
                if self.target_property in mof_data:
                    sample = {
                        'id': mof_id,
                        'target': float(mof_data[self.target_property]),
                        'structure': mof_data.get('structure', None)
                    }
                    self.samples.append(sample)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 构建图结构
        graph_data = self._structure_to_graph(sample)
        
        return graph_data
    
    def _structure_to_graph(self, sample: Dict) -> Dict:
        """
        将晶体结构转换为图数据
        
        Args:
            sample: 包含结构信息的样本
            
        Returns:
            图数据字典
        """
        # 简化版本：使用虚拟数据
        # 实际应用中需要从 CIF 文件或 Structure 对象构建
        
        # 假设有 10 个原子
        num_atoms = 10
        
        # 原子特征（原子序数的 one-hot 编码简化版）
        atom_features = torch.randn(num_atoms, 92)  # 92 = 元素周期表元素数
        
        # 边索引（邻接关系）
        # 这里简化为全连接图
        edge_index = []
        edge_attr = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    edge_index.append([i, j])
                    # 边特征（距离）
                    edge_attr.append([np.random.rand()])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return {
            'x': atom_features,  # 节点特征
            'edge_index': edge_index,  # 边索引
            'edge_attr': edge_attr,  # 边特征
            'y': torch.tensor([sample['target']], dtype=torch.float),  # 目标值
            'id': sample['id']
        }


class MOFormerDataset(MOFDataset):
    """用于 MOFormer 的数据集（基于文本/MOFid）"""
    
    def __init__(
        self,
        data: Union[pd.DataFrame, List],
        target_property: str = 'band_gap',
        max_length: int = 512
    ):
        """
        初始化 MOFormer 数据集
        
        Args:
            data: 包含 MOFid 和目标属性的数据
            target_property: 目标属性
            max_length: 最大序列长度
        """
        self.max_length = max_length
        super().__init__(data, target_property)
    
    def _prepare_data(self):
        """准备 MOFormer 数据"""
        if isinstance(self.data, pd.DataFrame):
            for idx, row in self.data.iterrows():
                if self.target_property in row and not pd.isna(row[self.target_property]):
                    sample = {
                        'id': row.get('qmof_id', str(idx)),
                        'target': float(row[self.target_property]),
                        'mofid': row.get('mofid', ''),
                        'smiles': row.get('smiles', ''),
                    }
                    self.samples.append(sample)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 文本编码（简化版本）
        # 实际应用中需要使用 Transformer tokenizer
        text = sample.get('mofid', '') or sample.get('smiles', '') or str(sample['id'])
        
        # 简化的编码：使用字符级编码
        encoded = self._encode_text(text)
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'y': torch.tensor([sample['target']], dtype=torch.float),
            'id': sample['id']
        }
    
    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        简化的文本编码
        
        Args:
            text: 输入文本
            
        Returns:
            编码后的数据
        """
        # 字符到索引的映射（简化版）
        vocab = {char: idx for idx, char in enumerate(set(text))}
        vocab['<PAD>'] = len(vocab)
        
        # 编码
        indices = [vocab.get(char, vocab['<PAD>']) for char in text[:self.max_length]]
        
        # 填充
        padding_length = self.max_length - len(indices)
        input_ids = indices + [vocab['<PAD>']] * padding_length
        attention_mask = [1] * len(indices) + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_length], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:self.max_length], dtype=torch.long)
        }


def create_data_splits(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    划分数据集
    
    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from torch.utils.data import random_split
    
    # 设置随机种子
    torch.manual_seed(random_seed)
    
    # 计算数量
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 划分
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset


def collate_graph_batch(batch: List[Dict]) -> Dict:
    """
    图数据的批处理整合函数
    
    Args:
        batch: 批次数据列表
        
    Returns:
        整合后的批次数据
    """
    # 整合节点特征
    x_list = [item['x'] for item in batch]
    edge_index_list = [item['edge_index'] for item in batch]
    edge_attr_list = [item['edge_attr'] for item in batch]
    y_list = [item['y'] for item in batch]
    ids = [item['id'] for item in batch]
    
    # 批处理索引
    batch_indices = []
    cumsum = 0
    for i, x in enumerate(x_list):
        batch_indices.extend([i] * len(x))
        # 调整边索引
        edge_index_list[i] = edge_index_list[i] + cumsum
        cumsum += len(x)
    
    return {
        'x': torch.cat(x_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_attr': torch.cat(edge_attr_list, dim=0),
        'y': torch.cat(y_list, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'ids': ids
    }


def collate_text_batch(batch: List[Dict]) -> Dict:
    """
    文本数据的批处理整合函数
    
    Args:
        batch: 批次数据列表
        
    Returns:
        整合后的批次数据
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    y = torch.cat([item['y'] for item in batch])
    ids = [item['id'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'y': y,
        'ids': ids
    }

