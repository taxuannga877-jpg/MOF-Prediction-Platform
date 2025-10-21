"""
CGCNN模型完整实现
Crystal Graph Convolutional Neural Network - Full Implementation
集成原始CGCNN代码用于实际MOF预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres

from .base_model import BaseModel


class GaussianDistance(object):
    """
    高斯距离扩展
    将原子间距离扩展为高斯基函数
    """
    def __init__(self, dmin=0, dmax=8, step=0.2, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)


class AtomInitializer(object):
    """
    原子特征初始化器
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        
    def get_atom_len(self):
        return len(list(self._embedding.values())[0])


class CrystalDataset(Dataset):
    """
    晶体结构数据集
    """
    def __init__(self, structures, targets, radius=8, max_num_nbr=12):
        """
        Args:
            structures: {mof_id: Structure} 字典
            targets: {mof_id: property_value} 字典
            radius: 邻居搜索半径
            max_num_nbr: 最大邻居数
        """
        self.structures = structures
        self.targets = targets
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
        
        # 获取共同的ID
        self.ids = list(set(structures.keys()) & set(targets.keys()))
        
        # 初始化原子特征（使用元素周期表特征）
        self._init_atom_features()
        
    def _init_atom_features(self):
        """初始化原子特征向量"""
        # 创建原子特征字典（基于原子序数）
        self.atom_features = {}
        for atom_num in range(1, 101):  # 1-100号元素
            # 使用one-hot编码 + 原子序数归一化
            fea = np.zeros(92)  # 92维特征
            if atom_num <= 92:
                fea[atom_num - 1] = 1.0
            self.atom_features[atom_num] = fea
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        mof_id = self.ids[idx]
        structure = self.structures[mof_id]
        target = self.targets[mof_id]
        
        # 提取原子特征
        atom_fea = np.vstack([
            self.atom_features.get(site.specie.Z, np.zeros(92))
            for site in structure
        ])
        
        # 获取邻居信息
        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        # 构建邻接矩阵和距离矩阵
        nbr_fea_idx = []
        nbr_fea = []
        
        for nbrs in all_nbrs:
            if len(nbrs) < self.max_num_nbr:
                nbrs_padded = nbrs + [nbrs[0]] * (self.max_num_nbr - len(nbrs))
            else:
                nbrs_padded = nbrs[:self.max_num_nbr]
            
            nbr_fea_idx.append([nbr[2] for nbr in nbrs_padded])
            nbr_fea.append([nbr[1] for nbr in nbrs_padded])
        
        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        
        return (atom_fea, nbr_fea, nbr_fea_idx), target, mof_id


class ConvLayer(nn.Module):
    """
    卷积层 - CGCNN核心组件
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        
        # 获取邻居原子特征
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        
        # 拼接特征
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        
        return out


class CrystalGraphConvNet(nn.Module):
    """
    完整的CGCNN网络
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, n_classes=2):
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        
        # 原子特征嵌入
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        # 卷积层
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        # 池化层
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        # 全连接层
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                            for _ in range(n_h-1)])
        
        # 输出层
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, n_classes)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        
        # Dropout
        self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx=None):
        atom_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # 池化
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        crys_fea = self.dropout(crys_fea)
        out = self.fc_out(crys_fea)
        
        if self.classification:
            out = F.log_softmax(out, dim=1)
        
        return out

    def pooling(self, atom_fea, crystal_atom_idx=None):
        """晶体级池化"""
        if crystal_atom_idx is None:
            return torch.mean(atom_fea, dim=0, keepdim=True)
        
        summed_fea = []
        for i in range(len(crystal_atom_idx)):
            if i == 0:
                start_idx = 0
            else:
                start_idx = crystal_atom_idx[i-1]
            end_idx = crystal_atom_idx[i]
            summed_fea.append(torch.mean(atom_fea[start_idx:end_idx], dim=0))
        
        return torch.stack(summed_fea)


class CGCNNModel(BaseModel):
    """
    CGCNN模型完整实现
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.atom_fea_len = config.get('atom_fea_len', 64)
        self.nbr_fea_len = config.get('nbr_fea_len', 41)  # 高斯扩展后的长度
        self.h_fea_len = config.get('h_fea_len', 128)
        self.n_conv = config.get('n_conv', 3)
        self.n_h = config.get('n_h', 1)
        self.batch_size = config.get('batch_size', 32)
    
    def build_model(self):
        """构建完整的CGCNN模型"""
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=92,  # 原子特征维度
            nbr_fea_len=self.nbr_fea_len,
            atom_fea_len=self.atom_fea_len,
            n_conv=self.n_conv,
            h_fea_len=self.h_fea_len,
            n_h=self.n_h,
            classification=False
        )
        
        self.model.to(self.device)
        print(f"✅ CGCNN模型已构建，参数量: {self._count_parameters():,}")
    
    def collate_batch(self, dataset_list):
        """批处理函数"""
        batch_atom_fea = []
        batch_nbr_fea = []
        batch_nbr_fea_idx = []
        crystal_atom_idx = []
        batch_target = []
        batch_ids = []
        
        base_idx = 0
        for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, mof_id) in enumerate(dataset_list):
            n_i = atom_fea.shape[0]
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
            crystal_atom_idx.append(base_idx + n_i)
            batch_target.append(target)
            batch_ids.append(mof_id)
            base_idx += n_i
        
        return (torch.cat(batch_atom_fea, dim=0),
                torch.cat(batch_nbr_fea, dim=0),
                torch.cat(batch_nbr_fea_idx, dim=0),
                torch.LongTensor(crystal_atom_idx)), \
               torch.cat(batch_target, dim=0), \
               batch_ids
    
    def train(self, train_data: Dict[str, Any], val_data: Dict[str, Any] = None,
              epochs: int = 100, **kwargs) -> Dict[str, List[float]]:
        """
        训练CGCNN模型
        
        Args:
            train_data: {'structures': {id: Structure}, 'targets': {id: value}}
            val_data: 验证数据（同样格式）
            epochs: 训练轮数
        """
        if not self.model:
            self.build_model()
        
        # 创建数据集
        train_dataset = CrystalDataset(
            train_data['structures'],
            train_data['targets']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
            num_workers=0
        )
        
        if val_data:
            val_dataset = CrystalDataset(
                val_data['structures'],
                val_data['targets']
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_batch,
                num_workers=0
            )
        
        # 优化器和损失函数
        lr = kwargs.get('lr', self.config.get('lr', 0.001))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=kwargs.get('weight_decay', 0))
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_count = 0
            
            for (inputs, targets, ids) in train_loader:
                inputs = tuple(inp.to(self.device) for inp in inputs)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(*inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(targets)
                train_count += len(targets)
            
            avg_train_loss = train_loss / train_count
            self.history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            if val_data:
                self.model.eval()
                val_loss = 0
                val_count = 0
                
                with torch.no_grad():
                    for (inputs, targets, ids) in val_loader:
                        inputs = tuple(inp.to(self.device) for inp in inputs)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(*inputs)
                        loss = criterion(outputs.squeeze(), targets.squeeze())
                        
                        val_loss += loss.item() * len(targets)
                        val_count += len(targets)
                
                avg_val_loss = val_loss / val_count
                self.history['val_loss'].append(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}")
        
        self.is_trained = True
        print(f"✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")
        
        return self.history
    
    def predict(self, data: Dict[str, Structure]) -> Dict[str, float]:
        """
        预测MOF性质
        
        Args:
            data: {mof_id: Structure} 字典
            
        Returns:
            {mof_id: predicted_value} 字典
        """
        if not self.is_trained and not self.model:
            raise ValueError("模型未训练或未加载")
        
        # 创建虚拟目标值用于数据集
        dummy_targets = {k: 0.0 for k in data.keys()}
        
        dataset = CrystalDataset(data, dummy_targets)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
            num_workers=0
        )
        
        self.model.eval()
        predictions = {}
        
        with torch.no_grad():
            for (inputs, targets, ids) in loader:
                inputs = tuple(inp.to(self.device) for inp in inputs)
                outputs = self.model(*inputs)
                outputs = outputs.cpu().numpy().squeeze()
                
                for mof_id, pred in zip(ids, outputs):
                    predictions[mof_id] = float(pred)
        
        return predictions
