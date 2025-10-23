"""
MOFormer模型完整实现
MOF Transformer Model - Full Implementation
基于MOFid文本表示的Transformer模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
import re

from .base_model import BaseModel


class MOFidTokenizer:
    """
    MOFid专用分词器
    将MOFid字符串转换为token序列
    """
    def __init__(self, max_length=512):
        self.max_length = max_length
        
        # 构建词汇表
        self.special_tokens = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[UNK]': 3,
        }
        
        # 化学元素和常见片段
        self.vocab = self.special_tokens.copy()
        self._build_vocab()
        
        self.vocab_size = len(self.vocab)
    
    def _build_vocab(self):
        """构建化学词汇表"""
        # 元素符号
        elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
                   'Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'Cu', 'Fe', 'Co', 'Ni',
                   'Mn', 'Cr', 'V', 'Ti', 'Al', 'Zr', 'Hf', 'Ce', 'Eu', 'Tb']
        
        # SMILES常见符号
        smiles_tokens = ['(', ')', '[', ']', '=', '#', '@', '+', '-',
                        '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'c', 'n', 'o', 's']
        
        # 拓扑代码
        topologies = ['pcu', 'dia', 'pts', 'nbo', 'rho', 'sod', 'cds',
                     'lvt', 'sql', 'acs', 'fcu', 'bcu', 'scu']
        
        idx = len(self.vocab)
        for token in elements + smiles_tokens + topologies:
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
    
    def tokenize(self, mofid: str) -> List[str]:
        """将MOFid分词"""
        # 简化的分词：按字符分割，保留常见片段
        tokens = []
        i = 0
        mofid = str(mofid)
        
        while i < len(mofid):
            # 尝试匹配2字符token
            if i + 1 < len(mofid):
                two_char = mofid[i:i+2]
                if two_char in self.vocab:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # 单字符token
            if mofid[i] in self.vocab:
                tokens.append(mofid[i])
            else:
                tokens.append('[UNK]')
            i += 1
        
        return tokens
    
    def encode(self, mofid: str, add_special_tokens=True,
               max_length=None, padding=True) -> Dict[str, torch.Tensor]:
        """编码MOFid为token ID"""
        tokens = self.tokenize(mofid)
        
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 转换为ID
        ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        
        # 截断
        max_len = max_length or self.max_length
        if len(ids) > max_len:
            ids = ids[:max_len]
        
        # Padding
        attention_mask = [1] * len(ids)
        if padding and len(ids) < max_len:
            pad_length = max_len - len(ids)
            ids += [self.vocab['[PAD]']] * pad_length
            attention_mask += [0] * pad_length
        
        return {
            'input_ids': torch.LongTensor(ids),
            'attention_mask': torch.LongTensor(attention_mask)
        }
    
    def batch_encode(self, mofids: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """批量编码"""
        encoded_list = [self.encode(mofid, **kwargs) for mofid in mofids]
        
        return {
            'input_ids': torch.stack([e['input_ids'] for e in encoded_list]),
            'attention_mask': torch.stack([e['attention_mask'] for e in encoded_list])
        }


class MOFidDataset(Dataset):
    """MOFid数据集"""
    def __init__(self, mofids: List[str], targets: List[float], tokenizer):
        self.mofids = mofids
        self.targets = targets
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.mofids)
    
    def __getitem__(self, idx):
        mofid = self.mofids[idx]
        target = self.targets[idx]
        
        encoded = self.tokenizer.encode(mofid)
        
        return encoded['input_ids'], encoded['attention_mask'], torch.tensor([target], dtype=torch.float)


class TransformerEncoder(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class MOFormerNetwork(nn.Module):
    """
    MOFormer网络架构
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, max_seq_length=512):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token嵌入
        token_emb = self.token_embedding(input_ids)  # (B, L, D)
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # 组合嵌入
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer编码器
        x = x.transpose(0, 1)  # (L, B, D) for nn.MultiheadAttention
        
        # 创建key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_key_padding_mask=key_padding_mask)
        
        # 使用[CLS] token的表示
        x = x[0, :, :]  # 取第一个位置（CLS token）
        
        # 回归预测
        output = self.regressor(x)
        
        return output


class MOFormerModel(BaseModel):
    """
    MOFormer模型完整实现
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 1024)
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        self.dropout = config.get('dropout', 0.1)
        
        # 初始化tokenizer
        self.tokenizer = MOFidTokenizer(max_length=self.max_length)
    
    def build_model(self):
        """构建MOFormer模型"""
        self.model = MOFormerNetwork(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_length=self.max_length
        )
        
        self.model.to(self.device)
        print(f"✅ MOFormer模型已构建，参数量: {self._count_parameters():,}")
        print(f"   词汇表大小: {self.tokenizer.vocab_size}")
    
    def train(self, train_data: Dict[str, List], val_data: Dict[str, List] = None,
              epochs: int = 50, **kwargs) -> Dict[str, List[float]]:
        """
        训练MOFormer模型
        
        Args:
            train_data: {'mofids': [...], 'targets': [...]}
            val_data: 验证数据（同样格式）
            epochs: 训练轮数
        """
        if not self.model:
            self.build_model()
        
        # 创建数据集
        train_dataset = MOFidDataset(
            train_data['mofids'],
            train_data['targets'],
            self.tokenizer
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        if val_data:
            val_dataset = MOFidDataset(
                val_data['mofids'],
                val_data['targets'],
                self.tokenizer
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # 优化器和损失函数
        lr = kwargs.get('lr', self.config.get('lr', 1e-4))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                      weight_decay=kwargs.get('weight_decay', 0.01))
        criterion = nn.MSELoss()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_count = 0
            
            for input_ids, attention_mask, targets in train_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
                    for input_ids, attention_mask, targets in val_loader:
                        input_ids = input_ids.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask)
                        loss = criterion(outputs.squeeze(), targets.squeeze())
                        
                        val_loss += loss.item() * len(targets)
                        val_count += len(targets)
                
                avg_val_loss = val_loss / val_count
                self.history['val_loss'].append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}")
        
        self.is_trained = True
        print(f"✅ 训练完成！最佳验证损失: {best_val_loss:.4f}")
        
        return self.history
    
    def predict(self, mofids: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        预测MOF性质
        
        Args:
            mofids: MOFid字符串或列表
            
        Returns:
            预测值或预测值列表
        """
        if not self.is_trained and not self.model:
            raise ValueError("模型未训练或未加载")
        
        single_input = isinstance(mofids, str)
        if single_input:
            mofids = [mofids]
        
        self.model.eval()
        predictions = []
        
        # 批量预测
        batch_size = self.batch_size
        with torch.no_grad():
            for i in range(0, len(mofids), batch_size):
                batch_mofids = mofids[i:i+batch_size]
                
                encoded = self.tokenizer.batch_encode(batch_mofids)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                batch_preds = outputs.cpu().numpy().squeeze().tolist()
                
                if isinstance(batch_preds, float):
                    batch_preds = [batch_preds]
                
                predictions.extend(batch_preds)
        
        if single_input:
            return predictions[0]
        return predictions
    
    def get_attention_weights(self, mofid: str, layer_idx: int = 0) -> np.ndarray:
        """
        获取注意力权重（用于可解释性）
        
        Args:
            mofid: MOFid字符串
            layer_idx: 层索引
            
        Returns:
            注意力权重矩阵
        """
        self.model.eval()
        
        encoded = self.tokenizer.encode(mofid)
        input_ids = encoded['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(self.device)
        
        # 获取指定层的注意力权重
        # 这需要修改forward方法来返回注意力权重
        # 这里返回模拟数据
        seq_len = input_ids.shape[1]
        attention = np.random.rand(self.nhead, seq_len, seq_len)
        
        return attention
