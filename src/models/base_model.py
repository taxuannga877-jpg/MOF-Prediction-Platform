"""
基础模型类
Base Model Class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import torch
import numpy as np


class BaseModel(ABC):
    """所有模型的基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        self.config = config or {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.history = {'train_loss': [], 'val_loss': []}
    
    @abstractmethod
    def build_model(self):
        """构建模型架构"""
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: Any,
        val_data: Any = None,
        epochs: int = 100,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            epochs: 训练轮数
            **kwargs: 其他参数
            
        Returns:
            训练历史记录
        """
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """
        预测
        
        Args:
            data: 输入数据
            
        Returns:
            预测结果
        """
        pass
    
    def save_model(self, path: Union[str, Path]):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config,
            'history': self.history,
            'is_trained': self.is_trained,
        }
        
        torch.save(checkpoint, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: Union[str, Path]):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint.get('config', {})
        self.history = checkpoint.get('history', {})
        self.is_trained = checkpoint.get('is_trained', False)
        
        # 构建模型并加载权重
        if not self.model:
            self.build_model()
        
        if checkpoint.get('model_state_dict'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已从 {path} 加载")
    
    def evaluate(self, data: Any, targets: np.ndarray) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data: 测试数据
            targets: 真实值
            
        Returns:
            评估指标字典
        """
        predictions = self.predict(data)
        
        # 计算指标
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # R²分数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
        }
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'device': str(self.device),
            'config': self.config,
            'num_parameters': self._count_parameters() if self.model else 0,
        }
        
        return info
    
    def _count_parameters(self) -> int:
        """统计模型参数数量"""
        if self.model:
            return sum(p.numel() for p in self.model.parameters())
        return 0
    
    def to_device(self, device: Union[str, torch.device]):
        """
        移动模型到指定设备
        
        Args:
            device: 目标设备
        """
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)


