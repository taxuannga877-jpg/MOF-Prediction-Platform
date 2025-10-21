"""
集成模型实现
Ensemble Model - Combining CGCNN and MOFormer
"""

import numpy as np
from typing import Any, Dict, List, Union
from pathlib import Path

from .base_model import BaseModel
from .cgcnn_model import CGCNNModel
from .moformer_model import MOFormerModel


class EnsembleModel(BaseModel):
    """
    集成模型 - 结合CGCNN和MOFormer的预测
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 初始化子模型
        cgcnn_config = config.get('cgcnn', {})
        moformer_config = config.get('moformer', {})
        
        self.cgcnn = CGCNNModel(cgcnn_config)
        self.moformer = MOFormerModel(moformer_config)
        
        # 集成权重
        self.cgcnn_weight = config.get('cgcnn_weight', 0.6)
        self.moformer_weight = config.get('moformer_weight', 0.4)
        self.ensemble_method = config.get('ensemble_method', 'weighted')  # weighted, average, max, learned
        
        # 如果使用学习的权重
        if self.ensemble_method == 'learned':
            import torch.nn as nn
            self.weight_network = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.Softmax(dim=-1)
            )
    
    def build_model(self):
        """构建集成模型"""
        self.cgcnn.build_model()
        self.moformer.build_model()
        
        total_params = self.cgcnn._count_parameters() + self.moformer._count_parameters()
        print(f"✅ 集成模型已构建")
        print(f"   CGCNN参数: {self.cgcnn._count_parameters():,}")
        print(f"   MOFormer参数: {self.moformer._count_parameters():,}")
        print(f"   总参数量: {total_params:,}")
    
    def train(self, train_data: Dict[str, Any], val_data: Dict[str, Any] = None,
              epochs: int = 100, **kwargs) -> Dict[str, List[float]]:
        """
        训练集成模型
        
        Args:
            train_data: {
                'structures': {id: Structure},  # for CGCNN
                'mofids': [...],                # for MOFormer
                'targets': {id: value} or [...]
            }
        """
        print("=" * 50)
        print("训练CGCNN模型...")
        print("=" * 50)
        
        # 训练CGCNN
        if 'structures' in train_data:
            cgcnn_train = {
                'structures': train_data['structures'],
                'targets': train_data['targets'] if isinstance(train_data['targets'], dict) 
                          else {f"mof_{i}": v for i, v in enumerate(train_data['targets'])}
            }
            
            cgcnn_val = None
            if val_data and 'structures' in val_data:
                cgcnn_val = {
                    'structures': val_data['structures'],
                    'targets': val_data['targets'] if isinstance(val_data['targets'], dict)
                              else {f"mof_{i}": v for i, v in enumerate(val_data['targets'])}
                }
            
            self.cgcnn.train(cgcnn_train, cgcnn_val, epochs=epochs, **kwargs)
        
        print("\n" + "=" * 50)
        print("训练MOFormer模型...")
        print("=" * 50)
        
        # 训练MOFormer
        if 'mofids' in train_data:
            moformer_train = {
                'mofids': train_data['mofids'],
                'targets': train_data['targets'] if isinstance(train_data['targets'], list)
                          else list(train_data['targets'].values())
            }
            
            moformer_val = None
            if val_data and 'mofids' in val_data:
                moformer_val = {
                    'mofids': val_data['mofids'],
                    'targets': val_data['targets'] if isinstance(val_data['targets'], list)
                              else list(val_data['targets'].values())
                }
            
            self.moformer.train(moformer_train, moformer_val, epochs=epochs//2, **kwargs)
        
        self.is_trained = True
        
        # 合并训练历史
        self.history = {
            'cgcnn': self.cgcnn.history,
            'moformer': self.moformer.history,
        }
        
        print("\n" + "=" * 50)
        print("✅ 集成模型训练完成！")
        print("=" * 50)
        
        return self.history
    
    def predict(self, data: Dict[str, Any]) -> Union[Dict[str, float], List[float]]:
        """
        使用集成模型预测
        
        Args:
            data: {
                'structures': {id: Structure} (optional),
                'mofids': [...] (optional)
            }
            
        Returns:
            预测结果
        """
        cgcnn_preds = None
        moformer_preds = None
        
        # CGCNN预测
        if 'structures' in data and self.cgcnn.is_trained:
            print("🔮 CGCNN预测中...")
            cgcnn_preds = self.cgcnn.predict(data['structures'])
        
        # MOFormer预测
        if 'mofids' in data and self.moformer.is_trained:
            print("🔮 MOFormer预测中...")
            moformer_preds = self.moformer.predict(data['mofids'])
        
        # 集成预测
        if cgcnn_preds is not None and moformer_preds is not None:
            return self._ensemble_predictions(cgcnn_preds, moformer_preds)
        elif cgcnn_preds is not None:
            print("⚠️  仅使用CGCNN预测")
            return cgcnn_preds
        elif moformer_preds is not None:
            print("⚠️  仅使用MOFormer预测")
            return moformer_preds
        else:
            raise ValueError("没有可用的模型进行预测")
    
    def _ensemble_predictions(self, cgcnn_preds: Union[Dict, List], 
                            moformer_preds: Union[Dict, List]) -> Union[Dict, List]:
        """组合两个模型的预测"""
        
        if isinstance(cgcnn_preds, dict) and isinstance(moformer_preds, list):
            # 转换为相同格式
            moformer_dict = {k: v for k, v in zip(cgcnn_preds.keys(), moformer_preds)}
            moformer_preds = moformer_dict
        
        if isinstance(cgcnn_preds, dict):
            ensemble_preds = {}
            for key in cgcnn_preds.keys():
                if key in moformer_preds:
                    ensemble_preds[key] = self._combine_single(
                        cgcnn_preds[key], 
                        moformer_preds[key]
                    )
                else:
                    ensemble_preds[key] = cgcnn_preds[key]
            return ensemble_preds
        
        else:
            # 列表格式
            ensemble_preds = [
                self._combine_single(c, m) 
                for c, m in zip(cgcnn_preds, moformer_preds)
            ]
            return ensemble_preds
    
    def _combine_single(self, cgcnn_pred: float, moformer_pred: float) -> float:
        """组合单个预测值"""
        if self.ensemble_method == 'weighted':
            return (self.cgcnn_weight * cgcnn_pred + 
                   self.moformer_weight * moformer_pred)
        
        elif self.ensemble_method == 'average':
            return (cgcnn_pred + moformer_pred) / 2.0
        
        elif self.ensemble_method == 'max':
            return max(cgcnn_pred, moformer_pred)
        
        elif self.ensemble_method == 'min':
            return min(cgcnn_pred, moformer_pred)
        
        else:
            # 默认加权平均
            return (self.cgcnn_weight * cgcnn_pred + 
                   self.moformer_weight * moformer_pred)
    
    def save_model(self, path: Union[str, Path]):
        """保存集成模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 分别保存子模型
        self.cgcnn.save_model(path / 'cgcnn_model.pth')
        self.moformer.save_model(path / 'moformer_model.pth')
        
        # 保存集成配置
        import json
        config_path = path / 'ensemble_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'cgcnn_weight': self.cgcnn_weight,
                'moformer_weight': self.moformer_weight,
                'ensemble_method': self.ensemble_method,
                'is_trained': self.is_trained,
            }, f, indent=2)
        
        print(f"✅ 集成模型已保存到: {path}")
    
    def load_model(self, path: Union[str, Path]):
        """加载集成模型"""
        path = Path(path)
        
        # 加载子模型
        self.cgcnn.load_model(path / 'cgcnn_model.pth')
        self.moformer.load_model(path / 'moformer_model.pth')
        
        # 加载集成配置
        import json
        config_path = path / 'ensemble_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.cgcnn_weight = config.get('cgcnn_weight', 0.6)
                self.moformer_weight = config.get('moformer_weight', 0.4)
                self.ensemble_method = config.get('ensemble_method', 'weighted')
                self.is_trained = config.get('is_trained', True)
        
        print(f"✅ 集成模型已从 {path} 加载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取集成模型信息"""
        return {
            'model_type': 'EnsembleModel',
            'is_trained': self.is_trained,
            'ensemble_method': self.ensemble_method,
            'cgcnn_weight': self.cgcnn_weight,
            'moformer_weight': self.moformer_weight,
            'cgcnn_info': self.cgcnn.get_model_info(),
            'moformer_info': self.moformer.get_model_info(),
        }
    
    def evaluate(self, data: Dict[str, Any], targets: Union[Dict, List]) -> Dict[str, Any]:
        """
        评估集成模型
        
        Args:
            data: 测试数据
            targets: 真实值
            
        Returns:
            评估指标
        """
        predictions = self.predict(data)
        
        # 转换为numpy数组
        if isinstance(predictions, dict):
            pred_array = np.array(list(predictions.values()))
        else:
            pred_array = np.array(predictions)
        
        if isinstance(targets, dict):
            target_array = np.array(list(targets.values()))
        else:
            target_array = np.array(targets)
        
        # 计算指标
        mae = np.mean(np.abs(pred_array - target_array))
        rmse = np.sqrt(np.mean((pred_array - target_array) ** 2))
        
        ss_res = np.sum((target_array - pred_array) ** 2)
        ss_tot = np.sum((target_array - np.mean(target_array)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # 如果两个子模型都可用，也评估它们
        metrics = {
            'ensemble': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
            }
        }
        
        if 'structures' in data and self.cgcnn.is_trained:
            cgcnn_preds = self.cgcnn.predict(data['structures'])
            if isinstance(cgcnn_preds, dict):
                cgcnn_array = np.array(list(cgcnn_preds.values()))
            else:
                cgcnn_array = np.array(cgcnn_preds)
            
            metrics['cgcnn'] = {
                'mae': float(np.mean(np.abs(cgcnn_array - target_array))),
                'rmse': float(np.sqrt(np.mean((cgcnn_array - target_array) ** 2))),
                'r2': float(1 - np.sum((target_array - cgcnn_array) ** 2) / ss_tot),
            }
        
        if 'mofids' in data and self.moformer.is_trained:
            moformer_preds = self.moformer.predict(data['mofids'])
            moformer_array = np.array(moformer_preds)
            
            metrics['moformer'] = {
                'mae': float(np.mean(np.abs(moformer_array - target_array))),
                'rmse': float(np.sqrt(np.mean((moformer_array - target_array) ** 2))),
                'r2': float(1 - np.sum((target_array - moformer_array) ** 2) / ss_tot),
            }
        
        return metrics

