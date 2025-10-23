"""
K折交叉验证
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class CrossValidator:
    """K折交叉验证器"""
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        初始化交叉验证器
        
        Args:
            n_splits: 折数
            shuffle: 是否打乱数据
            random_state: 随机种子
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        self.results = {
            'fold_scores': [],
            'fold_predictions': [],
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
    
    def cross_validate(
        self,
        model_creator: Callable,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行K折交叉验证
        
        Args:
            model_creator: 模型创建函数
            X: 特征数据
            y: 标签数据
            model_params: 模型参数
            verbose: 是否显示详细信息
        
        Returns:
            交叉验证结果
        """
        if model_params is None:
            model_params = {}
        
        all_predictions = np.zeros_like(y, dtype=float)
        all_true = np.zeros_like(y, dtype=float)
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Fold {fold_idx + 1}/{self.n_splits}")
                print(f"{'='*60}")
            
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建并训练模型
            model = model_creator(**model_params)
            
            # 检查模型类型
            if hasattr(model, 'train'):
                # 自定义模型接口
                model.train(X_train, y_train, X_val, y_val)
                y_pred = model.predict(X_val)
            else:
                # sklearn接口
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            # 计算指标
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-10))) * 100
            
            fold_metrics = {
                'fold': fold_idx + 1,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'n_samples': len(y_val)
            }
            
            self.results['fold_metrics'].append(fold_metrics)
            self.results['fold_scores'].append(r2)
            
            # 保存预测结果
            all_predictions[val_idx] = y_pred
            all_true[val_idx] = y_val
            
            if verbose:
                print(f"📊 Fold {fold_idx + 1} 结果:")
                print(f"   MAE:  {mae:.4f}")
                print(f"   RMSE: {rmse:.4f}")
                print(f"   R²:   {r2:.4f}")
                print(f"   MAPE: {mape:.2f}%")
        
        # 保存所有预测
        self.results['fold_predictions'] = {
            'y_true': all_true,
            'y_pred': all_predictions
        }
        
        # 计算平均指标
        metrics_df = pd.DataFrame(self.results['fold_metrics'])
        for metric in ['mae', 'rmse', 'r2', 'mape']:
            self.results['mean_metrics'][metric] = metrics_df[metric].mean()
            self.results['std_metrics'][metric] = metrics_df[metric].std()
        
        if verbose:
            print(f"\n{'='*60}")
            print("📊 交叉验证总结")
            print(f"{'='*60}")
            print(f"平均 MAE:  {self.results['mean_metrics']['mae']:.4f} ± {self.results['std_metrics']['mae']:.4f}")
            print(f"平均 RMSE: {self.results['mean_metrics']['rmse']:.4f} ± {self.results['std_metrics']['rmse']:.4f}")
            print(f"平均 R²:   {self.results['mean_metrics']['r2']:.4f} ± {self.results['std_metrics']['r2']:.4f}")
            print(f"平均 MAPE: {self.results['mean_metrics']['mape']:.2f}% ± {self.results['std_metrics']['mape']:.2f}%")
            print(f"{'='*60}\n")
        
        return self.results
    
    def get_cv_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取交叉验证的所有预测结果"""
        return (
            self.results['fold_predictions']['y_true'],
            self.results['fold_predictions']['y_pred']
        )
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """获取每折的指标DataFrame"""
        return pd.DataFrame(self.results['fold_metrics'])
    
    def get_summary(self) -> Dict[str, float]:
        """获取汇总统计"""
        return {
            'mean_mae': self.results['mean_metrics']['mae'],
            'std_mae': self.results['std_metrics']['mae'],
            'mean_rmse': self.results['mean_metrics']['rmse'],
            'std_rmse': self.results['std_metrics']['rmse'],
            'mean_r2': self.results['mean_metrics']['r2'],
            'std_r2': self.results['std_metrics']['r2'],
            'mean_mape': self.results['mean_metrics']['mape'],
            'std_mape': self.results['std_metrics']['mape']
        }


def stratified_kfold_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_bins: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    分层K折分割（用于回归任务）
    将连续目标变量分箱后进行分层采样
    
    Args:
        X: 特征
        y: 标签
        n_splits: 折数
        n_bins: 分箱数
    
    Returns:
        训练/验证索引列表
    """
    # 将连续值分箱
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    
    for train_idx, val_idx in skf.split(X, y_binned):
        splits.append((train_idx, val_idx))
    
    return splits

