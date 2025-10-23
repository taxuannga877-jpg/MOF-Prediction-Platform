"""
模型集成策略
支持 Voting, Stacking, Blending
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel:
    """模型集成"""
    
    def __init__(
        self,
        base_models: List[Tuple[str, Any]],
        ensemble_method: str = 'voting',
        meta_learner: Optional[Any] = None
    ):
        """
        初始化集成模型
        
        Args:
            base_models: 基模型列表，格式 [(name, model), ...]
            ensemble_method: 集成方法 ('voting', 'stacking', 'blending')
            meta_learner: 元学习器（用于stacking）
        """
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner if meta_learner else Ridge()
        self.ensemble_model = None
        self.blending_models = None
        self.history = {
            'train_scores': [],
            'val_scores': [],
            'base_model_scores': {}
        }
        
        self._create_ensemble()
    
    def _create_ensemble(self):
        """创建集成模型"""
        if self.ensemble_method == 'voting':
            self.ensemble_model = VotingRegressor(
                estimators=self.base_models,
                n_jobs=-1
            )
        
        elif self.ensemble_method == 'stacking':
            self.ensemble_model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=self.meta_learner,
                n_jobs=-1
            )
        
        elif self.ensemble_method == 'blending':
            # Blending需要特殊处理
            self.blending_models = [model for _, model in self.base_models]
        
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        blend_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        训练集成模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            blend_ratio: Blending的训练/混合数据比例
        
        Returns:
            训练历史
        """
        if self.ensemble_method in ['voting', 'stacking']:
            # Voting和Stacking使用sklearn的标准接口
            self.ensemble_model.fit(X_train, y_train)
            
            # 计算各基模型的分数
            for name, model in self.base_models:
                if hasattr(model, 'fit'):
                    train_score = model.score(X_train, y_train)
                    self.history['base_model_scores'][name] = {
                        'train_score': train_score
                    }
                    if X_val is not None and y_val is not None:
                        val_score = model.score(X_val, y_val)
                        self.history['base_model_scores'][name]['val_score'] = val_score
            
            # 集成模型分数
            train_score = self.ensemble_model.score(X_train, y_train)
            self.history['train_scores'].append(train_score)
            
            if X_val is not None and y_val is not None:
                val_score = self.ensemble_model.score(X_val, y_val)
                self.history['val_scores'].append(val_score)
        
        elif self.ensemble_method == 'blending':
            # Blending: 分割训练数据
            split_idx = int(len(X_train) * blend_ratio)
            X_train_base = X_train[:split_idx]
            y_train_base = y_train[:split_idx]
            X_train_meta = X_train[split_idx:]
            y_train_meta = y_train[split_idx:]
            
            # 训练基模型
            meta_features_train = []
            for i, (name, model) in enumerate(self.base_models):
                print(f"训练基模型 {i+1}/{len(self.base_models)}: {name}")
                model.fit(X_train_base, y_train_base)
                
                # 生成meta特征
                meta_pred_train = model.predict(X_train_meta).reshape(-1, 1)
                meta_features_train.append(meta_pred_train)
                
                # 记录基模型性能
                train_score = model.score(X_train_base, y_train_base)
                self.history['base_model_scores'][name] = {
                    'train_score': train_score
                }
                if X_val is not None and y_val is not None:
                    val_score = model.score(X_val, y_val)
                    self.history['base_model_scores'][name]['val_score'] = val_score
            
            # 训练meta学习器
            meta_features_train = np.hstack(meta_features_train)
            self.meta_learner.fit(meta_features_train, y_train_meta)
            
            # 记录集成模型性能
            train_score = self.score(X_train, y_train)
            self.history['train_scores'].append(train_score)
            
            if X_val is not None and y_val is not None:
                val_score = self.score(X_val, y_val)
                self.history['val_scores'].append(val_score)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.ensemble_method in ['voting', 'stacking']:
            return self.ensemble_model.predict(X)
        
        elif self.ensemble_method == 'blending':
            # 获取基模型预测
            meta_features = []
            for _, model in self.base_models:
                pred = model.predict(X).reshape(-1, 1)
                meta_features.append(pred)
            
            meta_features = np.hstack(meta_features)
            return self.meta_learner.predict(meta_features)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """获取所有基模型的预测"""
        predictions = {}
        
        if self.ensemble_method in ['voting', 'stacking']:
            for name, model in self.base_models:
                predictions[name] = model.predict(X)
        
        elif self.ensemble_method == 'blending':
            for name, model in self.base_models:
                predictions[name] = model.predict(X)
        
        return predictions
    
    def save(self, path: str):
        """保存集成模型"""
        import joblib
        joblib.dump({
            'ensemble_model': self.ensemble_model,
            'base_models': self.base_models,
            'ensemble_method': self.ensemble_method,
            'meta_learner': self.meta_learner,
            'blending_models': self.blending_models,
            'history': self.history
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """加载集成模型"""
        import joblib
        data = joblib.load(path)
        instance = cls(
            base_models=data['base_models'],
            ensemble_method=data['ensemble_method'],
            meta_learner=data['meta_learner']
        )
        instance.ensemble_model = data['ensemble_model']
        instance.blending_models = data['blending_models']
        instance.history = data['history']
        return instance


def create_auto_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_types: List[str] = None,
    ensemble_method: str = 'stacking'
) -> EnsembleModel:
    """
    自动创建集成模型
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        model_types: 要使用的模型类型列表
        ensemble_method: 集成方法
    
    Returns:
        集成模型实例
    """
    from .traditional_ml import TraditionalMLModel, get_available_models
    
    if model_types is None:
        # 默认使用表现较好的模型组合
        model_types = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
    
    # 获取可用模型
    available = get_available_models()
    model_types = [mt for mt in model_types if available.get(mt, False)]
    
    # 创建基模型
    base_models = []
    for model_type in model_types:
        model = TraditionalMLModel(model_type=model_type)
        base_models.append((model_type, model.model))
    
    # 创建集成模型
    ensemble = EnsembleModel(
        base_models=base_models,
        ensemble_method=ensemble_method
    )
    
    return ensemble

