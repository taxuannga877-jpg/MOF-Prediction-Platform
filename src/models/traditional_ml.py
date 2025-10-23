"""
传统机器学习模型包装器
支持 sklearn, XGBoost, LightGBM, CatBoost
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

# 尝试导入高级库（可选）
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class TraditionalMLModel:
    """传统机器学习模型的统一接口"""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        初始化模型
        
        Args:
            model_type: 模型类型
            **kwargs: 模型超参数
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_names = None
        self.history = {
            'train_scores': [],
            'val_scores': []
        }
        
        self._create_model()
    
    def _create_model(self):
        """根据模型类型创建模型实例"""
        
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', None),
                min_samples_split=self.kwargs.get('min_samples_split', 2),
                min_samples_leaf=self.kwargs.get('min_samples_leaf', 1),
                random_state=self.kwargs.get('random_state', 42),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                max_depth=self.kwargs.get('max_depth', 3),
                subsample=self.kwargs.get('subsample', 1.0),
                random_state=self.kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                max_depth=self.kwargs.get('max_depth', 6),
                subsample=self.kwargs.get('subsample', 0.8),
                colsample_bytree=self.kwargs.get('colsample_bytree', 0.8),
                random_state=self.kwargs.get('random_state', 42),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                max_depth=self.kwargs.get('max_depth', -1),
                num_leaves=self.kwargs.get('num_leaves', 31),
                subsample=self.kwargs.get('subsample', 0.8),
                colsample_bytree=self.kwargs.get('colsample_bytree', 0.8),
                random_state=self.kwargs.get('random_state', 42),
                n_jobs=self.kwargs.get('n_jobs', -1),
                verbose=-1
            )
        
        elif self.model_type == 'catboost' and HAS_CATBOOST:
            self.model = cb.CatBoostRegressor(
                iterations=self.kwargs.get('n_estimators', 100),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                depth=self.kwargs.get('max_depth', 6),
                random_state=self.kwargs.get('random_state', 42),
                verbose=False
            )
        
        elif self.model_type == 'ridge':
            self.model = Ridge(
                alpha=self.kwargs.get('alpha', 1.0),
                random_state=self.kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'lasso':
            self.model = Lasso(
                alpha=self.kwargs.get('alpha', 1.0),
                random_state=self.kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'elasticnet':
            self.model = ElasticNet(
                alpha=self.kwargs.get('alpha', 1.0),
                l1_ratio=self.kwargs.get('l1_ratio', 0.5),
                random_state=self.kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'svr':
            self.model = SVR(
                kernel=self.kwargs.get('kernel', 'rbf'),
                C=self.kwargs.get('C', 1.0),
                epsilon=self.kwargs.get('epsilon', 0.1)
            )
        
        elif self.model_type == 'knn':
            self.model = KNeighborsRegressor(
                n_neighbors=self.kwargs.get('n_neighbors', 5),
                weights=self.kwargs.get('weights', 'uniform'),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.model_type == 'extra_trees':
            self.model = ExtraTreesRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', None),
                random_state=self.kwargs.get('random_state', 42),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.model_type == 'adaboost':
            self.model = AdaBoostRegressor(
                n_estimators=self.kwargs.get('n_estimators', 50),
                learning_rate=self.kwargs.get('learning_rate', 1.0),
                random_state=self.kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'gaussian_process':
            kernel = RBF() + WhiteKernel()
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                random_state=self.kwargs.get('random_state', 42),
                n_restarts_optimizer=self.kwargs.get('n_restarts_optimizer', 10)
            )
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            feature_names: 特征名称列表
        
        Returns:
            训练历史
        """
        self.feature_names = feature_names
        
        # 对于支持验证集的模型
        if self.model_type in ['xgboost', 'lightgbm', 'catboost'] and X_val is not None:
            if self.model_type == 'xgboost' and HAS_XGBOOST:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
            elif self.model_type == 'lightgbm' and HAS_LIGHTGBM:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    callbacks=[lgb.log_evaluation(0)]
                )
            elif self.model_type == 'catboost' and HAS_CATBOOST:
                self.model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    verbose=False
                )
        else:
            # 标准sklearn接口
            self.model.fit(X_train, y_train)
        
        # 记录训练和验证分数
        train_score = self.model.score(X_train, y_train)
        self.history['train_scores'].append(train_score)
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.history['val_scores'].append(val_score)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        return None
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'history': self.history
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        import joblib
        data = joblib.load(path)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.history = data['history']
        return instance


def get_available_models() -> Dict[str, bool]:
    """获取可用的模型列表"""
    return {
        'random_forest': True,
        'gradient_boosting': True,
        'ridge': True,
        'lasso': True,
        'elasticnet': True,
        'svr': True,
        'knn': True,
        'extra_trees': True,
        'adaboost': True,
        'gaussian_process': True,
        'xgboost': HAS_XGBOOST,
        'lightgbm': HAS_LIGHTGBM,
        'catboost': HAS_CATBOOST
    }


def get_model_display_names() -> Dict[str, str]:
    """获取模型的显示名称"""
    return {
        'random_forest': '🌲 随机森林 (Random Forest)',
        'gradient_boosting': '📈 梯度提升 (Gradient Boosting)',
        'xgboost': '🚀 XGBoost',
        'lightgbm': '⚡ LightGBM',
        'catboost': '🐱 CatBoost',
        'ridge': '📏 岭回归 (Ridge)',
        'lasso': '🎯 Lasso回归',
        'elasticnet': '🔗 ElasticNet',
        'svr': '🎓 支持向量回归 (SVR)',
        'knn': '👥 K近邻 (KNN)',
        'extra_trees': '🌳 极限树 (Extra Trees)',
        'adaboost': '💪 AdaBoost',
        'gaussian_process': '📊 高斯过程 (GP)'
    }


def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
    """获取模型的默认超参数"""
    defaults = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 1.0
        },
        'xgboost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'lightgbm': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'catboost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        },
        'ridge': {
            'alpha': 1.0
        },
        'lasso': {
            'alpha': 1.0
        },
        'elasticnet': {
            'alpha': 1.0,
            'l1_ratio': 0.5
        },
        'svr': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        },
        'knn': {
            'n_neighbors': 5,
            'weights': 'uniform'
        },
        'extra_trees': {
            'n_estimators': 100,
            'max_depth': None
        },
        'adaboost': {
            'n_estimators': 50,
            'learning_rate': 1.0
        },
        'gaussian_process': {
            'n_restarts_optimizer': 10
        }
    }
    
    return defaults.get(model_type, {})

