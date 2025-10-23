"""
贝叶斯超参数优化
使用 Optuna 进行智能超参数搜索
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
warnings.filterwarnings('ignore')

# 尝试导入Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class HyperparameterOptimizer:
    """贝叶斯超参数优化器（基于Optuna）"""
    
    def __init__(
        self,
        model_type: str,
        n_trials: int = 50,
        cv_folds: int = 3,
        random_state: int = 42,
        direction: str = 'maximize'
    ):
        """
        初始化优化器
        
        Args:
            model_type: 模型类型
            n_trials: 优化试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子
            direction: 优化方向 ('maximize' 或 'minimize')
        """
        if not HAS_OPTUNA:
            raise ImportError("请先安装 Optuna: pip install optuna")
        
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.direction = direction
        
        self.study = None
        self.best_params = None
        self.optimization_history = []
    
    def _get_search_space(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """
        定义各模型的超参数搜索空间
        
        Args:
            trial: Optuna试验对象
        
        Returns:
            超参数字典
        """
        if self.model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
        
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': self.random_state
            }
        
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state
            }
        
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'verbose': -1
            }
        
        elif self.model_type == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'verbose': False
            }
        
        elif self.model_type == 'ridge':
            return {
                'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
                'random_state': self.random_state
            }
        
        elif self.model_type == 'lasso':
            return {
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'random_state': self.random_state
            }
        
        elif self.model_type == 'elasticnet':
            return {
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'random_state': self.random_state
            }
        
        elif self.model_type == 'svr':
            return {
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
        
        elif self.model_type == 'knn':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
        
        else:
            return {}
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_creator: Callable,
        metric: str = 'r2',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            X: 特征数据
            y: 标签数据
            model_creator: 模型创建函数
            metric: 评估指标 ('r2', 'mae', 'rmse')
            verbose: 是否显示详细信息
        
        Returns:
            优化结果
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
        
        # 定义目标函数
        def objective(trial):
            # 获取超参数
            params = self._get_search_space(trial)
            
            # 创建模型
            model = model_creator(**params)
            
            # 如果是自定义模型接口，需要转换
            if hasattr(model, 'model'):
                model = model.model
            
            # 交叉验证评估
            if metric == 'r2':
                scores = cross_val_score(
                    model, X, y,
                    cv=self.cv_folds,
                    scoring='r2',
                    n_jobs=-1
                )
            elif metric == 'mae':
                scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                scores = cross_val_score(
                    model, X, y,
                    cv=self.cv_folds,
                    scoring=scorer,
                    n_jobs=-1
                )
            elif metric == 'rmse':
                scorer = make_scorer(
                    lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
                    greater_is_better=False
                )
                scores = cross_val_score(
                    model, X, y,
                    cv=self.cv_folds,
                    scoring=scorer,
                    n_jobs=-1
                )
            
            mean_score = scores.mean()
            
            # 记录历史
            self.optimization_history.append({
                'trial': trial.number,
                'params': params,
                'score': mean_score
            })
            
            return mean_score
        
        # 创建研究
        if verbose:
            print(f"🚀 开始贝叶斯超参数优化")
            print(f"   模型类型: {self.model_type}")
            print(f"   试验次数: {self.n_trials}")
            print(f"   交叉验证折数: {self.cv_folds}")
            print(f"   优化指标: {metric}")
            print(f"{'='*60}\n")
        
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        # 执行优化
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ 优化完成！")
            print(f"   最佳分数: {self.study.best_value:.4f}")
            print(f"   最佳参数:")
            for param, value in self.best_params.items():
                print(f"      {param}: {value}")
            print(f"{'='*60}\n")
        
        return {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'optimization_history': self.optimization_history
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """获取优化历史DataFrame"""
        return pd.DataFrame(self.optimization_history)
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        if not HAS_OPTUNA or self.study is None:
            return None
        
        import plotly.graph_objects as go
        
        # 获取历史数据
        trials = self.study.trials
        trial_numbers = [t.number for t in trials]
        scores = [t.value for t in trials]
        
        # 计算累积最优值
        if self.direction == 'maximize':
            best_scores = [max(scores[:i+1]) for i in range(len(scores))]
        else:
            best_scores = [min(scores[:i+1]) for i in range(len(scores))]
        
        # 创建图表
        fig = go.Figure()
        
        # 所有试验的分数
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=scores,
            mode='markers',
            name='试验分数',
            marker=dict(size=8, color='lightblue', opacity=0.6)
        ))
        
        # 累积最优分数
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=best_scores,
            mode='lines+markers',
            name='最优分数',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='超参数优化历史',
            xaxis_title='试验编号',
            yaxis_title='评分',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def plot_param_importances(self):
        """绘制参数重要性"""
        if not HAS_OPTUNA or self.study is None:
            return None
        
        import plotly.graph_objects as go
        
        # 计算参数重要性
        importance = optuna.importance.get_param_importances(self.study)
        
        params = list(importance.keys())
        values = list(importance.values())
        
        # 创建图表
        fig = go.Figure(go.Bar(
            x=values,
            y=params,
            orientation='h',
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='重要性')
            )
        ))
        
        fig.update_layout(
            title='超参数重要性',
            xaxis_title='重要性分数',
            yaxis_title='参数',
            height=max(300, len(params) * 30)
        )
        
        return fig


def quick_optimize(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 3,
    metric: str = 'r2',
    verbose: bool = True
) -> Tuple[Dict[str, Any], Any]:
    """
    快速超参数优化
    
    Args:
        model_type: 模型类型
        X: 特征
        y: 标签
        n_trials: 试验次数
        cv_folds: 交叉验证折数
        metric: 评估指标
        verbose: 是否显示详细信息
    
    Returns:
        (最佳参数, 优化器实例)
    """
    from .traditional_ml import TraditionalMLModel
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(
        model_type=model_type,
        n_trials=n_trials,
        cv_folds=cv_folds,
        direction='maximize' if metric == 'r2' else 'minimize'
    )
    
    # 执行优化
    def model_creator(**params):
        return TraditionalMLModel(model_type=model_type, **params)
    
    results = optimizer.optimize(X, y, model_creator, metric, verbose)
    
    return results['best_params'], optimizer
