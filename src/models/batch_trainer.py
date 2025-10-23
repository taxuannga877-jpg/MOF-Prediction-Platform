"""
批量训练器 - 用于一次性训练和对比所有可用的机器学习模型
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .traditional_ml import TraditionalMLModel, get_available_models, get_model_display_names


class BatchTrainer:
    """批量训练器类"""
    
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_cv: bool = True,
        n_folds: int = 5,
        train_ratio: float = 0.8,
        use_hpo: bool = False,
        n_trials: int = 20,
        progress_callback=None
    ) -> Dict:
        """
        训练所有可用的模型
        
        Parameters:
        -----------
        X : DataFrame
            特征数据
        y : Series
            目标变量
        use_cv : bool
            是否使用K折交叉验证
        n_folds : int
            交叉验证折数
        train_ratio : float
            训练集比例（如果不使用CV）
        use_hpo : bool
            是否使用超参数优化
        n_trials : int
            超参数优化试验次数
        progress_callback : callable
            进度回调函数
        
        Returns:
        --------
        dict : 所有模型的训练结果
        """
        # 获取所有可用模型
        available_models_dict = get_available_models()
        model_names = [k for k, v in available_models_dict.items() if v]
        display_names_dict = get_model_display_names()
        
        if len(model_names) == 0:
            raise ValueError("没有可用的模型！请安装相关库。")
        
        # 训练每个模型
        for idx, model_type in enumerate(model_names):
            model_display_name = display_names_dict[model_type]
            
            if progress_callback:
                progress_callback(idx, len(model_names), model_display_name)
            
            try:
                start_time = time.time()
                model = TraditionalMLModel(model_type=model_type)
                
                # 如果启用K折交叉验证
                if use_cv and n_folds:
                    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                    scores = cross_val_score(model.model, X, y, cv=kf, scoring='r2')
                    cv_score = scores.mean()
                    cv_std = scores.std()
                    training_method = f"{n_folds}折交叉验证"
                else:
                    # 简单训练测试集分割
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=1-train_ratio, random_state=42
                    )
                    model.train(X_train, y_train, X_test, y_test, feature_names=list(X.columns))
                    y_pred = model.predict(X_test)
                    cv_score = r2_score(y_test, y_pred)
                    cv_std = 0
                    training_method = "训练集/测试集分割"
                
                # 在全部数据上训练最终模型
                model.model.fit(X, y)
                y_pred_full = model.predict(X)
                
                training_time = time.time() - start_time
                
                # 计算指标
                mae = mean_absolute_error(y, y_pred_full)
                rmse = np.sqrt(mean_squared_error(y, y_pred_full))
                r2 = r2_score(y, y_pred_full)
                
                # 获取特征重要性
                feature_importance = model.get_feature_importance()
                
                # 保存结果
                self.results[model_type] = {
                    'display_name': model_display_name,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_score': cv_score,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'model': model,
                    'training_method': training_method,
                    'feature_importance': feature_importance
                }
                
            except Exception as e:
                print(f"警告: {model_display_name} 训练失败: {str(e)}")
        
        # 找到最佳模型
        if self.results:
            best_model_type = max(self.results.items(), key=lambda x: x[1]['r2'])[0]
            self.best_model = self.results[best_model_type]['model']
            self.best_model_name = self.results[best_model_type]['display_name']
        
        return self.results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """获取结果DataFrame"""
        results_data = []
        for model_type, result in self.results.items():
            results_data.append({
                '模型': result['display_name'],
                'R²': f"{result['r2']:.4f}",
                'MAE': f"{result['mae']:.4f}",
                'RMSE': f"{result['rmse']:.4f}",
                'CV Score': f"{result['cv_score']:.4f} ± {result['cv_std']:.4f}",
                '训练时间(s)': f"{result['training_time']:.2f}",
                '训练方法': result['training_method']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('R²', ascending=False)
        return results_df
    
    def get_best_model(self) -> Tuple:
        """获取最佳模型"""
        return self.best_model, self.best_model_name

