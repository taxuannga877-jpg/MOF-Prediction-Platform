"""
模型可解释性分析模块
Model Interpretability Module
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, List, Optional, Dict
import matplotlib.pyplot as plt
import io
import base64


def explain_with_shap(model: Any, X: pd.DataFrame, feature_names: List[str] = None,
                     sample_size: int = 100, save_path: Optional[str] = None):
    """
    使用SHAP进行模型解释
    
    Args:
        model: 训练好的模型
        X: 特征数据
        feature_names: 特征名称列表
        sample_size: 用于计算的样本数
        save_path: 保存路径
    
    Returns:
        SHAP值和可视化图像
    """
    if feature_names is None:
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'Feature {i}' for i in range(X.shape[1])]
    
    # 限制样本数以加快计算
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    try:
        # 创建SHAP explainer
        if hasattr(model, 'predict'):
            # 对于sklearn类型的模型
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = explainer(X_sample)
        else:
            # 对于其他类型的模型，使用KernelExplainer
            explainer = shap.KernelExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_names,
            'X_sample': X_sample
        }
    
    except Exception as e:
        print(f"SHAP计算失败: {e}")
        return None


def plot_shap_summary(shap_values, X_sample, feature_names, save_path: Optional[str] = None):
    """
    绘制SHAP summary图
    
    Args:
        shap_values: SHAP值
        X_sample: 样本数据
        feature_names: 特征名称
        save_path: 保存路径
    
    Returns:
        图像的base64编码
    """
    plt.figure(figsize=(12, 8))
    
    if hasattr(shap_values, 'values'):
        # 新版SHAP
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    else:
        # 旧版SHAP
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    
    # 转换为base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return img_base64


def plot_shap_waterfall(shap_values, X_sample, idx=0, save_path: Optional[str] = None):
    """
    绘制单个样本的SHAP waterfall图
    
    Args:
        shap_values: SHAP值
        X_sample: 样本数据
        idx: 样本索引
        save_path: 保存路径
    
    Returns:
        图像的base64编码
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(shap_values, 'values'):
        shap.waterfall_plot(shap_values[idx], show=False)
    else:
        shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                             base_values=shap_values.base_values[idx],
                                             data=X_sample.iloc[idx]), show=False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return img_base64


def plot_shap_force(shap_values, X_sample, idx=0, save_path: Optional[str] = None):
    """
    绘制SHAP force图
    
    Args:
        shap_values: SHAP值
        X_sample: 样本数据
        idx: 样本索引
        save_path: 保存路径
    
    Returns:
        图像HTML
    """
    try:
        if hasattr(shap_values, 'values'):
            force_plot = shap.force_plot(
                shap_values.base_values[idx],
                shap_values.values[idx],
                X_sample.iloc[idx],
                matplotlib=False
            )
        else:
            force_plot = shap.force_plot(
                shap_values.base_values[idx],
                shap_values[idx],
                X_sample.iloc[idx],
                matplotlib=False
            )
        
        if save_path:
            shap.save_html(save_path, force_plot)
        
        return force_plot
    except Exception as e:
        print(f"Force图生成失败: {e}")
        return None


def plot_shap_dependence(shap_values, X_sample, feature_idx, feature_names, 
                        save_path: Optional[str] = None):
    """
    绘制SHAP dependence图
    
    Args:
        shap_values: SHAP值
        X_sample: 样本数据
        feature_idx: 特征索引
        feature_names: 特征名称列表
        save_path: 保存路径
    
    Returns:
        图像的base64编码
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(shap_values, 'values'):
        shap.dependence_plot(feature_idx, shap_values.values, X_sample, 
                           feature_names=feature_names, show=False)
    else:
        shap.dependence_plot(feature_idx, shap_values, X_sample,
                           feature_names=feature_names, show=False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return img_base64


def calculate_feature_importance_from_shap(shap_values, feature_names):
    """
    从SHAP值计算特征重要性
    
    Args:
        shap_values: SHAP值
        feature_names: 特征名称列表
    
    Returns:
        特征重要性DataFrame
    """
    if hasattr(shap_values, 'values'):
        values = shap_values.values
    else:
        values = shap_values
    
    # 计算平均绝对SHAP值
    importance = np.abs(values).mean(axis=0)
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_shap_feature_importance(importance_df, top_k=20, save_path: Optional[str] = None):
    """
    绘制基于SHAP的特征重要性
    
    Args:
        importance_df: 特征重要性DataFrame
        top_k: 显示前k个特征
        save_path: 保存路径
    
    Returns:
        Plotly figure
    """
    top_features = importance_df.head(top_k)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_features['importance'].values,
            y=top_features['feature'].values,
            orientation='h',
            marker=dict(
                color=top_features['importance'].values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="SHAP重要性")
            )
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_k} 特征重要性 (基于SHAP)',
        xaxis_title='平均|SHAP值|',
        yaxis_title='特征名称',
        height=max(400, top_k * 25),
        yaxis=dict(autorange="reversed")
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def create_shap_analysis(model, X_test, feature_names=None, max_display=20, sample_size=100):
    """
    创建SHAP分析可视化（专门为Streamlit优化）
    
    Args:
        model: 训练好的模型（可以是sklearn模型对象）
        X_test: 测试数据（numpy数组或DataFrame）
        feature_names: 特征名称列表
        max_display: 最多显示的特征数
        sample_size: 用于计算的样本数
    
    Returns:
        matplotlib figure 或 None
    """
    try:
        # 转换为DataFrame
        if not isinstance(X_test, pd.DataFrame):
            if feature_names is not None:
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
            else:
                X_test_df = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(X_test.shape[1])])
        else:
            X_test_df = X_test
        
        if feature_names is None:
            feature_names = X_test_df.columns.tolist()
        
        # 限制样本数以加快计算
        if len(X_test_df) > sample_size:
            X_sample = X_test_df.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_test_df
        
        # 根据模型类型选择合适的SHAP Explainer
        from sklearn.ensemble import (
            RandomForestRegressor, GradientBoostingRegressor,
            ExtraTreesRegressor, AdaBoostRegressor
        )
        try:
            from xgboost import XGBRegressor
            has_xgboost = True
        except ImportError:
            has_xgboost = False
        
        try:
            from lightgbm import LGBMRegressor
            has_lightgbm = True
        except ImportError:
            has_lightgbm = False
        
        try:
            from catboost import CatBoostRegressor
            has_catboost = True
        except ImportError:
            has_catboost = False
        
        # TreeExplainer支持的树模型（不包括AdaBoost，因为SHAP不直接支持）
        tree_explainer_models = [
            RandomForestRegressor, 
            GradientBoostingRegressor,
            ExtraTreesRegressor
        ]
        
        if has_xgboost:
            tree_explainer_models.append(XGBRegressor)
        if has_lightgbm:
            tree_explainer_models.append(LGBMRegressor)
        if has_catboost:
            tree_explainer_models.append(CatBoostRegressor)
        
        tree_explainer_models = tuple(tree_explainer_models)
        
        # 选择合适的explainer
        if isinstance(model, tree_explainer_models):
            # 对于TreeExplainer支持的树模型，使用TreeExplainer（速度更快）
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception as e:
                # 如果TreeExplainer失败，降级到通用Explainer
                print(f"TreeExplainer失败，使用通用Explainer: {e}")
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
        elif isinstance(model, AdaBoostRegressor):
            # AdaBoost需要特殊处理，使用通用Explainer
            try:
                # 首先尝试使用通用Explainer（较快）
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
            except:
                # 如果失败，使用KernelExplainer（较慢但更通用）
                print("使用KernelExplainer处理AdaBoost...")
                background = shap.sample(X_sample, min(50, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample)
        else:
            # 对于其他模型，使用通用Explainer
            try:
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
            except:
                # 如果失败，使用KernelExplainer（较慢但通用）
                print("使用KernelExplainer...")
                background = shap.sample(X_sample, min(50, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample)
        
        # 创建summary plot
        fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.3)))
        
        # 根据shap_values的类型选择绘图方式
        if hasattr(shap_values, 'values'):
            # 新版SHAP (Explanation对象)
            shap.summary_plot(
                shap_values.values,
                X_sample,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
        else:
            # 旧版SHAP (numpy数组)
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"❌ SHAP分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

