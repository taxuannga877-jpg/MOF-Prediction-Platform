"""
模型比较和性能对比可视化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> go.Figure:
    """
    绘制多个模型的性能对比图
    
    Args:
        results: {model_name: {metric_name: value, ...}, ...}
        metrics: 要显示的指标列表
    
    Returns:
        Plotly图表对象
    """
    if metrics is None:
        # 从第一个模型获取所有指标
        first_model = list(results.values())[0]
        metrics = list(first_model.keys())
    
    model_names = list(results.keys())
    n_metrics = len(metrics)
    
    # 创建子图
    fig = make_subplots(
        rows=1, cols=n_metrics,
        subplot_titles=[m.upper() for m in metrics],
        specs=[[{"type": "bar"}] * n_metrics]
    )
    
    # 为每个指标创建条形图
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in model_names]
        
        # 确定颜色（R²越高越好，MAE/RMSE越低越好）
        if metric.lower() in ['r2', 'r²', 'r_squared']:
            colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in values]
        else:
            colors = ['green' if v <= np.median(values) else 'orange' for v in values]
        
        fig.add_trace(
            go.Bar(
                name=metric,
                x=model_names,
                y=values,
                marker=dict(color=colors),
                text=[f"{v:.4f}" for v in values],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title_text="🏆 模型性能对比",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def plot_cv_results_comparison(
    cv_results: Dict[str, pd.DataFrame]
) -> go.Figure:
    """
    绘制交叉验证结果对比
    
    Args:
        cv_results: {model_name: cv_metrics_dataframe, ...}
    
    Returns:
        Plotly图表对象
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['MAE', 'RMSE', 'R²', 'MAPE'],
        specs=[[{"type": "box"}, {"type": "box"}],
               [{"type": "box"}, {"type": "box"}]]
    )
    
    metrics = ['mae', 'rmse', 'r2', 'mape']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, (row, col) in zip(metrics, positions):
        for model_name, df in cv_results.items():
            if metric in df.columns:
                fig.add_trace(
                    go.Box(
                        y=df[metric],
                        name=model_name,
                        boxmean='sd',
                        showlegend=(row == 1 and col == 1)
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title_text="📊 K折交叉验证结果对比",
        height=600,
        showlegend=True
    )
    
    return fig


def plot_training_time_comparison(
    training_times: Dict[str, float]
) -> go.Figure:
    """
    绘制训练时间对比
    
    Args:
        training_times: {model_name: training_time_seconds, ...}
    
    Returns:
        Plotly图表对象
    """
    model_names = list(training_times.keys())
    times = list(training_times.values())
    
    # 转换为更友好的单位
    if max(times) > 3600:
        times = [t / 3600 for t in times]
        unit = '小时'
    elif max(times) > 60:
        times = [t / 60 for t in times]
        unit = '分钟'
    else:
        unit = '秒'
    
    # 创建图表
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=times,
            text=[f"{t:.2f}{unit}" for t in times],
            textposition='outside',
            marker=dict(
                color=times,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f'时间({unit})')
            )
        )
    ])
    
    fig.update_layout(
        title='⏱️ 模型训练时间对比',
        xaxis_title='模型',
        yaxis_title=f'训练时间 ({unit})',
        height=400
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def plot_prediction_scatter_comparison(
    predictions: Dict[str, Dict[str, np.ndarray]],
    max_models: int = 4
) -> go.Figure:
    """
    绘制多个模型的预测散点图对比
    
    Args:
        predictions: {model_name: {'y_true': [...], 'y_pred': [...]}, ...}
        max_models: 最多显示的模型数
    
    Returns:
        Plotly图表对象
    """
    model_names = list(predictions.keys())[:max_models]
    n_models = len(model_names)
    
    # 计算子图布局
    n_cols = min(2, n_models)
    n_rows = (n_models + 1) // 2
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=model_names,
        specs=[[{"type": "scatter"}] * n_cols] * n_rows
    )
    
    for idx, model_name in enumerate(model_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        pred_data = predictions[model_name]
        y_true = pred_data['y_true']
        y_pred = pred_data['y_pred']
        
        # 计算R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # 预测值散点
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name=model_name,
                marker=dict(size=6, opacity=0.6),
                text=[f"真实: {t:.3f}<br>预测: {p:.3f}" for t, p in zip(y_true, y_pred)],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # 理想线 y=x
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='理想线',
                line=dict(color='red', dash='dash', width=2),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # 添加R²标注
        fig.add_annotation(
            x=0.05, y=0.95,
            xref=f'x{idx+1} domain' if idx > 0 else 'x domain',
            yref=f'y{idx+1} domain' if idx > 0 else 'y domain',
            text=f'R² = {r2:.4f}',
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="🎯 模型预测对比",
        height=300 * n_rows,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="真实值")
    fig.update_yaxes(title_text="预测值")
    
    return fig


def create_model_ranking_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    创建模型排名表
    
    Args:
        results: {model_name: {metric_name: value, ...}, ...}
        metrics: 要排名的指标
    
    Returns:
        DataFrame排名表
    """
    if metrics is None:
        first_model = list(results.values())[0]
        metrics = list(first_model.keys())
    
    # 创建DataFrame
    df = pd.DataFrame(results).T
    
    # 为每个指标计算排名
    ranking = pd.DataFrame(index=df.index)
    
    for metric in metrics:
        if metric in df.columns:
            # R²越大越好，其他指标越小越好
            if metric.lower() in ['r2', 'r²', 'r_squared']:
                ranking[f'{metric}_rank'] = df[metric].rank(ascending=False)
            else:
                ranking[f'{metric}_rank'] = df[metric].rank(ascending=True)
    
    # 计算平均排名
    rank_columns = [col for col in ranking.columns if col.endswith('_rank')]
    ranking['avg_rank'] = ranking[rank_columns].mean(axis=1)
    ranking['final_rank'] = ranking['avg_rank'].rank()
    
    # 合并原始数据和排名
    result_df = pd.concat([df, ranking], axis=1)
    result_df = result_df.sort_values('avg_rank')
    
    return result_df


def plot_ensemble_weights(
    ensemble_model,
    model_names: List[str] = None
) -> go.Figure:
    """
    绘制集成模型的权重分布
    
    Args:
        ensemble_model: 集成模型实例
        model_names: 基模型名称列表
    
    Returns:
        Plotly图表对象
    """
    # 尝试获取模型权重
    weights = None
    
    if hasattr(ensemble_model, 'weights_'):
        weights = ensemble_model.weights_
    elif hasattr(ensemble_model, 'estimators_'):
        # 对于VotingRegressor，默认权重相等
        n_estimators = len(ensemble_model.estimators_)
        weights = np.ones(n_estimators) / n_estimators
    
    if weights is None:
        return None
    
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(weights))]
    
    # 创建饼图
    fig = go.Figure(data=[
        go.Pie(
            labels=model_names,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            marker=dict(
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
            )
        )
    ])
    
    fig.update_layout(
        title='🎯 集成模型权重分布',
        height=400
    )
    
    return fig


def plot_feature_importance_comparison(
    feature_importances: Dict[str, np.ndarray],
    feature_names: List[str],
    top_k: int = 20
) -> go.Figure:
    """
    绘制多个模型的特征重要性对比
    
    Args:
        feature_importances: {model_name: importance_array, ...}
        feature_names: 特征名称列表
        top_k: 显示前K个重要特征
    
    Returns:
        Plotly图表对象
    """
    # 计算平均重要性
    all_importances = np.array(list(feature_importances.values()))
    mean_importance = all_importances.mean(axis=0)
    
    # 获取top-k特征
    top_indices = np.argsort(mean_importance)[-top_k:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    # 创建图表
    fig = go.Figure()
    
    for model_name, importances in feature_importances.items():
        top_importances = importances[top_indices]
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=top_features,
            y=top_importances,
            text=[f"{v:.4f}" for v in top_importances],
            textposition='outside'
        ))
    
    fig.update_layout(
        title=f'🔍 特征重要性对比 (Top {top_k})',
        xaxis_title='特征',
        yaxis_title='重要性',
        barmode='group',
        height=500,
        xaxis_tickangle=45
    )
    
    return fig

