"""
可视化绘图模块
Visualization Plotting Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典，包含 'train_loss', 'val_loss', 'learning_rate' 等
        save_path: 保存路径（可选）
    
    Returns:
        Plotly figure 对象
    """
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('训练&验证损失', '学习率变化', '损失对比（对数）', 'R² Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. 训练和验证损失
    if 'train_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='训练损失',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
    
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='验证损失',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
    
    # 2. 学习率变化
    if 'learning_rate' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['learning_rate'], name='学习率',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
    
    # 3. 对数尺度的损失
    if 'train_loss' in history and 'val_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_loss'], name='训练损失（log）',
                      line=dict(color='blue', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='验证损失（log）',
                      line=dict(color='red', width=2, dash='dash')),
            row=2, col=1
        )
        fig.update_yaxes(type="log", row=2, col=1)
    
    # 4. R² Score
    if 'train_r2' in history and 'val_r2' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_r2'], name='训练R²',
                      line=dict(color='blue', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_r2'], name='验证R²',
                      line=dict(color='red', width=2, dash='dash')),
            row=2, col=2
        )
    
    # 更新布局
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
    fig.update_yaxes(title_text="Loss (log)", row=2, col=1)
    fig.update_yaxes(title_text="R² Score", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="训练过程可视化",
        title_x=0.5
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_predictions(y_true, y_pred, dataset_name='测试集', save_path: Optional[str] = None):
    """
    绘制预测vs真实值对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        dataset_name: 数据集名称
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 创建子图
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'预测 vs 真实值 ({dataset_name})',
            '残差分布',
            '误差直方图'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # 1. 预测 vs 真实值散点图
    fig.add_trace(
        go.Scatter(
            x=y_true, y=y_pred,
            mode='markers',
            marker=dict(size=8, color=y_true, colorscale='Viridis', showscale=True),
            name='预测点',
            text=[f'真实:{t:.3f}<br>预测:{p:.3f}' for t, p in zip(y_true, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 添加理想线（y=x）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='理想线 (y=x)'
        ),
        row=1, col=1
    )
    
    # 添加指标文本
    fig.add_annotation(
        text=f'MAE: {mae:.4f}<br>RMSE: {rmse:.4f}<br>R²: {r2:.4f}',
        xref="x", yref="y",
        x=min_val + 0.05 * (max_val - min_val),
        y=max_val - 0.1 * (max_val - min_val),
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        row=1, col=1
    )
    
    # 2. 残差图
    residuals = y_pred - y_true
    fig.add_trace(
        go.Scatter(
            x=y_true, y=residuals,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='残差'
        ),
        row=1, col=2
    )
    
    # 添加零线
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[0, 0],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='零线'
        ),
        row=1, col=2
    )
    
    # 3. 误差直方图
    errors = np.abs(residuals)
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=30,
            name='误差分布',
            marker_color='lightblue'
        ),
        row=1, col=3
    )
    
    # 更新布局
    fig.update_xaxes(title_text="真实值", row=1, col=1)
    fig.update_yaxes(title_text="预测值", row=1, col=1)
    
    fig.update_xaxes(title_text="真实值", row=1, col=2)
    fig.update_yaxes(title_text="残差", row=1, col=2)
    
    fig.update_xaxes(title_text="绝对误差", row=1, col=3)
    fig.update_yaxes(title_text="频数", row=1, col=3)
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text=f"{dataset_name} 预测结果分析",
        title_x=0.5
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, 
                            top_k: int = 20, save_path: Optional[str] = None):
    """
    绘制特征重要性图
    
    Args:
        feature_names: 特征名称列表
        importances: 特征重要性值
        top_k: 显示前k个重要特征
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    # 排序
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # 创建条形图
    fig = go.Figure(data=[
        go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="重要性")
            )
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_k} 特征重要性',
        xaxis_title='重要性',
        yaxis_title='特征名称',
        height=max(400, top_k * 25),
        yaxis=dict(autorange="reversed")
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_error_distribution(y_true, y_pred, bins=50, save_path: Optional[str] = None):
    """
    绘制误差分布图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        bins: 直方图箱数
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('误差分布（带核密度）', '绝对误差分布', 
                       '误差百分位图', '误差vs真实值'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "scatter"}]]
    )
    
    # 1. 误差直方图
    fig.add_trace(
        go.Histogram(x=errors, nbinsx=bins, name='误差', 
                    marker_color='lightblue', opacity=0.7),
        row=1, col=1
    )
    
    # 2. 绝对误差直方图
    fig.add_trace(
        go.Histogram(x=abs_errors, nbinsx=bins, name='绝对误差',
                    marker_color='lightcoral', opacity=0.7),
        row=1, col=2
    )
    
    # 3. 百分位图
    percentiles = np.percentile(abs_errors, np.arange(0, 101, 5))
    fig.add_trace(
        go.Scatter(x=np.arange(0, 101, 5), y=percentiles, 
                  mode='lines+markers', name='百分位',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    # 4. 误差 vs 真实值
    fig.add_trace(
        go.Scatter(x=y_true, y=abs_errors, mode='markers',
                  marker=dict(size=6, color=abs_errors, colorscale='Reds', showscale=True),
                  name='误差点'),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_xaxes(title_text="误差", row=1, col=1)
    fig.update_xaxes(title_text="绝对误差", row=1, col=2)
    fig.update_xaxes(title_text="百分位", row=2, col=1)
    fig.update_xaxes(title_text="真实值", row=2, col=2)
    
    fig.update_yaxes(title_text="频数", row=1, col=1)
    fig.update_yaxes(title_text="频数", row=1, col=2)
    fig.update_yaxes(title_text="绝对误差", row=2, col=1)
    fig.update_yaxes(title_text="绝对误差", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="误差分布详细分析",
        title_x=0.5
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores, save_path: Optional[str] = None):
    """
    绘制学习曲线
    
    Args:
        train_sizes: 训练集大小
        train_scores: 训练分数（shape: [n_sizes, n_folds]）
        val_scores: 验证分数（shape: [n_sizes, n_folds]）
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # 训练分数
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='训练分数',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean + train_std,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean - train_std,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0,100,255,0.2)',
        fill='tonexty',
        name='训练±std',
        hoverinfo='skip'
    ))
    
    # 验证分数
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='验证分数',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean + val_std,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean - val_std,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255,0,0,0.2)',
        fill='tonexty',
        name='验证±std',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='学习曲线',
        xaxis_title='训练集大小',
        yaxis_title='分数',
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_kfold_results(fold_results: List[Dict], save_path: Optional[str] = None):
    """
    绘制K折交叉验证结果
    
    Args:
        fold_results: 每折结果列表，每个元素包含 'fold', 'train_score', 'val_score'
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    folds = [r['fold'] for r in fold_results]
    train_scores = [r['train_score'] for r in fold_results]
    val_scores = [r['val_score'] for r in fold_results]
    
    fig = go.Figure()
    
    # 训练分数
    fig.add_trace(go.Bar(
        x=folds,
        y=train_scores,
        name='训练分数',
        marker_color='lightblue'
    ))
    
    # 验证分数
    fig.add_trace(go.Bar(
        x=folds,
        y=val_scores,
        name='验证分数',
        marker_color='lightcoral'
    ))
    
    # 添加平均线
    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    
    fig.add_hline(y=train_mean, line_dash="dash", line_color="blue",
                  annotation_text=f"训练平均: {train_mean:.4f}")
    fig.add_hline(y=val_mean, line_dash="dash", line_color="red",
                  annotation_text=f"验证平均: {val_mean:.4f}")
    
    fig.update_layout(
        title='K折交叉验证结果',
        xaxis_title='Fold',
        yaxis_title='分数',
        barmode='group',
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_predictions_scatter(y_true, y_pred, dataset_name='Test Set', save_path: Optional[str] = None):
    """
    绘制预测值 vs 真实值的散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        dataset_name: 数据集名称
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 创建散点图
    fig = go.Figure()
    
    # 添加散点
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=8,
            color=np.abs(y_true - y_pred),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='绝对误差'),
            line=dict(width=0.5, color='white')
        ),
        name='预测点',
        text=[f'真实: {t:.2f}<br>预测: {p:.2f}<br>误差: {abs(t-p):.2f}' 
              for t, p in zip(y_true, y_pred)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # 添加理想线 y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='理想预测线 (y=x)'
    ))
    
    fig.update_layout(
        title=f'{dataset_name} - 预测 vs 真实值<br><sub>MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}</sub>',
        xaxis_title='真实值',
        yaxis_title='预测值',
        height=500,
        width=600,
        hovermode='closest',
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_residuals(y_true, y_pred, save_path: Optional[str] = None):
    """
    绘制残差图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    # 残差散点图
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=8,
            color=np.abs(residuals),
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='|残差|'),
            line=dict(width=0.5, color='white')
        ),
        name='残差',
        text=[f'预测: {p:.2f}<br>残差: {r:.2f}' for p, r in zip(y_pred, residuals)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # 零线
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    
    # 添加±2σ参考线
    std = np.std(residuals)
    fig.add_hline(y=2*std, line_dash="dot", line_color="orange", 
                  annotation_text=f"+2σ ({2*std:.2f})")
    fig.add_hline(y=-2*std, line_dash="dot", line_color="orange",
                  annotation_text=f"-2σ ({-2*std:.2f})")
    
    fig.update_layout(
        title=f'残差分析<br><sub>均值: {np.mean(residuals):.4f} | 标准差: {std:.4f}</sub>',
        xaxis_title='预测值',
        yaxis_title='残差 (真实值 - 预测值)',
        height=500,
        width=600,
        hovermode='closest'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_feature_importance_bar(importance, feature_names, top_k=20, save_path: Optional[str] = None):
    """
    绘制特征重要性条形图
    
    Args:
        importance: 特征重要性数组
        feature_names: 特征名称列表
        top_k: 显示前k个特征
        save_path: 保存路径
    
    Returns:
        Plotly figure 对象
    """
    # 创建DataFrame并排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 取前k个
    top_features = importance_df.head(top_k)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_features['importance'].values,
        y=top_features['feature'].values,
        orientation='h',
        marker=dict(
            color=top_features['importance'].values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="重要性")
        ),
        text=[f'{v:.4f}' for v in top_features['importance'].values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f'Top {top_k} 特征重要性',
        xaxis_title='重要性得分',
        yaxis_title='特征名称',
        height=max(400, top_k * 25),
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig

