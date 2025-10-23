"""
可视化模块
Visualization Module
"""

from .plots import (
    plot_training_history,
    plot_predictions,
    plot_feature_importance,
    plot_error_distribution,
    plot_learning_curve,
    plot_kfold_results
)

from .interpretability import (
    explain_with_shap,
    plot_shap_summary,
    plot_shap_waterfall,
    plot_shap_force,
    plot_shap_dependence,
    calculate_feature_importance_from_shap,
    plot_shap_feature_importance
)

from .model_comparison import (
    plot_model_comparison,
    plot_cv_results_comparison,
    plot_training_time_comparison,
    plot_prediction_scatter_comparison,
    create_model_ranking_table,
    plot_ensemble_weights,
    plot_feature_importance_comparison
)

__all__ = [
    'plot_training_history',
    'plot_predictions',
    'plot_feature_importance',
    'plot_error_distribution',
    'plot_learning_curve',
    'plot_kfold_results',
    'explain_with_shap',
    'plot_shap_summary',
    'plot_shap_waterfall',
    'plot_shap_force',
    'plot_shap_dependence',
    'calculate_feature_importance_from_shap',
    'plot_shap_feature_importance',
    'plot_model_comparison',
    'plot_cv_results_comparison',
    'plot_training_time_comparison',
    'plot_prediction_scatter_comparison',
    'create_model_ranking_table',
    'plot_ensemble_weights',
    'plot_feature_importance_comparison'
]
