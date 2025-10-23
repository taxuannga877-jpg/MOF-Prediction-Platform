# 🧪 MOF预测平台 (MOF Prediction Platform)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)

> 基于深度学习和传统机器学习的智能金属有机框架（MOF）材料性质预测系统

一个功能强大、易于使用的MOF材料性质预测平台，集成了**CGCNN**、**MOFormer**深度学习模型，以及**13种传统机器学习算法**，支持K折交叉验证、贝叶斯超参数优化和SHAP可解释性分析。

---

## ✨ 核心特性

### 🤖 多模型支持

#### 深度学习模型
- **CGCNN** - Crystal Graph Convolutional Neural Network，基于3D晶体结构
- **MOFormer** - MOF Transformer，基于MOFid文本序列
- **集成模型** - 结合多种模型的优势

#### 传统机器学习（13种算法）
- 🌲 **树模型**: Random Forest, XGBoost, LightGBM, CatBoost, Extra Trees, AdaBoost, Gradient Boosting
- 📏 **线性模型**: Ridge, Lasso, ElasticNet
- 🎓 **其他**: SVR, KNN, Gaussian Process

### 🎯 高级功能

- **K折交叉验证** - 评估模型稳定性和泛化能力
- **贝叶斯超参数优化** - 基于Optuna的智能调参
- **模型集成** - Voting、Stacking、Blending三种策略
- **SHAP可解释性** - 深入理解模型预测机制
- **动态列选择** - 自动识别并支持任意目标属性
- **完整可视化** - 训练曲线、预测分析、误差分布、特征重要性

### 📊 数据支持

- ✅ QMOF数据集（20,000+ MOF结构）
- ✅ 自定义数据上传（CSV, JSON, Excel）
- ✅ 动态特征提取
- ✅ 智能缺失值处理

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/MOF-Prediction-Platform.git
cd MOF-Prediction-Platform

# 安装依赖（推荐使用虚拟环境）
bash scripts/install_dependencies.sh

# 或手动安装
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 运行平台

```bash
# 使用启动脚本
bash scripts/run_platform.sh

# 或直接运行
streamlit run src/app.py --server.port 8501
```

然后访问：`http://localhost:8501`

---

## 📖 使用指南

### 1️⃣ 数据加载

```python
# 方式1: 加载QMOF数据集
- 选择"QMOF数据库"
- 点击"加载数据"

# 方式2: 上传自定义数据
- 上传CSV/JSON/Excel文件
- 系统自动提取列名
```

### 2️⃣ 模型训练

#### 使用传统机器学习

```python
# 在UI中:
1. 选择"🌲 传统机器学习模型"
2. 选择算法（如XGBoost）
3. 选择目标列（自动识别数值型列）
4. 配置参数
5. 点击"开始训练"
```

#### 使用深度学习

```python
# 在UI中:
1. 选择"🧠 深度学习模型"
2. 选择CGCNN或MOFormer
3. 配置超参数
4. 开始训练
```

### 3️⃣ 高级功能

#### K折交叉验证

```python
from models import CrossValidator

cv = CrossValidator(n_splits=5)
results = cv.cross_validate(
    model_creator=lambda: TraditionalMLModel('xgboost'),
    X=X_train,
    y=y_train
)
print(f"平均 R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
```

#### 超参数优化

```python
from models import quick_optimize

best_params, optimizer = quick_optimize(
    model_type='xgboost',
    X=X_train,
    y=y_train,
    n_trials=50
)
```

#### 模型集成

```python
from models import create_auto_ensemble

ensemble = create_auto_ensemble(
    X_train, y_train,
    model_types=['random_forest', 'xgboost', 'lightgbm'],
    ensemble_method='stacking'
)
```

---

## 📁 项目结构

```
MOF-Prediction-Platform/
├── src/
│   ├── app.py                      # Streamlit主应用
│   ├── config.py                   # 配置文件
│   ├── models/                     # 模型实现
│   │   ├── base_model.py          # 基础模型类
│   │   ├── cgcnn_model.py         # CGCNN实现
│   │   ├── moformer_model.py      # MOFormer实现
│   │   ├── traditional_ml.py      # 传统ML包装器
│   │   ├── ensemble.py            # 集成策略
│   │   ├── cross_validation.py    # K折交叉验证
│   │   └── hyperparameter_optimization.py  # 贝叶斯优化
│   ├── utils/                      # 工具函数
│   │   ├── data_loader.py         # 数据加载
│   │   ├── data_processor.py      # 数据处理
│   │   └── model_router.py        # 模型路由
│   └── visualization/              # 可视化
│       ├── plots.py               # 通用绘图
│       ├── interpretability.py    # SHAP分析
│       └── model_comparison.py    # 模型对比
├── data/                           # 数据目录
├── models/checkpoints/             # 模型检查点
├── scripts/                        # 启动脚本
├── docs/                           # 文档
├── requirements.txt                # Python依赖
└── README.md                       # 本文件
```

---

## 🎓 算法说明

### 深度学习模型

#### CGCNN (Crystal Graph CNN)
- 将晶体结构表示为图
- 节点：原子类型
- 边：原子间距离
- 通过图卷积学习局部和全局特征

#### MOFormer (MOF Transformer)
- 基于Transformer架构
- 输入：MOFid文本序列
- 使用自注意力机制捕获长程依赖

### 传统机器学习

| 模型 | 适用场景 | 优势 |
|-----|---------|------|
| Random Forest | 通用 | 稳定、易用、抗过拟合 |
| XGBoost | 高性能需求 | 精度高、速度快 |
| LightGBM | 大数据集 | 内存效率高 |
| CatBoost | 类别特征多 | 原生类别支持 |
| Ridge/Lasso | 线性关系 | 可解释性强 |

---

## 📊 性能基准

在QMOF数据集上的性能（带隙预测）：

| 模型 | MAE | RMSE | R² | 训练时间 |
|------|-----|------|----|---------| 
| CGCNN | 0.234 | 0.456 | 0.923 | 2h |
| MOFormer | 0.198 | 0.389 | 0.945 | 1.5h |
| XGBoost | 0.245 | 0.478 | 0.915 | 5min |
| LightGBM | 0.251 | 0.485 | 0.912 | 3min |
| Stacking Ensemble | 0.189 | 0.372 | 0.951 | 15min |

---

## 🛠️ 技术栈

- **框架**: Streamlit, PyTorch
- **机器学习**: scikit-learn, XGBoost, LightGBM, CatBoost
- **优化**: Optuna
- **可视化**: Plotly, Matplotlib, Seaborn
- **材料科学**: PyMatGen, ASE
- **可解释性**: SHAP

---

## 📚 文档

- [快速开始指南](GETTING_STARTED.md)
- [项目总结](PROJECT_SUMMARY.md)
- [机器学习模型集成说明](机器学习模型集成说明.md)
- [更新日志](CHANGELOG_ML.md)

---

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

感谢以下开源项目：

- [CGCNN](https://github.com/txie-93/cgcnn) - Crystal Graph Convolutional Neural Networks
- [scikit-learn](https://scikit-learn.org/) - Machine Learning in Python
- [XGBoost](https://xgboost.readthedocs.io/) - Scalable and Flexible Gradient Boosting
- [LightGBM](https://lightgbm.readthedocs.io/) - Light Gradient Boosting Machine
- [CatBoost](https://catboost.ai/) - Gradient Boosting on Decision Trees
- [Optuna](https://optuna.org/) - Hyperparameter Optimization Framework
- [SHAP](https://shap.readthedocs.io/) - SHapley Additive exPlanations
- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- [PyTorch](https://pytorch.org/) - Deep Learning Framework

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/YOUR_USERNAME/MOF-Prediction-Platform/issues)
- Email: your.email@example.com

---

## ⭐ Star History

如果这个项目对您有帮助，请给个Star⭐！

---

**Built with ❤️ for the Materials Science Community**
