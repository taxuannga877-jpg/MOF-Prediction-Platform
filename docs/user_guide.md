# MOF预测平台 - 用户指南

## 目录

1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [性质预测](#性质预测)
5. [结果分析](#结果分析)
6. [高级功能](#高级功能)

## 快速开始

### 安装

```bash
# 1. 进入项目目录
cd MOF-Prediction-Platform

# 2. 安装依赖
bash scripts/install_dependencies.sh

# 3. 启动平台
bash scripts/run_platform.sh
```

### 首次使用

1. 访问 http://localhost:8501
2. 在侧边栏选择"📂 数据管理"
3. 加载QMOF数据集或上传自己的数据
4. 选择"🤖 模型训练"开始训练
5. 使用"🔮 性质预测"进行预测

## 数据准备

### 使用QMOF数据集

平台已集成QMOF数据集，支持以下数据源：

| 数据源 | 格式 | 适用模型 | 大小 |
|--------|------|---------|------|
| qmof.json | JSON | MOFormer | ~90MB |
| qmof.csv | CSV | MOFormer | ~21MB |
| relaxed_structures.zip | CIF | CGCNN | ~114MB |
| qmof_structure_data.json | JSON | CGCNN | ~3GB |

**操作步骤**:

1. 进入"📂 数据管理" → "📦 QMOF数据集"
2. 选择数据源
3. 设置加载数量限制（建议先用100条测试）
4. 点击"📥 加载QMOF数据"

### 上传自定义数据

#### CIF文件（用于CGCNN）

```
your_mof.cif  # 单个结构
structures.zip  # 批量结构
```

#### JSON文件（用于MOFormer）

```json
[
  {
    "qmof_id": "mof-001",
    "info": {
      "mofid": {
        "mofid": "[Zn].[O-]C(=O)c1ccc(cc1)C(=O)[O-].pcu"
      }
    },
    "outputs": {
      "pbe": {
        "bandgap": 3.45
      }
    }
  }
]
```

#### CSV文件（用于MOFormer）

```csv
qmof_id,mofid,bandgap
mof-001,[Zn]...,3.45
mof-002,[Cu]...,2.10
```

## 模型训练

### 模型选择

平台提供智能模型推荐：

- **自动推荐**: 根据数据格式自动选择最优模型
- **手动选择**: 根据需求自行选择

#### CGCNN
- **优势**: 高精度，物理意义明确
- **劣势**: 需要3D结构，计算较慢
- **适用**: 已知结构的精确预测

#### MOFormer
- **优势**: 速度快，不需要3D结构
- **劣势**: 精度略低于CGCNN
- **适用**: 快速筛选，大规模预测

#### 集成模型
- **优势**: 最高精度，鲁棒性强
- **劣势**: 计算时间最长
- **适用**: 重要决策，最高精度需求

### 训练参数配置

```python
# 推荐配置
epochs: 100          # CGCNN推荐100-200轮
batch_size: 32       # 根据显存调整
learning_rate: 1e-3  # CGCNN: 1e-3, MOFormer: 1e-4
train_ratio: 0.8     # 训练集80%
val_ratio: 0.1       # 验证集10%
```

### 训练流程

1. **数据准备**
   - 加载数据
   - 选择目标属性
   - 数据预处理

2. **模型配置**
   - 选择模型类型
   - 设置超参数
   - 配置训练选项

3. **开始训练**
   - 实时显示训练进度
   - 监控损失曲线
   - 自动保存最佳模型

4. **训练完成**
   - 查看训练历史
   - 评估模型性能
   - 保存模型checkpoint

## 性质预测

### 单个MOF预测

#### 使用MOFormer

```
输入: MOFid字符串
例如: [Zn].[O-]C(=O)c1ccc(cc1)C(=O)[O-].pcu
输出: 预测的能带隙值
```

#### 使用CGCNN

```
输入: CIF文件
输出: 预测的性质值
```

### 批量预测

1. 准备包含多个MOF的文件
2. 选择"批量预测"模式
3. 上传文件
4. 下载预测结果（CSV/Excel）

### 从数据集预测

1. 在已加载的数据集中选择MOF
2. 点击"预测选中"
3. 查看预测结果
4. 导出结果

## 结果分析

### 预测准确性分析

- **散点图**: 预测值 vs 真实值
- **回归线**: 评估线性关系
- **R²分数**: 评估拟合度

### 误差分析

- **残差图**: 查看预测误差分布
- **误差直方图**: 分析误差特征
- **MAE/RMSE**: 定量评估精度

### 可解释性分析

#### SHAP值分析

- **特征重要性**: 哪些特征最影响预测
- **SHAP summary plot**: 全局特征影响
- **依赖图**: 特征与预测的关系

#### 注意力权重（MOFormer）

- 可视化模型关注的MOFid部分
- 理解模型决策过程

## 高级功能

### 模型集成

结合CGCNN和MOFormer的预测：

```python
# 加权平均
ensemble_pred = 0.6 * cgcnn_pred + 0.4 * moformer_pred
```

### 超参数优化

使用网格搜索或贝叶斯优化：

```python
# 在模型训练页面启用
☑️ 启用超参数优化
优化方法: [贝叶斯优化]
优化轮数: 20
```

### 迁移学习

使用预训练模型微调：

1. 加载预训练模型
2. 冻结部分层
3. 在新数据上微调
4. 评估性能提升

### 批量导出

导出预测结果：

- CSV格式
- Excel格式
- JSON格式
- PDF报告

## 常见问题

### Q1: 内存不足怎么办？

**A**: 
- 减小batch_size
- 限制加载数据量
- 使用CPU模式（较慢）

### Q2: 训练太慢怎么办？

**A**:
- 使用GPU加速
- 减少训练轮数
- 使用MOFormer代替CGCNN
- 减小模型层数

### Q3: 预测精度不高怎么办？

**A**:
- 增加训练数据量
- 调整学习率
- 使用集成模型
- 增加训练轮数
- 尝试不同的模型架构

### Q4: 如何选择合适的模型？

**A**:
- 有3D结构 → CGCNN
- 只有MOFid → MOFormer  
- 追求最高精度 → 集成模型
- 快速筛选 → MOFormer

## 技术支持

- 📧 Email: support@mof-platform.com
- 🐛 Issues: GitHub Issues
- 📖 文档: docs/
- 💬 讨论: GitHub Discussions

## 参考文献

1. **CGCNN**: Xie, T. & Grossman, J. C. *Phys. Rev. Lett.* **120**, 145301 (2018)
2. **MOFormer**: Cao, Z. et al. *J. Am. Chem. Soc.* **145**, 2958-2967 (2023)
3. **QMOF**: Rosen, A. S. et al. *npj Comput. Mater.* **8**, 112 (2022)


