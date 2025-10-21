# 🚀 快速开始指南

## 欢迎使用MOF预测平台！

这是一个5分钟快速入门指南，帮你快速上手平台。

---

## 📋 前置要求

- Python 3.8+
- 8GB+ RAM
- （可选）NVIDIA GPU with CUDA

---

## ⚡ 三步启动

### 步骤1: 安装依赖

```bash
cd /home/tangboshi/MOF-Prediction-Platform
bash scripts/install_dependencies.sh
```

### 步骤2: 配置QMOF数据路径

数据路径已预配置在 `src/config.py`:

```python
QMOF_CONFIG = {
    "qmof_json": "/home/tangboshi/QMOF/qmof_database/qmof_database/qmof.json",
    "qmof_csv": "/home/tangboshi/QMOF/qmof_database/qmof_database/qmof.csv",
    "qmof_structure_data": "/home/tangboshi/QMOF/qmof_database/qmof_database/qmof_structure_data.json",
    "relaxed_structures_zip": "/home/tangboshi/QMOF/qmof_database/qmof_database/relaxed_structures.zip",
}
```

如果路径不同，请修改此配置。

### 步骤3: 启动平台

```bash
bash scripts/run_platform.sh
```

然后访问: **http://localhost:8501**

---

## 🎯 第一次使用

### 在Web界面中：

1. **📂 数据管理**
   - 点击"📦 QMOF数据集"标签
   - 选择"qmof.json（属性数据）"
   - 限制加载数量: 100（测试用）
   - 点击"📥 加载QMOF数据"

2. **🤖 模型训练**
   - 系统会自动推荐模型（MOFormer for JSON数据）
   - 选择目标属性: "能带隙 (Band Gap)"
   - 设置参数:
     - 训练轮数: 50
     - 批次大小: 16
     - 学习率: 1e-4
   - 点击"🚀 开始训练"

3. **🔮 性质预测**
   - 输入MOFid或上传CIF文件
   - 点击"🔮 预测"
   - 查看结果

---

## 💻 Python API 快速示例

### 示例1: 使用MOFormer（最快）

```python
from src.models import MOFormerModel
from src.utils import DataLoader
from src.config import QMOF_CONFIG

# 加载数据
loader = DataLoader(QMOF_CONFIG)
data, _ = loader.load_qmof_data('json', limit=100)

# 准备训练数据
# ... (见完整示例)

# 创建模型
model = MOFormerModel()
model.build_model()

# 训练
model.train(train_data, val_data, epochs=50)

# 预测
prediction = model.predict("your_mofid_string")
print(f"预测能带隙: {prediction:.3f} eV")
```

### 示例2: 使用CGCNN（最准确）

```python
from src.models import CGCNNModel
from src.utils import DataLoader

# 加载结构数据
loader = DataLoader(QMOF_CONFIG)
structures, _ = loader.load_qmof_data('structures', limit=100)

# 创建模型
model = CGCNNModel()
model.build_model()

# 训练
model.train(train_data, val_data, epochs=100)

# 预测
predictions = model.predict(test_structures)
```

---

## 📊 推荐的学习路径

### 新手（第1天）
1. ✅ 使用Web界面加载QMOF数据
2. ✅ 尝试MOFormer模型（快速）
3. ✅ 进行简单预测

### 进阶（第2-3天）
1. ✅ 学习Python API使用
2. ✅ 尝试CGCNN模型
3. ✅ 了解模型参数调整

### 高级（第4-7天）
1. ✅ 使用集成模型
2. ✅ 自定义数据训练
3. ✅ 批量预测和导出

---

## 🔍 关键概念

### 数据格式

| 格式 | 用途 | 适用模型 | 大小 |
|------|------|---------|------|
| **CIF** | 晶体结构 | CGCNN | 小 |
| **JSON (MOFid)** | 文本表示 | MOFormer | 中 |
| **JSON (Structure)** | 完整结构 | CGCNN | 大 |
| **CSV** | 表格数据 | MOFormer | 小 |

### 模型选择

| 需求 | 推荐模型 | 原因 |
|------|---------|------|
| 最高精度 | 集成模型 | 结合两者优势 |
| 快速筛选 | MOFormer | 速度快5倍 |
| 精确预测 | CGCNN | MAE 0.27 eV |
| 大规模预测 | MOFormer | 低内存占用 |

---

## 🎓 学习资源

1. **完整用户指南**: `docs/user_guide.md`
2. **代码示例**: `docs/quickstart_example.md`
3. **API文档**: `docs/api_reference.md`
4. **项目总结**: `PROJECT_SUMMARY.md`

---

## 🆘 遇到问题？

### 常见问题解决

**Q: 内存不足**
```bash
# 解决: 减小batch_size或限制数据量
limit = 50  # 减少加载数量
batch_size = 8  # 减小批次
```

**Q: CUDA错误**
```python
# 解决: 使用CPU模式
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

**Q: 数据加载失败**
```bash
# 检查路径
ls /home/tangboshi/QMOF/qmof_database/qmof_database/qmof.json

# 如果路径不对，修改 src/config.py
```

### 获取帮助

- 📖 查看文档: `docs/`
- 🐛 提交Issue: GitHub Issues
- 💬 讨论: GitHub Discussions

---

## ✅ 快速检查清单

在开始之前，确保：

- [ ] Python 3.8+ 已安装
- [ ] QMOF数据已下载
- [ ] 依赖已安装 (`pip list | grep torch`)
- [ ] 数据路径已配置 (`src/config.py`)
- [ ] 平台可以启动 (`http://localhost:8501` 可访问)

---

## 🎯 下一步

完成快速开始后，推荐：

1. 📚 阅读[完整用户指南](docs/user_guide.md)
2. 💻 尝试[Python示例](docs/quickstart_example.md)
3. 🔬 使用自己的数据训练模型
4. 📊 探索可视化和可解释性功能

---

<div align="center">

**🎉 准备好了！开始你的MOF预测之旅吧！**

[完整文档](docs/user_guide.md) | [示例代码](docs/quickstart_example.md) | [项目总结](PROJECT_SUMMARY.md)

</div>


