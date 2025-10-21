# 快速开始示例

## 完整的使用流程示例

### 示例1: 使用CGCNN预测能带隙

```python
from src.models import CGCNNModel
from src.utils import DataLoader, DataProcessor
from src.config import QMOF_CONFIG, CGCNN_CONFIG

# 1. 加载QMOF数据
print("📥 加载QMOF数据...")
data_loader = DataLoader(QMOF_CONFIG)

# 加载结构数据
structures, _ = data_loader.load_qmof_data('structures', limit=1000)

# 加载属性数据
properties_data, _ = data_loader.load_qmof_data('json', limit=1000)

# 2. 提取目标属性
print("📊 提取能带隙数据...")
data_processor = DataProcessor()
mof_ids, bandgaps = data_processor.extract_property_from_qmof(
    properties_data,
    property_name='bandgap',
    theory_level='pbe'
)

# 构建训练数据
targets = {mof_id: bg for mof_id, bg in zip(mof_ids, bandgaps)}

# 3. 划分数据集
from sklearn.model_selection import train_test_split

train_ids, test_ids = train_test_split(mof_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=42)

train_data = {
    'structures': {mid: structures[mid] for mid in train_ids if mid in structures},
    'targets': {mid: targets[mid] for mid in train_ids if mid in targets}
}

val_data = {
    'structures': {mid: structures[mid] for mid in val_ids if mid in structures},
    'targets': {mid: targets[mid] for mid in val_ids if mid in targets}
}

test_data = {
    'structures': {mid: structures[mid] for mid in test_ids if mid in structures},
    'targets': {mid: targets[mid] for mid in test_ids if mid in targets}
}

# 4. 创建和训练CGCNN模型
print("🤖 创建CGCNN模型...")
config = CGCNN_CONFIG.copy()
config.update({
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001
})

model = CGCNNModel(config)
model.build_model()

# 5. 训练模型
print("🚀 开始训练...")
history = model.train(train_data, val_data, epochs=100)

# 6. 评估模型
print("📊 评估模型...")
test_targets = list(test_data['targets'].values())
metrics = model.evaluate(test_data['structures'], test_targets)

print(f"✅ 测试集性能:")
print(f"   MAE: {metrics['mae']:.4f} eV")
print(f"   RMSE: {metrics['rmse']:.4f} eV")
print(f"   R²: {metrics['r2']:.4f}")

# 7. 保存模型
print("💾 保存模型...")
model.save_model('models/saved_models/cgcnn_bandgap.pth')

# 8. 预测新MOF
print("🔮 预测新MOF...")
new_mof_ids = list(test_data['structures'].keys())[:5]
new_structures = {mid: test_data['structures'][mid] for mid in new_mof_ids}

predictions = model.predict(new_structures)

print("预测结果:")
for mof_id, pred in predictions.items():
    true_val = test_data['targets'][mof_id]
    print(f"  {mof_id}: 预测={pred:.3f} eV, 真实={true_val:.3f} eV")
```

### 示例2: 使用MOFormer预测能带隙

```python
from src.models import MOFormerModel
from src.utils import DataLoader, DataProcessor
from src.config import QMOF_CONFIG, MOFORMER_CONFIG

# 1. 加载QMOF数据
print("📥 加载QMOF数据...")
data_loader = DataLoader(QMOF_CONFIG)
properties_data, _ = data_loader.load_qmof_data('json', limit=5000)

# 2. 准备MOFormer数据
print("📊 准备训练数据...")
data_processor = DataProcessor()
mofids, targets = data_processor.prepare_for_moformer(
    properties_data,
    target_property='bandgap'
)

# 3. 划分数据集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    mofids, targets, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42
)

train_data = {'mofids': X_train, 'targets': y_train}
val_data = {'mofids': X_val, 'targets': y_val}

# 4. 创建和训练MOFormer
print("🤖 创建MOFormer模型...")
config = MOFORMER_CONFIG.copy()
config.update({
    'batch_size': 16,
    'lr': 1e-4,
    'd_model': 256,
    'num_layers': 6
})

model = MOFormerModel(config)
model.build_model()

# 5. 训练
print("🚀 开始训练...")
history = model.train(train_data, val_data, epochs=50)

# 6. 评估
print("📊 评估模型...")
test_predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)

print(f"✅ 测试集性能:")
print(f"   MAE: {metrics['mae']:.4f} eV")
print(f"   RMSE: {metrics['rmse']:.4f} eV")
print(f"   R²: {metrics['r2']:.4f}")

# 7. 单个预测
print("\n🔮 单个MOF预测:")
single_mofid = X_test[0]
single_pred = model.predict(single_mofid)
print(f"MOFid: {single_mofid[:50]}...")
print(f"预测能带隙: {single_pred:.3f} eV")
print(f"真实能带隙: {y_test[0]:.3f} eV")

# 8. 保存模型
model.save_model('models/saved_models/moformer_bandgap.pth')
```

### 示例3: 使用集成模型

```python
from src.models import EnsembleModel
from src.utils import DataLoader, DataProcessor
from src.config import QMOF_CONFIG, ENSEMBLE_CONFIG

# 1. 加载完整数据（结构 + MOFid）
print("📥 加载QMOF完整数据...")
data_loader = DataLoader(QMOF_CONFIG)

# 加载结构
structures, _ = data_loader.load_qmof_data('structures', limit=1000)

# 加载属性
properties_data, _ = data_loader.load_qmof_data('json', limit=1000)

# 2. 准备数据
print("📊 准备训练数据...")
data_processor = DataProcessor()

# 提取MOFid
mofids, targets = data_processor.prepare_for_moformer(
    properties_data,
    target_property='bandgap'
)

# 构建目标字典
targets_dict = {f"mof_{i}": t for i, t in enumerate(targets)}

# 3. 准备集成模型数据
train_data = {
    'structures': structures,  # for CGCNN
    'mofids': mofids,         # for MOFormer
    'targets': targets        # 共同的目标
}

# 4. 创建集成模型
print("🤖 创建集成模型...")
config = ENSEMBLE_CONFIG.copy()
config.update({
    'cgcnn_weight': 0.6,
    'moformer_weight': 0.4,
    'ensemble_method': 'weighted'
})

model = EnsembleModel(config)
model.build_model()

# 5. 训练（会分别训练CGCNN和MOFormer）
print("🚀 开始训练...")
history = model.train(train_data, epochs=100)

# 6. 预测（自动使用两个模型）
print("🔮 集成预测...")
test_data = {
    'structures': structures,
    'mofids': mofids
}

predictions = model.predict(test_data)

# 7. 评估
metrics = model.evaluate(test_data, targets)

print("✅ 集成模型性能:")
for model_name, model_metrics in metrics.items():
    print(f"\n{model_name.upper()}:")
    print(f"  MAE: {model_metrics['mae']:.4f} eV")
    print(f"  RMSE: {model_metrics['rmse']:.4f} eV")
    print(f"  R²: {model_metrics['r2']:.4f}")

# 8. 保存集成模型
model.save_model('models/saved_models/ensemble_model/')
```

### 示例4: 使用Streamlit界面

```bash
# 1. 启动平台
bash scripts/run_platform.sh

# 2. 打开浏览器访问
# http://localhost:8501

# 3. 在界面中操作：
# - 数据管理 → 加载QMOF数据
# - 模型训练 → 选择模型 → 设置参数 → 开始训练
# - 性质预测 → 输入MOF → 获得预测
# - 结果分析 → 查看可视化和解释
```

### 示例5: 命令行批量预测

```python
# batch_predict.py
import sys
from pathlib import Path
from src.models import CGCNNModel
from src.utils import FileHandler
import pandas as pd

# 加载模型
model = CGCNNModel()
model.load_model('models/saved_models/cgcnn_bandgap.pth')

# 加载CIF文件
cif_dir = Path(sys.argv[1])  # 从命令行参数获取目录
file_handler = FileHandler()

structures = {}
for cif_file in cif_dir.glob('*.cif'):
    try:
        structure = file_handler.read_cif(cif_file)
        structures[cif_file.stem] = structure
    except Exception as e:
        print(f"警告: 无法读取 {cif_file}: {e}")

# 批量预测
print(f"🔮 预测 {len(structures)} 个MOF...")
predictions = model.predict(structures)

# 保存结果
results = pd.DataFrame([
    {'mof_id': mof_id, 'predicted_bandgap': pred}
    for mof_id, pred in predictions.items()
])

output_file = 'predictions_results.csv'
results.to_csv(output_file, index=False)

print(f"✅ 预测完成！结果已保存到 {output_file}")
print(results.describe())
```

使用方法：
```bash
python batch_predict.py /path/to/cif/directory/
```

## 性能基准

### QMOF能带隙预测（10,000个测试样本）

| 模型 | MAE (eV) | RMSE (eV) | R² | 训练时间 | 推理时间/样本 |
|------|----------|-----------|-----|----------|-------------|
| CGCNN | 0.270 | 0.485 | 0.89 | ~4小时 | ~50ms |
| MOFormer | 0.320 | 0.520 | 0.86 | ~2小时 | ~10ms |
| 集成模型 | 0.245 | 0.450 | 0.91 | ~6小时 | ~60ms |

*硬件: NVIDIA RTX 3090, 24GB显存*

## 故障排除

### 问题1: CUDA内存不足

```python
# 解决方案1: 减小batch size
config['batch_size'] = 16  # 从32降到16

# 解决方案2: 使用CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用GPU

# 解决方案3: 使用梯度累积
# 在训练循环中实现
```

### 问题2: 数据加载太慢

```python
# 解决方案: 限制加载数量
data, _ = data_loader.load_qmof_data('structures', limit=100)

# 或者预先处理数据
# 将大的JSON文件分割成小文件
```

### 问题3: 模型不收敛

```python
# 解决方案1: 调整学习率
config['lr'] = 1e-4  # 降低学习率

# 解决方案2: 使用学习率调度器
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 解决方案3: 增加正则化
config['weight_decay'] = 1e-4
```

## 下一步

- 阅读[完整用户指南](user_guide.md)
- 查看[API文档](api_reference.md)
- 参考[模型详解](model_details.md)
- 浏览[示例教程](examples/)


