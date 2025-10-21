# MOF-Prediction-Platform

基于 QMOF 数据库的金属有机框架材料性能预测平台

集成了 MOFormer、CGCNN 等先进的深度学习模型，用于预测 MOF 材料的各种性能。

## 🚀 快速开始

详细教程请查看 [GETTING_STARTED.md](GETTING_STARTED.md)

## 📊 数据准备

本平台需要 QMOF 数据库：

1. **下载 QMOF 数据库**：
   - 访问：https://github.com/arosen93/QMOF
   - 或：https://materialsproject.org/
   
2. **数据路径配置**：
   - 编辑 `src/config.py` 中的 `QMOF_CONFIG`
   - 将路径指向你的 QMOF 数据位置

⚠️ **注意**：QMOF 数据库较大（数 GB），不包含在本仓库中。请单独下载并配置路径。

## 🛠️ 技术栈

- **后端框架**：Streamlit
- **深度学习**：PyTorch
- **模型**：MOFormer、CGCNN、Ensemble
- **数据处理**：Pandas、NumPy

## 📁 项目结构

```
MOF-Prediction-Platform/
├── src/              # 源代码
├── scripts/          # 运行脚本
├── docs/            # 文档
├── data/            # 数据目录（本地使用，不上传到 git）
└── models/          # 模型文件
```

## 📖 文档

- [快速开始指南](GETTING_STARTED.md)
- [用户手册](docs/user_guide.md)
- [项目概述](PROJECT_SUMMARY.md)

## 📄 许可证

MIT License
