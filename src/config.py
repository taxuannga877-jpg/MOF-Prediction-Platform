"""
配置文件 - MOF预测平台
Configuration File - MOF Prediction Platform
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
# Path Configuration

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
QMOF_DATA_DIR = DATA_DIR / "qmof"

# 模型目录
MODELS_DIR = ROOT_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# 日志目录
LOGS_DIR = ROOT_DIR / "logs"

# 资源目录
ASSETS_DIR = ROOT_DIR / "assets"

# 确保目录存在
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, QMOF_DATA_DIR,
                 MODELS_DIR, PRETRAINED_DIR, SAVED_MODELS_DIR, LOGS_DIR, ASSETS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================== QMOF数据配置 ====================
# QMOF Data Configuration

QMOF_CONFIG = {
    # QMOF数据文件路径（用户需要配置）
    "qmof_json": "/home/tangboshi/QMOF/qmof_database/qmof_database/qmof.json",
    "qmof_csv": "/home/tangboshi/QMOF/qmof_database/qmof_database/qmof.csv",
    "qmof_structure_data": "/home/tangboshi/QMOF/qmof_database/qmof_database/qmof_structure_data.json",
    "relaxed_structures_zip": "/home/tangboshi/QMOF/qmof_database/qmof_database/relaxed_structures.zip",
    "relaxed_structures_dir": "/home/tangboshi/QMOF/qmof_database/qmof_database/relaxed_structures",
}


# ==================== 模型配置 ====================
# Model Configuration

# CGCNN配置
CGCNN_CONFIG = {
    "atom_fea_len": 64,
    "h_fea_len": 128,
    "n_conv": 3,
    "n_h": 1,
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 32,
    "epochs": 100,
    "random_seed": 42,
}

# MOFormer配置
MOFORMER_CONFIG = {
    "model_name": "bert-base-uncased",  # 基础Transformer
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 6,
    "max_length": 512,
    "lr": 2e-5,
    "batch_size": 16,
    "epochs": 50,
    "warmup_steps": 500,
}

# 集成模型配置
ENSEMBLE_CONFIG = {
    "cgcnn_weight": 0.6,
    "moformer_weight": 0.4,
    "voting_method": "weighted",  # weighted, average, max
}


# ==================== 训练配置 ====================
# Training Configuration

TRAINING_CONFIG = {
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "cross_validation": False,
    "cv_folds": 5,
    "early_stopping": True,
    "patience": 10,
    "save_best_model": True,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}


# ==================== 预测任务配置 ====================
# Prediction Task Configuration

SUPPORTED_PROPERTIES = {
    # QMOF CSV 实际列名（匹配数据库）
    "outputs.pbe.bandgap": {
        "name": "能带隙 (PBE Band Gap)",
        "unit": "eV",
        "type": "regression",
        "description": "PBE泛函计算的电子能带隙",
    },
    "outputs.hse06.bandgap": {
        "name": "能带隙 (HSE06 Band Gap)",
        "unit": "eV",
        "type": "regression",
        "description": "HSE06泛函计算的电子能带隙",
    },
    "outputs.pbe.cbm": {
        "name": "导带最小值 (CBM)",
        "unit": "eV",
        "type": "regression",
        "description": "导带最小值",
    },
    "outputs.pbe.vbm": {
        "name": "价带最大值 (VBM)",
        "unit": "eV",
        "type": "regression",
        "description": "价带最大值",
    },
    "info.pld": {
        "name": "孔限制直径 (PLD)",
        "unit": "Å",
        "type": "regression",
        "description": "孔隙限制直径",
    },
    "info.lcd": {
        "name": "最大腔体直径 (LCD)",
        "unit": "Å",
        "type": "regression",
        "description": "最大腔体直径",
    },
    "info.density": {
        "name": "密度",
        "unit": "g/cm³",
        "type": "regression",
        "description": "材料密度",
    },
    "outputs.pbe.energy_total": {
        "name": "总能量 (PBE)",
        "unit": "eV",
        "type": "regression",
        "description": "PBE泛函计算的总能量",
    },
    "info.volume": {
        "name": "晶胞体积",
        "unit": "Å³",
        "type": "regression",
        "description": "晶胞体积",
    },
    "info.natoms": {
        "name": "原子数",
        "unit": "个",
        "type": "regression",
        "description": "晶胞中的原子总数",
    },
}


# ==================== 可视化配置 ====================
# Visualization Configuration

PLOT_CONFIG = {
    "theme": "plotly_white",
    "color_scheme": "viridis",
    "figure_size": (10, 6),
    "dpi": 100,
    "font_size": 12,
    "title_font_size": 16,
}


# ==================== 界面配置 ====================
# UI Configuration

UI_CONFIG = {
    "page_title": "MOF预测平台",
    "page_icon": "🧪",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 200,  # MB
}


# ==================== 日志配置 ====================
# Logging Configuration

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file": LOGS_DIR / "platform.log",
}


# ==================== 数据处理配置 ====================
# Data Processing Configuration

DATA_PROCESSING_CONFIG = {
    "max_atoms": 500,  # 最大原子数
    "max_features": 100,  # 最大特征数
    "normalize": True,
    "remove_outliers": True,
    "outlier_std": 3,  # 异常值标准差阈值
}


# ==================== 模型路由规则 ====================
# Model Routing Rules

ROUTING_RULES = {
    ".cif": "cgcnn",
    ".json": {
        "has_structure": "cgcnn",
        "has_mofid": "moformer",
        "default": "moformer",
    },
    ".csv": "moformer",
    "auto": "ensemble",  # 自动集成
}


# ==================== 预训练模型路径 ====================
# Pretrained Model Paths

PRETRAINED_MODELS = {
    "cgcnn": {
        "bandgap": PRETRAINED_DIR / "cgcnn_bandgap.pth",
        "formation_energy": PRETRAINED_DIR / "cgcnn_formation_energy.pth",
    },
    "moformer": {
        "bandgap": PRETRAINED_DIR / "moformer_bandgap.pth",
        "properties": PRETRAINED_DIR / "moformer_multi_properties.pth",
    },
}


# ==================== 开发模式 ====================
# Development Mode

DEBUG_MODE = os.environ.get("DEBUG", "False").lower() == "true"
VERBOSE = os.environ.get("VERBOSE", "False").lower() == "true"


# ==================== API配置 ====================
# API Configuration

API_CONFIG = {
    "enable": False,
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
}


if __name__ == "__main__":
    print("=" * 50)
    print("MOF预测平台配置信息")
    print("=" * 50)
    print(f"项目根目录: {ROOT_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print(f"调试模式: {DEBUG_MODE}")
    print("=" * 50)


