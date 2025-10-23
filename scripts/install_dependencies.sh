#!/bin/bash
# MOF预测平台依赖安装脚本

echo "====================================================="
echo "MOF预测平台 - 依赖安装脚本"
echo "====================================================="

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo ""
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo ""
echo "📦 升级pip..."
pip install --upgrade pip

# 安装核心依赖
echo ""
echo "📦 安装核心依赖..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "📦 安装科学计算库..."
pip install numpy pandas scipy scikit-learn

echo ""
echo "📦 安装化学和材料科学库..."
pip install pymatgen ase

echo ""
echo "📦 安装Web框架..."
pip install streamlit plotly

echo ""
echo "📦 安装其他依赖..."
pip install -r requirements.txt

echo ""
echo "====================================================="
echo "✅ 依赖安装完成！"
echo "====================================================="
echo ""
echo "下一步："
echo "1. 运行 bash scripts/run_platform.sh 启动平台"
echo "2. 访问 http://localhost:8501"
echo ""


