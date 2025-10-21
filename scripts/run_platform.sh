#!/bin/bash
# MOF预测平台启动脚本

echo "====================================================="
echo "🧪 MOF预测平台"
echo "====================================================="

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查是否安装了依赖
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit未安装"
    echo "请先运行: bash scripts/install_dependencies.sh"
    exit 1
fi

echo "✅ 依赖检查通过"
echo ""
echo "🚀 启动Streamlit应用..."
echo ""
echo "访问地址: http://localhost:8501"
echo "按 Ctrl+C 停止服务"
echo ""
echo "====================================================="

# 启动Streamlit
streamlit run src/app.py --server.port 8501 --server.address localhost


