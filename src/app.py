"""
MOF预测平台 - Streamlit主应用
MOF Prediction Platform - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# 添加路径
sys.path.append(str(Path(__file__).parent))

from config import *
from utils import DataLoader, DataProcessor, ModelRouter, FileHandler, setup_logger
from models import CGCNNModel, MOFormerModel, EnsembleModel

# 页面配置
st.set_page_config(
    page_title="MOF预测平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# 侧边栏
with st.sidebar:
    # 显示 logo（如果存在）
    logo_path = ASSETS_DIR / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path))
    
    st.title("🧪 MOF预测平台")
    st.markdown("---")
    
    page = st.radio(
        "导航",
        ["🏠 主页", "📂 数据管理", "🤖 模型训练", "🔮 性质预测", "📊 结果分析", "⚙️ 设置"],
        key="navigation"
    )
    
    st.markdown("---")
    st.markdown("### 📈 系统状态")
    st.info(f"数据已加载: {'✅' if st.session_state.data_loaded else '❌'}")
    st.info(f"模型已训练: {'✅' if st.session_state.model and st.session_state.model.is_trained else '❌'}")

# 主页
if page == "🏠 主页":
    st.title("🧪 MOF预测平台")
    st.markdown("### 基于CGCNN和MOFormer的智能金属有机框架材料性质预测系统")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤖 双模型架构")
        st.write("""
        - **CGCNN**: 基于3D晶体结构
        - **MOFormer**: 基于MOFid文本
        - **集成模型**: 结合两者优势
        """)
    
    with col2:
        st.markdown("#### 📊 支持的性质")
        st.write("""
        - 能带隙 (Band Gap)
        - 导带/价带位置
        - 孔径特征
        - 密度和能量
        """)
    
    with col3:
        st.markdown("#### 🔍 核心功能")
        st.write("""
        - 智能模型路由
        - 交互式可视化
        - SHAP可解释性
        - 批量预测
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 快速开始")
    st.info("""
    1. 📂 **数据管理**: 上传数据或加载QMOF数据集
    2. 🤖 **模型训练**: 选择模型并训练
    3. 🔮 **性质预测**: 对新MOF进行预测
    4. 📊 **结果分析**: 可视化和解释预测结果
    """)
    
    st.markdown("### 📚 参考文献")
    st.markdown("""
    - **CGCNN**: Xie & Grossman, *Phys. Rev. Lett.* 2018
    - **MOFormer**: Cao et al., *JACS* 2023  
    - **QMOF**: Rosen et al., *npj Comput. Mater.* 2022
    """)

# 数据管理页面
elif page == "📂 数据管理":
    st.title("📂 数据管理")
    
    tab1, tab2, tab3 = st.tabs(["📤 上传数据", "📦 QMOF数据集", "📊 数据预览"])
    
    with tab1:
        st.markdown("### 上传本地数据")
        
        upload_type = st.selectbox(
            "选择数据类型",
            ["CIF文件（晶体结构）", "JSON文件（QMOF格式）", "CSV文件（表格数据）"]
        )
        
        uploaded_file = st.file_uploader(
            "选择文件",
            type=['cif', 'json', 'csv', 'zip'],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            st.success(f"✅ 文件已上传: {uploaded_file.name}")
            
            # 保存文件
            save_path = RAW_DATA_DIR / uploaded_file.name
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🔄 加载数据"):
                try:
                    data_loader = DataLoader(QMOF_CONFIG)
                    data, data_type = data_loader.load_from_file(save_path)
                    
                    st.session_state.data = data
                    st.session_state.data_type = data_type
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ 数据加载成功！类型: {data_type}")
                    
                    # 显示摘要
                    summary = data_loader.get_data_summary(data, data_type)
                    st.json(summary)
                    
                except Exception as e:
                    st.error(f"❌ 加载失败: {str(e)}")
    
    with tab2:
        st.markdown("### QMOF数据集")
        
        qmof_source = st.selectbox(
            "选择QMOF数据源",
            ["qmof.json（属性数据）", "qmof.csv（表格格式）", 
             "结构数据（CIF）", "完整结构数据（JSON）"]
        )
        
        limit = st.number_input("限制加载数量（0=全部）", min_value=0, max_value=20373, value=100)
        
        if st.button("📥 加载QMOF数据"):
            try:
                data_loader = DataLoader(QMOF_CONFIG)
                
                source_map = {
                    "qmof.json（属性数据）": "json",
                    "qmof.csv（表格格式）": "csv",
                    "结构数据（CIF）": "cif",
                    "完整结构数据（JSON）": "structures",
                }
                
                with st.spinner("⏳ 加载QMOF数据中..."):
                    data, data_type = data_loader.load_qmof_data(
                        source_map[qmof_source],
                        limit=limit if limit > 0 else None
                    )
                
                st.session_state.data = data
                st.session_state.data_type = data_type
                st.session_state.data_loaded = True
                
                st.success(f"✅ QMOF数据加载成功！")
                st.info(f"数据类型: {data_type}")
                
                # 显示统计信息
                if isinstance(data, pd.DataFrame):
                    st.write(f"📊 加载了 {len(data)} 个MOF")
                    st.dataframe(data.head())
                elif isinstance(data, dict):
                    st.write(f"📊 加载了 {len(data)} 个结构")
                elif isinstance(data, list):
                    st.write(f"📊 加载了 {len(data)} 条记录")
                
            except Exception as e:
                st.error(f"❌ 加载失败: {str(e)}")
                st.exception(e)
    
    with tab3:
        st.markdown("### 数据预览")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ 请先加载数据")
        else:
            data = st.session_state.data
            data_type = st.session_state.data_type
            
            st.success(f"当前数据类型: **{data_type}**")
            
            if isinstance(data, pd.DataFrame):
                st.dataframe(data, use_container_width=True)
                
                st.markdown("#### 统计信息")
                st.write(data.describe())
                
            elif isinstance(data, list):
                st.write(f"记录数: {len(data)}")
                st.json(data[:3])  # 显示前3条
                
            elif isinstance(data, dict):
                st.write(f"结构数: {len(data)}")
                st.write("MOF IDs:", list(data.keys())[:10])

# 模型训练页面
elif page == "🤖 模型训练":
    st.title("🤖 模型训练")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ 请先在【数据管理】页面加载数据")
    else:
        # 模型路由推荐
        st.markdown("### 🧭 智能模型推荐")
        
        router = ModelRouter()
        recommendation = router.recommend_model(
            data=st.session_state.data
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"""
            **推荐模型**: {recommendation['primary'].upper()}
            
            **理由**: {recommendation['reason']}
            
            **置信度**: {recommendation['confidence']*100:.0f}%
            """)
        
        with col2:
            st.write("**备选模型**:")
            for alt in recommendation['alternatives']:
                st.write(f"- {alt.upper()}")
        
        # 模型选择
        st.markdown("### 🎯 选择模型")
        
        model_choice = st.selectbox(
            "模型类型",
            ["CGCNN", "MOFormer", "集成模型"],
            index=["cgcnn", "moformer", "ensemble"].index(recommendation['primary'])
        )
        
        # 模型配置
        st.markdown("### ⚙️ 模型配置")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input("训练轮数", min_value=10, max_value=500, value=100)
            batch_size = st.number_input("批次大小", min_value=8, max_value=128, value=32)
        
        with col2:
            learning_rate = st.select_slider(
                "学习率",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                value=1e-3
            )
            train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.8)
        
        with col3:
            val_ratio = st.slider("验证集比例", 0.05, 0.3, 0.1)
            test_ratio = 1.0 - train_ratio - val_ratio
            st.metric("测试集比例", f"{test_ratio:.2f}")
        
        # 目标属性选择
        st.markdown("### 🎯 目标属性")
        
        property_choice = st.selectbox(
            "选择要预测的性质",
            list(SUPPORTED_PROPERTIES.keys()),
            format_func=lambda x: SUPPORTED_PROPERTIES[x]['name']
        )
        
        st.info(f"**描述**: {SUPPORTED_PROPERTIES[property_choice]['description']}")
        st.info(f"**单位**: {SUPPORTED_PROPERTIES[property_choice]['unit']}")
        
        # 开始训练
        if st.button("🚀 开始训练", type="primary"):
            try:
                # 准备数据
                data_processor = DataProcessor()
                
                with st.spinner("⏳ 准备训练数据..."):
                    # 这里需要根据实际数据类型处理
                    # 简化演示
                    st.info("📊 数据准备中...")
                
                # 创建模型
                with st.spinner(f"🔨 构建{model_choice}模型..."):
                    if model_choice == "CGCNN":
                        config = CGCNN_CONFIG.copy()
                        config.update({'batch_size': batch_size, 'lr': learning_rate})
                        model = CGCNNModel(config)
                    elif model_choice == "MOFormer":
                        config = MOFORMER_CONFIG.copy()
                        config.update({'batch_size': batch_size, 'lr': learning_rate})
                        model = MOFormerModel(config)
                    else:
                        config = ENSEMBLE_CONFIG.copy()
                        model = EnsembleModel(config)
                    
                    model.build_model()
                
                st.session_state.model = model
                st.success("✅ 模型构建完成！")
                
                st.info("💡 由于这是演示版本，实际训练需要准备好的数据。请参考文档进行完整训练。")
                
            except Exception as e:
                st.error(f"❌ 训练失败: {str(e)}")
                st.exception(e)

# 性质预测页面
elif page == "🔮 性质预测":
    st.title("🔮 性质预测")
    
    if st.session_state.model is None:
        st.warning("⚠️ 请先训练或加载模型")
    else:
        st.markdown("### 📥 输入数据")
        
        pred_mode = st.radio(
            "预测模式",
            ["单个MOF预测", "批量预测", "从数据集预测"]
        )
        
        if pred_mode == "单个MOF预测":
            if isinstance(st.session_state.model, MOFormerModel):
                mofid = st.text_area(
                    "输入MOFid",
                    placeholder="例如: [Zn].[O-]C(=O)c1ccc(cc1)C(=O)[O-]...",
                    height=100
                )
                
                if st.button("🔮 预测"):
                    if mofid:
                        try:
                            result = st.session_state.model.predict(mofid)
                            st.success(f"✅ 预测结果: **{result:.4f}** eV")
                        except Exception as e:
                            st.error(f"❌ 预测失败: {str(e)}")
            else:
                st.info("💡 请上传CIF文件进行CGCNN预测")
        
        elif pred_mode == "批量预测":
            st.info("📄 上传包含多个MOF的文件")
            # 批量预测逻辑
        
        else:
            st.info("📊 从已加载的数据集中选择预测")

# 结果分析页面
elif page == "📊 结果分析":
    st.title("📊 结果分析")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ 暂无预测结果")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 预测vs真实", "📊 分布分析", "🔍 误差分析", "💡 可解释性"]
        )
        
        with tab1:
            st.markdown("### 预测值 vs 真实值")
            st.info("📊 散点图和回归线")
        
        with tab2:
            st.markdown("### 数值分布")
            st.info("📊 直方图和密度曲线")
        
        with tab3:
            st.markdown("### 误差分析")
            st.info("📊 残差图和误差分布")
        
        with tab4:
            st.markdown("### SHAP可解释性分析")
            st.info("🔍 特征重要性和依赖图")

# 设置页面
elif page == "⚙️ 设置":
    st.title("⚙️ 系统设置")
    
    tab1, tab2, tab3 = st.tabs(["📁 路径配置", "🎨 界面设置", "ℹ️ 关于"])
    
    with tab1:
        st.markdown("### QMOF数据路径")
        
        for key, path in QMOF_CONFIG.items():
            new_path = st.text_input(
                key,
                value=path,
                key=f"path_{key}"
            )
            
            if Path(new_path).exists():
                st.success(f"✅ 路径有效")
            else:
                st.warning(f"⚠️ 路径不存在")
    
    with tab2:
        st.markdown("### 可视化主题")
        theme = st.selectbox("选择主题", ["Light", "Dark", "Auto"])
        
    with tab3:
        st.markdown("### 关于MOF预测平台")
        st.markdown("""
        **版本**: 1.0.0
        
        **开发团队**: MOF预测平台开发组
        
        **技术栈**:
        - Python 3.9+
        - PyTorch
        - Streamlit
        - Pymatgen
        
        **许可证**: MIT License
        """)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🧪 MOF预测平台 | Built with ❤️ for MOF Research Community"
    "</div>",
    unsafe_allow_html=True
)


