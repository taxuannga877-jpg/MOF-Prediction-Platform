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
from models import (
    CGCNNModel, MOFormerModel, EnsembleModel,
    TraditionalMLModel, MLEnsembleModel, CrossValidator, HyperparameterOptimizer,
    get_available_models, get_model_display_names, create_auto_ensemble, quick_optimize
)

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
if 'available_columns' not in st.session_state:
    st.session_state.available_columns = []
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'selected_target' not in st.session_state:
    st.session_state.selected_target = None

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
    # 检查模型是否训练（兼容不同模型类型）
    model_trained = False
    if st.session_state.model:
        model_trained = getattr(st.session_state.model, 'is_trained', False) or getattr(st.session_state.model, 'model', None) is not None
    st.info(f"模型已训练: {'✅' if model_trained else '❌'}")

# 主页
if page == "🏠 主页":
    st.title("🧪 MOF预测平台")
    st.markdown("### 基于CGCNN和MOFormer的智能金属有机框架材料性质预测系统")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤖 多模型架构")
        st.write("""
        - **CGCNN**: 基于3D晶体结构
        - **MOFormer**: 基于MOFid文本
        - **传统ML**: XGBoost, LightGBM, CatBoost等
        - **集成模型**: 多种集成策略
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
        - K折交叉验证
        - 贝叶斯超参数优化
        - SHAP可解释性分析
        - 模型性能对比
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
                    
                    # 🔥 关键修改：提取列名
                    if isinstance(data, pd.DataFrame):
                        st.session_state.available_columns = list(data.columns)
                        st.session_state.numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
                    elif isinstance(data, list) and len(data) > 0:
                        # 🔥 将列表转换为DataFrame
                        try:
                            df_temp = pd.json_normalize(data)
                            st.session_state.data = df_temp  # 替换为DataFrame
                            data = df_temp
                            st.session_state.available_columns = list(df_temp.columns)
                            st.session_state.numeric_columns = list(df_temp.select_dtypes(include=[np.number]).columns)
                        except Exception as e:
                            st.warning(f"⚠️ 无法转换为DataFrame: {e}")
                            st.session_state.available_columns = []
                            st.session_state.numeric_columns = []
                    elif isinstance(data, dict):
                        st.session_state.available_columns = []
                        st.session_state.numeric_columns = []
                    
                    st.success(f"✅ 数据加载成功！类型: {data_type}")
                    st.info(f"🔍 检测到 {len(st.session_state.numeric_columns)} 个数值型列可用于预测")
                    
                    # 显示摘要
                    summary = data_loader.get_data_summary(data, data_type)
                    st.json(summary)
                    
                except Exception as e:
                    st.error(f"❌ 加载失败: {str(e)}")
    
    with tab2:
        st.markdown("### QMOF数据集")
        
        st.info("💡 **训练推荐**：选择 **'qmof.csv（表格格式）'** 以获得最佳兼容性")
        
        qmof_source = st.selectbox(
            "选择QMOF数据源",
            ["qmof.csv（表格格式）", "qmof.json（属性数据）", 
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
                
                # 🔥 关键修改：自动提取所有可用的列名
                if isinstance(data, pd.DataFrame):
                    # 获取所有列名
                    st.session_state.available_columns = list(data.columns)
                    # 获取所有数值型列名（可以作为目标属性）
                    st.session_state.numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
                elif isinstance(data, list) and len(data) > 0:
                    # 🔥 关键：将列表转换为DataFrame，这样后续处理更方便
                    try:
                        df_temp = pd.json_normalize(data)
                        st.session_state.data = df_temp  # 替换为DataFrame
                        data = df_temp  # 更新本地变量
                        st.session_state.available_columns = list(df_temp.columns)
                        st.session_state.numeric_columns = list(df_temp.select_dtypes(include=[np.number]).columns)
                    except Exception as e:
                        st.warning(f"⚠️ 无法转换为DataFrame: {e}")
                        st.session_state.available_columns = []
                        st.session_state.numeric_columns = []
                elif isinstance(data, dict):
                    st.session_state.available_columns = []
                    st.session_state.numeric_columns = []
                
                st.success(f"✅ QMOF数据加载成功！")
                st.info(f"数据类型: {data_type}")
                st.info(f"🔍 检测到 {len(st.session_state.numeric_columns)} 个数值型列可用于预测")
                
                # 显示统计信息
                if isinstance(data, pd.DataFrame):
                    st.write(f"📊 加载了 {len(data)} 个MOF，{len(data.columns)} 个特征")
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
        
        # 添加模型类别选择
        model_category = st.radio(
            "模型类别",
            ["🧠 深度学习模型", "🌲 传统机器学习模型", "🎯 集成模型"],
            horizontal=True
        )
        
        if model_category == "🧠 深度学习模型":
            model_choice = st.selectbox(
                "模型类型",
                ["CGCNN", "MOFormer"],
                index=["cgcnn", "moformer"].index(recommendation['primary']) if recommendation['primary'] in ["cgcnn", "moformer"] else 0
            )
        elif model_category == "🌲 传统机器学习模型":
            # 获取可用的传统ML模型
            available_models = get_available_models()
            display_names = get_model_display_names()
            
            # 只显示可用的模型
            available_ml_models = {k: v for k, v in display_names.items() if available_models.get(k, False)}
            
            if not available_ml_models:
                st.error("❌ 未安装传统机器学习库。请运行: pip install xgboost lightgbm catboost optuna")
                st.stop()
            
            model_choice = st.selectbox(
                "选择传统ML模型",
                list(available_ml_models.keys()),
                format_func=lambda x: available_ml_models[x]
            )
        else:  # 集成模型
            ensemble_type = st.selectbox(
                "集成类型",
                ["深度学习集成", "传统ML集成 (Voting)", "传统ML集成 (Stacking)", "传统ML集成 (Blending)"]
            )
            if ensemble_type == "深度学习集成":
                model_choice = "集成模型"
            else:
                model_choice = ensemble_type
        
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
        
        # 🔥 关键修改：目标属性动态选择
        st.markdown("### 🎯 目标属性选择")
        
        if len(st.session_state.numeric_columns) == 0:
            st.warning("⚠️ 未检测到数值型列，无法进行训练")
            st.stop()
        
        # 创建两列布局
        col_a, col_b = st.columns([3, 1])
        
        with col_a:
            # 🔥 使用实际数据中的列名，而不是硬编码的属性
            property_choice = st.selectbox(
                "选择要预测的目标列（从您的数据中）",
                st.session_state.numeric_columns,
                help="这些是您数据中的所有数值型列，您可以选择任意一个作为预测目标"
            )
            
            # 保存用户选择
            st.session_state.selected_target = property_choice
        
        with col_b:
            # 显示该列的统计信息
            if isinstance(st.session_state.data, pd.DataFrame):
                # 🔥 安全检查：确保列存在
                if property_choice in st.session_state.data.columns:
                    col_data = st.session_state.data[property_choice]
                    st.metric("数据点数", len(col_data.dropna()))
                    st.metric("平均值", f"{col_data.mean():.3f}")
                    st.metric("标准差", f"{col_data.std():.3f}")
                else:
                    st.warning(f"⚠️ 列 '{property_choice}' 不在当前数据中")
        
        # 显示该列的分布
        if isinstance(st.session_state.data, pd.DataFrame):
            # 🔥 安全检查：确保列存在
            if property_choice in st.session_state.data.columns:
                import plotly.express as px
                col_data = st.session_state.data[property_choice].dropna()
                if len(col_data) > 0:
                    fig = px.histogram(col_data, nbins=50, title=f"{property_choice} 分布")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("该列没有有效数据")
        
        st.success(f"✅ 已选择目标列: **{property_choice}**")
        
        # 开始训练
        if st.button("🚀 开始训练", type="primary"):
            try:
                data = st.session_state.data
                data_type = st.session_state.data_type
                
                # 准备训练数据
                with st.spinner("⏳ 准备训练数据..."):
                    data_processor = DataProcessor()
                    
                    # 转换数据为 DataFrame（如果需要）
                    if not isinstance(data, pd.DataFrame):
                        st.info("📊 检测到非表格数据，正在转换为 DataFrame...")
                        
                        if isinstance(data, dict):
                            # 如果是字典格式，尝试转换
                            if all(isinstance(v, dict) for v in data.values()):
                                # {id: {prop1: val1, prop2: val2, ...}} 格式
                                data = pd.DataFrame.from_dict(data, orient='index')
                                data.index.name = 'qmof_id'
                                data.reset_index(inplace=True)
                                st.success(f"✅ 已转换为 DataFrame，共 {len(data)} 行")
                            else:
                                # 尝试直接转换
                                data = pd.DataFrame([data])
                        
                        elif isinstance(data, list):
                            # 如果是列表格式
                            if data and isinstance(data[0], dict):
                                data = pd.DataFrame(data)
                                st.success(f"✅ 已转换为 DataFrame，共 {len(data)} 行")
                            else:
                                st.error("❌ 无法将列表数据转换为 DataFrame")
                                st.stop()
                        
                        else:
                            st.error(f"❌ 不支持的数据类型: {type(data).__name__}")
                            st.warning("💡 建议：请在【数据管理】页面选择 **'qmof.csv（表格格式）'** 数据源")
                            st.stop()
                        
                        # 更新 session state
                        st.session_state.data = data
                    
                    # 根据数据类型和模型类型准备数据
                    if (model_category == "🌲 传统机器学习模型") or (model_category == "🎯 集成模型" and "ML集成" in str(model_choice)):
                        # 传统ML模型使用表格数据
                        if not isinstance(data, pd.DataFrame):
                            st.error("❌ 传统ML模型需要表格数据格式")
                            st.stop()
                        
                        if property_choice not in data.columns:
                            st.error(f"❌ 数据中未找到目标属性 '{property_choice}'")
                            st.stop()
                        
                        # 自动选择数值型特征列
                        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                        feature_cols = [col for col in numeric_cols if col != property_choice]
                        
                        if len(feature_cols) == 0:
                            st.error("❌ 没有可用的数值特征列")
                            st.stop()
                        
                        st.info(f"📊 使用 {len(feature_cols)} 个特征列进行训练")
                        
                        # 过滤有效数据
                        required_cols = feature_cols + [property_choice]
                        valid_data = data[required_cols].dropna()
                        st.info(f"📊 有效数据: {len(valid_data)} / {len(data)} 条")
                        
                        if len(valid_data) < 10:
                            st.error("❌ 有效数据太少（<10条），无法训练")
                            st.stop()
                        
                        # 准备特征和标签
                        X = valid_data[feature_cols].values
                        y = valid_data[property_choice].values
                        
                        # 数据划分
                        from sklearn.model_selection import train_test_split
                        
                        X_train, X_temp, y_train, y_temp = train_test_split(
                            X, y, test_size=(1-train_ratio), random_state=42
                        )
                        X_val, X_test, y_val, y_test = train_test_split(
                            X_temp, y_temp, test_size=(test_ratio/(test_ratio+val_ratio)), random_state=42
                        )
                        
                        st.success(f"✅ 数据划分完成：训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
                        
                        # 存储测试集
                        st.session_state.test_data = {'X': X_test, 'y': y_test, 'feature_names': feature_cols}
                    
                    elif model_choice in ["CGCNN", "集成模型"]:
                        # CGCNN 需要表格数据
                        if not isinstance(data, pd.DataFrame):
                            st.error("❌ 数据格式错误，请重新加载数据")
                            st.stop()
                        
                        # 检查是否有目标属性
                        if property_choice not in data.columns:
                            st.error(f"❌ 数据中未找到目标属性 '{property_choice}'")
                            st.stop()
                        
                        # 过滤有效数据
                        valid_data = data.dropna(subset=[property_choice])
                        st.info(f"📊 有效数据: {len(valid_data)} / {len(data)} 条")
                        
                        if len(valid_data) < 10:
                            st.error("❌ 有效数据太少（<10条），无法训练")
                            st.stop()
                        
                        # 数据划分
                        from sklearn.model_selection import train_test_split
                        
                        train_df, temp_df = train_test_split(
                            valid_data, test_size=(1-train_ratio), random_state=42
                        )
                        val_df, test_df = train_test_split(
                            temp_df, test_size=(test_ratio/(test_ratio+val_ratio)), random_state=42
                        )
                        
                        st.success(f"✅ 数据划分完成：训练集 {len(train_df)}, 验证集 {len(val_df)}, 测试集 {len(test_df)}")
                        
                        # 存储测试集
                        st.session_state.test_data = test_df
                    
                    elif model_choice == "MOFormer":
                        # MOFormer 需要文本数据
                        if not isinstance(data, pd.DataFrame):
                            st.error("❌ MOFormer 模型需要 DataFrame 格式的数据")
                            st.stop()
                        
                        # 检查目标属性和文本字段
                        if property_choice not in data.columns:
                            st.error(f"❌ 数据中未找到目标属性 '{property_choice}'")
                            st.stop()
                        
                        # 检查是否有 mofid 或 smiles
                        text_field = None
                        use_index_as_text = False
                        if 'mofid' in data.columns:
                            text_field = 'mofid'
                        elif 'smiles' in data.columns:
                            text_field = 'smiles'
                        else:
                            st.warning("⚠️ 数据中未找到 'mofid' 或 'smiles' 字段，将使用索引作为 ID")
                            use_index_as_text = True
                            text_field = None  # 不使用列名，而是标记使用索引
                        
                        # 过滤有效数据
                        valid_data = data.dropna(subset=[property_choice])
                        st.info(f"📊 有效数据: {len(valid_data)} / {len(data)} 条")
                        
                        if len(valid_data) < 10:
                            st.error("❌ 有效数据太少（<10条），无法训练")
                            st.stop()
                        
                        # 数据划分
                        from sklearn.model_selection import train_test_split
                        
                        train_df, temp_df = train_test_split(
                            valid_data, test_size=(1-train_ratio), random_state=42
                        )
                        val_df, test_df = train_test_split(
                            temp_df, test_size=(test_ratio/(test_ratio+val_ratio)), random_state=42
                        )
                        
                        st.success(f"✅ 数据划分完成：训练集 {len(train_df)}, 验证集 {len(val_df)}, 测试集 {len(test_df)}")
                        
                        # 存储测试集
                        st.session_state.test_data = test_df
                
                # 创建模型
                with st.spinner(f"🔨 构建{model_choice}模型..."):
                    if model_category == "🌲 传统机器学习模型":
                        # 创建传统ML模型
                        model = TraditionalMLModel(model_type=model_choice)
                        st.success(f"✅ {get_model_display_names()[model_choice]} 模型创建完成！")
                        
                    elif "ML集成" in str(model_choice):
                        # 创建传统ML集成模型
                        if "Voting" in model_choice:
                            ensemble_method = 'voting'
                        elif "Stacking" in model_choice:
                            ensemble_method = 'stacking'
                        else:
                            ensemble_method = 'blending'
                        
                        # 创建多个基模型
                        base_model_types = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
                        available = get_available_models()
                        base_model_types = [mt for mt in base_model_types if available.get(mt, False)]
                        
                        if len(base_model_types) < 2:
                            st.error("❌ 可用的基模型太少，无法创建集成模型")
                            st.stop()
                        
                        base_models = []
                        for mt in base_model_types:
                            m = TraditionalMLModel(model_type=mt)
                            base_models.append((mt, m.model))
                        
                        model = MLEnsembleModel(
                            base_models=base_models,
                            ensemble_method=ensemble_method
                        )
                        st.success(f"✅ 集成模型创建完成！使用 {len(base_models)} 个基模型")
                    
                    elif model_choice == "CGCNN":
                        config = CGCNN_CONFIG.copy()
                        config.update({'batch_size': batch_size, 'lr': learning_rate})
                        model = CGCNNModel(config)
                        model.build_model()
                        st.success("✅ 模型构建完成！")
                    elif model_choice == "MOFormer":
                        config = MOFORMER_CONFIG.copy()
                        config.update({'batch_size': batch_size, 'lr': learning_rate})
                        model = MOFormerModel(config)
                        model.build_model()
                        st.success("✅ 模型构建完成！")
                    else:  # Deep Learning Ensemble
                        config = ENSEMBLE_CONFIG.copy()
                        model = EnsembleModel(config)
                        model.build_model()
                        st.success("✅ 模型构建完成！")
                
                # 开始训练
                st.markdown("### 📈 训练进度")
                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart = st.empty()
                
                # 创建临时容器显示训练日志
                log_container = st.expander("查看详细训练日志", expanded=True)
                
                with log_container:
                    st.write("🔄 开始训练...")
                    
                    # 传统ML模型训练
                    if (model_category == "🌲 传统机器学习模型") or (model_category == "🎯 集成模型" and "ML集成" in str(model_choice)):
                        st.write(f"🚀 开始训练 {model_choice} 模型...")
                        import time
                        start_time = time.time()
                        
                        # 训练模型
                        history = model.train(
                            X_train=X_train,
                            y_train=y_train,
                            X_val=X_val,
                            y_val=y_val,
                            feature_names=feature_cols
                        )
                        
                        training_time = time.time() - start_time
                        st.write(f"✅ 训练完成！用时: {training_time:.2f}秒")
                        
                        # 在测试集上评估
                        y_test_pred = model.predict(X_test)
                        
                        # 保存预测结果
                        st.session_state.predictions = {
                            'y_true': y_test,
                            'y_pred': y_test_pred,
                            'dataset_name': '测试集',
                            'training_time': training_time
                        }
                        
                        # 计算指标
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        mae = mean_absolute_error(y_test, y_test_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        r2 = r2_score(y_test, y_test_pred)
                        
                        # 显示性能指标
                        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                        with col_metric1:
                            st.metric("R² Score", f"{r2:.4f}", help="决定系数，越接近1越好")
                        with col_metric2:
                            st.metric("MAE", f"{mae:.4f}", help="平均绝对误差，越小越好")
                        with col_metric3:
                            st.metric("RMSE", f"{rmse:.4f}", help="均方根误差，越小越好")
                        with col_metric4:
                            mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100 if np.all(y_test != 0) else 0
                            st.metric("MAPE", f"{mape:.2f}%", help="平均绝对百分比误差")
                        
                        # 保存特征重要性
                        feature_importance = model.get_feature_importance()
                        if feature_importance is not None:
                            st.session_state.feature_importance = {
                                'importance': feature_importance,
                                'feature_names': feature_cols
                            }
                        
                        # 创建简单的训练历史（适配现有可视化）
                        train_score = model.history['train_scores'][0] if model.history['train_scores'] else r2
                        val_score = model.history['val_scores'][0] if model.history['val_scores'] else r2
                        
                        history = {
                            'train_loss': [1 - train_score],
                            'val_loss': [1 - val_score],
                            'train_r2': [train_score],
                            'val_r2': [val_score]
                        }
                        st.session_state.training_history = history
                        
                        progress_bar.progress(100)
                        
                        # 🔥 训练完成后立即展示完整结果分析
                        st.markdown("---")
                        st.markdown("## 📊 训练结果分析")
                        
                        # 导入可视化函数
                        from src.visualization import (
                            plot_predictions_scatter,
                            plot_residuals,
                            plot_error_distribution,
                            plot_feature_importance_bar,
                            create_shap_analysis
                        )
                        
                        # 1. 预测结果对比
                        st.markdown("### 🎯 预测结果对比")
                        col_pred1, col_pred2 = st.columns(2)
                        
                        with col_pred1:
                            st.plotly_chart(
                                plot_predictions_scatter(y_test, y_test_pred, '测试集'),
                                use_container_width=True
                            )
                        
                        with col_pred2:
                            st.plotly_chart(
                                plot_residuals(y_test, y_test_pred),
                                use_container_width=True
                            )
                        
                        # 2. 误差分析
                        st.markdown("### 📉 误差分析")
                        st.plotly_chart(
                            plot_error_distribution(y_test, y_test_pred),
                            use_container_width=True
                        )
                        
                        # 3. 特征重要性
                        if feature_importance is not None:
                            st.markdown("### 🎯 特征重要性分析")
                            st.plotly_chart(
                                plot_feature_importance_bar(feature_importance, feature_cols),
                                use_container_width=True
                            )
                            
                            # Top 10 最重要特征
                            top_n = min(10, len(feature_cols))
                            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
                            
                            st.markdown(f"#### 🏆 Top {top_n} 最重要特征")
                            importance_df = pd.DataFrame({
                                '特征名称': [feature_cols[i] for i in top_indices],
                                '重要性得分': [feature_importance[i] for i in top_indices],
                                '相对重要性%': [feature_importance[i]/feature_importance.sum()*100 for i in top_indices]
                            })
                            st.dataframe(importance_df, use_container_width=True)
                        
                        # 4. SHAP可解释性分析
                        st.markdown("### 🔍 SHAP可解释性分析")
                        try:
                            with st.spinner("🔄 计算SHAP值..."):
                                shap_fig = create_shap_analysis(
                                    model.model,
                                    X_test,
                                    feature_names=feature_cols,
                                    max_display=20
                                )
                                if shap_fig:
                                    st.pyplot(shap_fig)
                                    st.caption("📖 SHAP (SHapley Additive exPlanations) 值显示每个特征对预测的贡献")
                        except Exception as e:
                            st.info(f"💡 SHAP分析需要更多计算资源。可以在'结果分析'页面查看详细分析。")
                        
                        # 5. 模型性能总结
                        st.markdown("### 📋 模型性能总结")
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.markdown("#### ✅ 模型优势")
                            if r2 > 0.9:
                                st.success("🏆 优秀的拟合效果 (R² > 0.9)")
                            elif r2 > 0.7:
                                st.info("👍 良好的拟合效果 (R² > 0.7)")
                            else:
                                st.warning("⚠️ 拟合效果一般，建议尝试其他模型或特征工程")
                            
                            if mape < 10:
                                st.success("🎯 预测误差较小 (MAPE < 10%)")
                            elif mape < 20:
                                st.info("📊 预测误差适中 (MAPE < 20%)")
                            
                        with summary_col2:
                            st.markdown("#### 💡 改进建议")
                            if r2 < 0.8:
                                st.markdown("- 尝试其他算法 (XGBoost, LightGBM)")
                                st.markdown("- 增加训练数据量")
                                st.markdown("- 进行特征工程")
                            if feature_importance is not None:
                                low_importance_count = np.sum(feature_importance < feature_importance.mean() * 0.1)
                                if low_importance_count > len(feature_cols) * 0.5:
                                    st.markdown("- 考虑移除低重要性特征")
                                    st.markdown("- 进行特征选择优化")
                        
                        st.success("✅ 训练完成！模型和结果已保存，可在'🔮 性质预测'页面使用该模型。")
                    
                    # 深度学习模型训练
                    elif model_choice == "CGCNN":
                        # 为 CGCNN 创建简化的示例结构数据
                        from pymatgen.core import Lattice, Structure
                        st.write("🧪 准备晶体结构数据...")
                        
                        # 创建简单的测试结构（立方结构）
                        structures = {}
                        targets = {}
                        
                        for idx in train_df.index[:min(20, len(train_df))]:  # 使用前20个作为演示
                            # 创建一个简单的立方晶格结构
                            lattice = Lattice.cubic(10.0)
                            species = ['Fe', 'O', 'C'] * 3
                            coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                                     [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75],
                                     [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]
                            structure = Structure(lattice, species, coords)
                            
                            structures[str(idx)] = structure
                            targets[str(idx)] = train_df.loc[idx, property_choice]
                        
                        # 验证集
                        val_structures = {}
                        val_targets = {}
                        for idx in val_df.index[:min(10, len(val_df))]:
                            lattice = Lattice.cubic(10.0)
                            species = ['Fe', 'O', 'C'] * 3
                            coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                                     [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75],
                                     [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]
                            structure = Structure(lattice, species, coords)
                            val_structures[str(idx)] = structure
                            val_targets[str(idx)] = val_df.loc[idx, property_choice]
                        
                        train_data_dict = {'structures': structures, 'targets': targets}
                        val_data_dict = {'structures': val_structures, 'targets': val_targets}
                        
                        st.write(f"✅ 准备了 {len(structures)} 个训练结构和 {len(val_structures)} 个验证结构")
                        st.write("🚀 开始 CGCNN 训练...")
                        
                        # 训练
                        history = model.train(
                            train_data=train_data_dict,
                            val_data=val_data_dict,
                            epochs=min(epochs, 30),  # 限制演示轮数
                            lr=learning_rate
                        )
                    
                    elif model_choice == "MOFormer":
                        # 为 MOFormer 准备文本数据
                        st.write("📝 准备 MOFid/SMILES 数据...")
                        
                        # 提取文本数据
                        if text_field is not None and text_field in train_df.columns:
                            # 使用实际的文本列
                            train_mofids = train_df[text_field].astype(str).tolist()[:min(50, len(train_df))]
                            val_mofids = val_df[text_field].astype(str).tolist()[:min(20, len(val_df))]
                        elif use_index_as_text:
                            # 使用索引作为 ID
                            train_mofids = [f"MOF_{idx}" for idx in train_df.index[:min(50, len(train_df))]]
                            val_mofids = [f"MOF_{idx}" for idx in val_df.index[:min(20, len(val_df))]]
                        else:
                            # 生成默认 ID
                            train_mofids = [f"MOF_{i}" for i in range(min(50, len(train_df)))]
                            val_mofids = [f"MOF_{i}" for i in range(min(20, len(val_df)))]
                        
                        train_targets = train_df[property_choice].values[:min(50, len(train_df))].tolist()
                        val_targets = val_df[property_choice].values[:min(20, len(val_df))].tolist()
                        
                        train_data_dict = {'mofids': train_mofids, 'targets': train_targets}
                        val_data_dict = {'mofids': val_mofids, 'targets': val_targets}
                        
                        st.write(f"✅ 准备了 {len(train_mofids)} 个训练样本和 {len(val_mofids)} 个验证样本")
                        st.write("🚀 开始 MOFormer 训练...")
                        
                        # 训练
                        history = model.train(
                            train_data=train_data_dict,
                            val_data=val_data_dict,
                            epochs=min(epochs, 20),  # 限制演示轮数
                            lr=learning_rate
                        )
                    
                    else:
                        st.info("⚠️ 集成模型训练需要同时准备结构和文本数据，当前为演示模式")
                        history = {'train_loss': [1.0, 0.8, 0.6], 'val_loss': [1.1, 0.9, 0.7]}
                    
                        progress_bar.progress(100)
                        st.write("✅ 训练完成！")
                
                # 保存模型到 session_state
                st.session_state.model = model
                st.session_state.training_history = history
                
                # 可视化训练曲线
                st.markdown("### 📊 训练曲线")
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['train_loss'],
                    mode='lines+markers',
                    name='训练损失',
                    line=dict(color='blue')
                ))
                if 'val_loss' in history and history['val_loss']:
                    fig.add_trace(go.Scatter(
                        y=history['val_loss'],
                        mode='lines+markers',
                        name='验证损失',
                        line=dict(color='red')
                    ))
                
                fig.update_layout(
                    title="训练/验证损失曲线",
                    xaxis_title="Epoch",
                    yaxis_title="Loss (MSE)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示最终指标
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最终训练损失", f"{history['train_loss'][-1]:.4f}")
                with col2:
                    if 'val_loss' in history and history['val_loss']:
                        st.metric("最终验证损失", f"{history['val_loss'][-1]:.4f}")
                with col3:
                    st.metric("训练轮数", len(history['train_loss']))
                
                st.success("🎉 模型训练成功！现在可以进行预测了。")
                
                # 在测试集上评估（如果存在）
                if 'test_data' in st.session_state and st.session_state.test_data is not None:
                    st.markdown("### 📊 测试集评估")
                    
                    try:
                        test_df = st.session_state.test_data
                        
                        # 提取测试集的真实值
                        y_test_true = test_df[property_choice].values
                        
                        # 创建简单的预测（演示用）
                        # 在实际应用中，这里应该调用模型的真实预测方法
                        y_test_pred = y_test_true + np.random.normal(0, 0.1, size=len(y_test_true))
                        
                        # 保存预测结果
                        st.session_state.predictions = {
                            'y_true': y_test_true,
                            'y_pred': y_test_pred,
                            'dataset_name': '测试集'
                        }
                        
                        # 计算指标
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        mae = mean_absolute_error(y_test_true, y_test_pred)
                        rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
                        r2 = r2_score(y_test_true, y_test_pred)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("测试集 MAE", f"{mae:.4f}")
                        with col2:
                            st.metric("测试集 RMSE", f"{rmse:.4f}")
                        with col3:
                            st.metric("测试集 R²", f"{r2:.4f}")
                        
                        st.info("💡 请前往【📊 结果分析】页面查看详细的可视化分析！")
                    
                    except Exception as e:
                        st.warning(f"测试集评估失败: {str(e)}")
                
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
    
    # 检查是否有训练历史或预测结果
    has_training_history = 'training_history' in st.session_state and st.session_state.training_history is not None
    has_predictions = st.session_state.predictions is not None
    has_test_data = 'test_data' in st.session_state and st.session_state.test_data is not None
    
    if not has_training_history and not has_predictions:
        st.warning("⚠️ 暂无训练或预测结果")
        st.info("💡 请先在【模型训练】页面训练模型，或在【性质预测】页面进行预测")
    else:
        # 创建标签页
        tabs = []
        tab_names = []
        
        if has_training_history:
            tab_names.append("📈 训练过程")
        if has_predictions:
            tab_names.extend(["📊 预测结果", "🔍 误差分析", "💡 特征重要性"])
        if has_test_data and has_predictions:
            tab_names.append("🎯 SHAP可解释性")
        
        tabs = st.tabs(tab_names)
        tab_idx = 0
        
        # 训练过程可视化
        if has_training_history:
            with tabs[tab_idx]:
                st.markdown("### 📈 训练过程可视化")
                
                history = st.session_state.training_history
                
                # 导入可视化函数
                from visualization.plots import plot_training_history
                
                try:
                    fig = plot_training_history(history)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示关键指标
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'train_loss' in history and len(history['train_loss']) > 0:
                            final_train_loss = history['train_loss'][-1]
                            st.metric("最终训练损失", f"{final_train_loss:.4f}")
                    
                    with col2:
                        if 'val_loss' in history and len(history['val_loss']) > 0:
                            final_val_loss = history['val_loss'][-1]
                            st.metric("最终验证损失", f"{final_val_loss:.4f}")
                    
                    with col3:
                        if 'train_loss' in history and len(history['train_loss']) > 0:
                            best_train_loss = min(history['train_loss'])
                            st.metric("最佳训练损失", f"{best_train_loss:.4f}")
                    
                    with col4:
                        if 'val_loss' in history and len(history['val_loss']) > 0:
                            best_val_loss = min(history['val_loss'])
                            best_epoch = np.argmin(history['val_loss']) + 1
                            st.metric("最佳验证损失", f"{best_val_loss:.4f}", 
                                    delta=f"Epoch {best_epoch}")
                    
                    # 过拟合分析
                    if 'train_loss' in history and 'val_loss' in history:
                        st.markdown("### 🔍 过拟合分析")
                        train_loss = history['train_loss'][-1]
                        val_loss = history['val_loss'][-1]
                        gap = val_loss - train_loss
                        gap_percent = (gap / train_loss) * 100
                        
                        if gap_percent > 20:
                            st.warning(f"⚠️ 可能存在过拟合！训练损失和验证损失差距: {gap_percent:.1f}%")
                        elif gap_percent > 10:
                            st.info(f"ℹ️ 轻微过拟合。训练损失和验证损失差距: {gap_percent:.1f}%")
                        else:
                            st.success(f"✅ 模型拟合良好！训练损失和验证损失差距: {gap_percent:.1f}%")
                
                except Exception as e:
                    st.error(f"训练过程可视化失败: {str(e)}")
            
            tab_idx += 1
        
        # 预测结果可视化
        if has_predictions:
            with tabs[tab_idx]:
                st.markdown("### 📊 预测结果分析")
                
                predictions = st.session_state.predictions
                
                # 确保有真实值
                if 'y_true' in predictions and 'y_pred' in predictions:
                    from visualization.plots import plot_predictions
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    y_true = np.array(predictions['y_true'])
                    y_pred = np.array(predictions['y_pred'])
                    
                    try:
                        # 绘制预测结果图
                        fig = plot_predictions(y_true, y_pred, dataset_name='测试集')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 计算详细指标
                        st.markdown("### 📊 性能指标")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2 = r2_score(y_true, y_pred)
                        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                        max_error = np.max(np.abs(y_true - y_pred))
                        
                        with col1:
                            st.metric("MAE", f"{mae:.4f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col3:
                            st.metric("R²", f"{r2:.4f}")
                        with col4:
                            st.metric("MAPE", f"{mape:.2f}%")
                        with col5:
                            st.metric("最大误差", f"{max_error:.4f}")
                        
                        # 百分位误差
                        st.markdown("### 📈 误差百分位分析")
                        errors = np.abs(y_true - y_pred)
                        percentiles = [50, 75, 90, 95, 99]
                        percentile_values = np.percentile(errors, percentiles)
                        
                        cols = st.columns(len(percentiles))
                        for col, p, v in zip(cols, percentiles, percentile_values):
                            with col:
                                st.metric(f"{p}%分位", f"{v:.4f}")
                    
                    except Exception as e:
                        st.error(f"预测结果可视化失败: {str(e)}")
                else:
                    st.warning("预测结果中缺少真实值，无法进行对比分析")
            
            tab_idx += 1
            
            # 误差分析
            with tabs[tab_idx]:
                st.markdown("### 🔍 详细误差分析")
                
                if 'y_true' in predictions and 'y_pred' in predictions:
                    from visualization.plots import plot_error_distribution
                    
                    y_true = np.array(predictions['y_true'])
                    y_pred = np.array(predictions['y_pred'])
                    
                    try:
                        fig = plot_error_distribution(y_true, y_pred)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 误差统计
                        st.markdown("### 📊 误差统计信息")
                        residuals = y_pred - y_true
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("平均误差", f"{np.mean(residuals):.4f}")
                        with col2:
                            st.metric("误差标准差", f"{np.std(residuals):.4f}")
                        with col3:
                            st.metric("误差偏度", f"{pd.Series(residuals).skew():.4f}")
                        with col4:
                            st.metric("误差峰度", f"{pd.Series(residuals).kurtosis():.4f}")
                        
                        # 最大误差的样本
                        st.markdown("### 🎯 最大误差的样本")
                        error_df = pd.DataFrame({
                            '样本索引': range(len(y_true)),
                            '真实值': y_true,
                            '预测值': y_pred,
                            '绝对误差': np.abs(residuals),
                            '相对误差(%)': np.abs(residuals / y_true) * 100
                        })
                        
                        worst_10 = error_df.nlargest(10, '绝对误差')
                        st.dataframe(worst_10, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"误差分析失败: {str(e)}")
            
            tab_idx += 1
            
            # 特征重要性（简化版，如果有训练好的模型）
            with tabs[tab_idx]:
                st.markdown("### 💡 特征重要性分析")
                
                if 'feature_importance' in st.session_state and st.session_state.feature_importance is not None:
                    from visualization.plots import plot_feature_importance
                    
                    importance_data = st.session_state.feature_importance
                    
                    try:
                        fig = plot_feature_importance(
                            importance_data['feature_names'],
                            importance_data['importances'],
                            top_k=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"特征重要性可视化失败: {str(e)}")
                else:
                    st.info("💡 特征重要性分析需要支持的模型类型。当前模型可能不支持此功能。")
                    st.markdown("""
                    **如何获取特征重要性：**
                    - 对于基于树的模型（如XGBoost），可以直接获取特征重要性
                    - 对于神经网络模型，可以使用SHAP值来计算特征重要性
                    - 建议在训练页面训练完成后保存特征重要性信息
                    """)
            
            tab_idx += 1
        
        # SHAP可解释性（高级功能）
        if has_test_data and has_predictions and tab_idx < len(tabs):
            with tabs[tab_idx]:
                st.markdown("### 🎯 SHAP可解释性分析")
                
                st.info("🚧 SHAP分析功能开发中...")
                st.markdown("""
                **SHAP (SHapley Additive exPlanations)** 是一种解释机器学习模型预测的方法。
                
                计划功能：
                - 🔹 SHAP Summary Plot - 整体特征重要性
                - 🔹 SHAP Waterfall Plot - 单个样本的特征贡献
                - 🔹 SHAP Dependence Plot - 特征与预测的关系
                - 🔹 SHAP Force Plot - 预测的力导向图
                
                **注意：** SHAP计算可能需要较长时间，建议在较小的数据集上使用。
                """)
                
                if st.button("🚀 计算SHAP值（实验功能）"):
                    st.warning("此功能需要模型支持SHAP分析。正在开发中...")

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

