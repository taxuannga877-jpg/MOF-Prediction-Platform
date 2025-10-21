"""
数据处理器
Data Processor Module
"""

from typing import Union, List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """MOF数据处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据处理器
        
        Args:
            config: 处理配置
        """
        self.config = config or {}
        self.scaler = None
        self.feature_names = []
    
    def extract_features_from_mofid(
        self,
        data: Union[List[Dict], pd.DataFrame],
        target_property: str = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        从MOFid数据中提取特征
        
        Args:
            data: MOFid数据
            target_property: 目标属性名称
            
        Returns:
            (特征DataFrame, 目标Series)元组
        """
        if isinstance(data, list):
            df = pd.json_normalize(data)
        else:
            df = data.copy()
        
        # 提取MOFid相关特征
        feature_columns = []
        
        # 基础信息特征
        if 'info.natoms' in df.columns:
            feature_columns.append('info.natoms')
        if 'info.pld' in df.columns:
            feature_columns.append('info.pld')
        if 'info.lcd' in df.columns:
            feature_columns.append('info.lcd')
        if 'info.density' in df.columns:
            feature_columns.append('info.density')
        
        # MOFid文本特征（需要进一步处理）
        mofid_cols = [col for col in df.columns if 'mofid' in col.lower()]
        
        # 提取数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_columns.extend([col for col in numeric_cols if col not in feature_columns])
        
        # 过滤掉目标列
        if target_property:
            feature_columns = [col for col in feature_columns if col != target_property]
        
        X = df[feature_columns] if feature_columns else df
        y = df[target_property] if target_property and target_property in df.columns else None
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def extract_property_from_qmof(
        self,
        data: Union[List[Dict], pd.DataFrame],
        property_name: str,
        theory_level: str = 'pbe'
    ) -> Tuple[List[str], List[float]]:
        """
        从QMOF数据中提取指定属性
        
        Args:
            data: QMOF数据
            property_name: 属性名称（如'bandgap'）
            theory_level: 理论水平（'pbe', 'hle17', 'hse06'等）
            
        Returns:
            (MOF ID列表, 属性值列表)元组
        """
        ids = []
        values = []
        
        if isinstance(data, list):
            for entry in data:
                mof_id = entry.get('qmof_id', entry.get('name', 'unknown'))
                
                # 尝试从outputs中提取
                try:
                    if 'outputs' in entry and theory_level in entry['outputs']:
                        value = entry['outputs'][theory_level].get(property_name)
                        if value is not None:
                            ids.append(mof_id)
                            values.append(value)
                    # 尝试直接从info中提取（如pld, lcd等）
                    elif 'info' in entry:
                        value = entry['info'].get(property_name)
                        if value is not None:
                            ids.append(mof_id)
                            values.append(value)
                except Exception:
                    continue
        
        elif isinstance(data, pd.DataFrame):
            # CSV格式
            col_name = f'outputs.{theory_level}.{property_name}'
            if col_name in data.columns:
                mask = data[col_name].notna()
                ids = data.loc[mask, 'qmof_id'].tolist()
                values = data.loc[mask, col_name].tolist()
            elif property_name in data.columns:
                mask = data[property_name].notna()
                ids = data.loc[mask, 'qmof_id'].tolist() if 'qmof_id' in data.columns else data.index.tolist()
                values = data.loc[mask, property_name].tolist()
        
        return ids, values
    
    def normalize_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        method: str = 'standard',
        fit: bool = True
    ) -> np.ndarray:
        """
        标准化特征
        
        Args:
            X: 特征矩阵
            method: 标准化方法（'standard', 'minmax'）
            fit: 是否拟合scaler
            
        Returns:
            标准化后的特征矩阵
        """
        if method == 'standard':
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                X_normalized = self.scaler.fit_transform(X)
            else:
                X_normalized = self.scaler.transform(X)
        
        elif method == 'minmax':
            if fit or self.scaler is None:
                self.scaler = MinMaxScaler()
                X_normalized = self.scaler.fit_transform(X)
            else:
                X_normalized = self.scaler.transform(X)
        
        else:
            X_normalized = X
        
        return X_normalized
    
    def split_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分训练集、验证集和测试集
        
        Args:
            X: 特征矩阵
            y: 目标向量
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)元组
        """
        # 首先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_state
        )
        
        # 从剩余数据中分出验证集
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def remove_outliers(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        std_threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        移除异常值
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            std_threshold: 标准差阈值
            
        Returns:
            (过滤后的X, 过滤后的y)元组
        """
        # 计算Z-score
        z_scores = np.abs((X - X.mean()) / X.std())
        
        # 找出所有特征都在阈值内的样本
        mask = (z_scores < std_threshold).all(axis=1)
        
        X_filtered = X[mask]
        y_filtered = y[mask] if y is not None else None
        
        return X_filtered, y_filtered
    
    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            X: 特征DataFrame
            strategy: 填充策略（'mean', 'median', 'drop'）
            
        Returns:
            处理后的DataFrame
        """
        if strategy == 'drop':
            return X.dropna()
        
        elif strategy == 'mean':
            return X.fillna(X.mean())
        
        elif strategy == 'median':
            return X.fillna(X.median())
        
        else:
            return X
    
    def get_statistics(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Args:
            data: 数据
            
        Returns:
            统计信息字典
        """
        if isinstance(data, pd.DataFrame):
            stats = {
                'count': len(data),
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'missing': data.isnull().sum().to_dict(),
                'numeric_summary': data.describe().to_dict(),
            }
        
        elif isinstance(data, np.ndarray):
            stats = {
                'count': data.shape[0],
                'shape': data.shape,
                'mean': np.mean(data, axis=0).tolist() if data.ndim > 1 else float(np.mean(data)),
                'std': np.std(data, axis=0).tolist() if data.ndim > 1 else float(np.std(data)),
                'min': np.min(data, axis=0).tolist() if data.ndim > 1 else float(np.min(data)),
                'max': np.max(data, axis=0).tolist() if data.ndim > 1 else float(np.max(data)),
            }
        
        else:
            stats = {'error': '不支持的数据类型'}
        
        return stats
    
    def prepare_for_cgcnn(
        self,
        structures: Dict[str, Any],
        properties: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        为CGCNN准备数据
        
        Args:
            structures: {mof_id: Structure}字典
            properties: {mof_id: property_value}字典
            
        Returns:
            CGCNN格式的数据列表
        """
        data = []
        
        for mof_id, structure in structures.items():
            if mof_id in properties:
                data.append({
                    'id': mof_id,
                    'structure': structure,
                    'target': properties[mof_id],
                })
        
        return data
    
    def prepare_for_moformer(
        self,
        mofid_data: Union[List[Dict], pd.DataFrame],
        target_property: str
    ) -> Tuple[List[str], List[float]]:
        """
        为MOFormer准备数据
        
        Args:
            mofid_data: MOFid数据
            target_property: 目标属性
            
        Returns:
            (MOFid列表, 目标值列表)元组
        """
        mofids = []
        targets = []
        
        if isinstance(mofid_data, list):
            for entry in mofid_data:
                # 提取MOFid
                mofid = None
                if 'info' in entry and 'mofid' in entry.get('info', {}):
                    mofid = entry['info']['mofid'].get('mofid')
                elif 'mofid' in entry:
                    mofid = entry['mofid']
                
                # 提取目标值
                target = None
                if 'outputs' in entry:
                    for theory in ['pbe', 'hle17', 'hse06']:
                        if theory in entry['outputs']:
                            target = entry['outputs'][theory].get(target_property)
                            if target is not None:
                                break
                
                if mofid and target is not None:
                    mofids.append(mofid)
                    targets.append(target)
        
        elif isinstance(mofid_data, pd.DataFrame):
            mofid_col = 'info.mofid.mofid' if 'info.mofid.mofid' in mofid_data.columns else 'mofid'
            target_col = f'outputs.pbe.{target_property}' if f'outputs.pbe.{target_property}' in mofid_data.columns else target_property
            
            if mofid_col in mofid_data.columns and target_col in mofid_data.columns:
                mask = mofid_data[target_col].notna() & mofid_data[mofid_col].notna()
                mofids = mofid_data.loc[mask, mofid_col].tolist()
                targets = mofid_data.loc[mask, target_col].tolist()
        
        return mofids, targets


