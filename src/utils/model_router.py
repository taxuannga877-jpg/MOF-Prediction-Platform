"""
模型路由器 - 智能选择合适的模型
Model Router - Intelligent Model Selection
"""

from typing import Union, Dict, Any, List
from pathlib import Path
import pandas as pd


class ModelRouter:
    """
    智能模型路由器
    根据输入数据格式自动选择最优模型
    """
    
    def __init__(self):
        self.routing_rules = {
            'cif': 'cgcnn',
            'json_structure': 'cgcnn',
            'json_mofid': 'moformer',
            'csv': 'moformer',
            'mixed': 'ensemble',
        }
    
    def route(self, data_type: str, data_info: Dict[str, Any] = None) -> str:
        """
        路由到合适的模型
        
        Args:
            data_type: 数据类型
            data_info: 数据附加信息
            
        Returns:
            推荐的模型名称: 'cgcnn', 'moformer', 或 'ensemble'
        """
        if data_type == 'cif':
            return 'cgcnn'
        
        elif data_type == 'json':
            # 检查JSON内容类型
            if data_info:
                if self._has_structure_data(data_info):
                    return 'cgcnn'
                elif self._has_mofid_data(data_info):
                    return 'moformer'
            return 'moformer'  # 默认
        
        elif data_type == 'csv':
            return 'moformer'
        
        else:
            return 'ensemble'  # 未知类型使用集成模型
    
    def _has_structure_data(self, data: Dict[str, Any]) -> bool:
        """检查是否包含结构数据"""
        # 检查是否有structure字段
        if isinstance(data, dict):
            if 'structure' in data:
                return True
            if 'lattice' in data and 'sites' in data:
                return True
        
        # 检查列表第一个元素
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                if 'structure' in first_item:
                    return True
        
        return False
    
    def _has_mofid_data(self, data: Dict[str, Any]) -> bool:
        """检查是否包含MOFid数据"""
        # 检查是否有mofid相关字段
        mofid_keys = ['mofid', 'mof_id', 'smiles', 'smiles_nodes', 'smiles_linkers']
        
        if isinstance(data, dict):
            for key in mofid_keys:
                if key in data:
                    return True
            
            # 检查info.mofid结构
            if 'info' in data and isinstance(data['info'], dict):
                if 'mofid' in data['info']:
                    return True
        
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                for key in mofid_keys:
                    if key in first_item:
                        return True
                
                if 'info' in first_item and isinstance(first_item['info'], dict):
                    if 'mofid' in first_item['info']:
                        return True
        
        return False
    
    def recommend_model(
        self,
        file_path: Union[str, Path] = None,
        data: Any = None,
        user_preference: str = None
    ) -> Dict[str, Any]:
        """
        推荐模型（综合考虑多种因素）
        
        Args:
            file_path: 文件路径
            data: 数据对象
            user_preference: 用户偏好
            
        Returns:
            推荐结果字典
        """
        recommendations = {
            'primary': None,
            'alternatives': [],
            'reason': '',
            'confidence': 0.0,
        }
        
        # 用户明确指定
        if user_preference in ['cgcnn', 'moformer', 'ensemble']:
            recommendations['primary'] = user_preference
            recommendations['reason'] = '用户指定'
            recommendations['confidence'] = 1.0
            return recommendations
        
        # 基于文件路径
        if file_path:
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()
            
            if suffix == '.cif':
                recommendations['primary'] = 'cgcnn'
                recommendations['alternatives'] = ['ensemble']
                recommendations['reason'] = 'CIF文件适合CGCNN（基于3D结构）'
                recommendations['confidence'] = 0.95
                
            elif suffix == '.json':
                # 需要检查内容
                if data:
                    if self._has_structure_data(data):
                        recommendations['primary'] = 'cgcnn'
                        recommendations['alternatives'] = ['ensemble']
                        recommendations['reason'] = 'JSON包含3D结构数据'
                        recommendations['confidence'] = 0.90
                    elif self._has_mofid_data(data):
                        recommendations['primary'] = 'moformer'
                        recommendations['alternatives'] = ['ensemble']
                        recommendations['reason'] = 'JSON包含MOFid文本数据'
                        recommendations['confidence'] = 0.90
                    else:
                        recommendations['primary'] = 'moformer'
                        recommendations['alternatives'] = ['cgcnn', 'ensemble']
                        recommendations['reason'] = 'JSON格式，默认使用MOFormer'
                        recommendations['confidence'] = 0.60
                else:
                    recommendations['primary'] = 'moformer'
                    recommendations['alternatives'] = ['cgcnn', 'ensemble']
                    recommendations['reason'] = 'JSON格式（未检查内容）'
                    recommendations['confidence'] = 0.50
            
            elif suffix == '.csv':
                recommendations['primary'] = 'moformer'
                recommendations['alternatives'] = ['ensemble']
                recommendations['reason'] = 'CSV表格数据适合MOFormer'
                recommendations['confidence'] = 0.85
        
        # 基于数据内容
        elif data:
            if isinstance(data, pd.DataFrame):
                # 检查是否有MOFid列
                mofid_columns = ['mofid', 'mof_id', 'smiles']
                has_mofid = any(col in data.columns for col in mofid_columns)
                
                if has_mofid:
                    recommendations['primary'] = 'moformer'
                    recommendations['reason'] = '表格包含MOFid信息'
                    recommendations['confidence'] = 0.85
                else:
                    recommendations['primary'] = 'moformer'
                    recommendations['reason'] = '表格数据'
                    recommendations['confidence'] = 0.70
                
                recommendations['alternatives'] = ['ensemble']
            
            elif isinstance(data, (dict, list)):
                if self._has_structure_data(data):
                    recommendations['primary'] = 'cgcnn'
                    recommendations['reason'] = '包含3D结构数据'
                    recommendations['confidence'] = 0.90
                elif self._has_mofid_data(data):
                    recommendations['primary'] = 'moformer'
                    recommendations['reason'] = '包含MOFid数据'
                    recommendations['confidence'] = 0.90
                else:
                    recommendations['primary'] = 'ensemble'
                    recommendations['reason'] = '数据类型不明确，使用集成模型'
                    recommendations['confidence'] = 0.50
                
                recommendations['alternatives'] = ['cgcnn', 'moformer']
        
        # 默认情况
        if recommendations['primary'] is None:
            recommendations['primary'] = 'ensemble'
            recommendations['alternatives'] = ['cgcnn', 'moformer']
            recommendations['reason'] = '无明确数据类型，使用集成模型'
            recommendations['confidence'] = 0.40
        
        return recommendations
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典
        """
        model_info = {
            'cgcnn': {
                'name': 'CGCNN',
                'full_name': 'Crystal Graph Convolutional Neural Network',
                'description': '基于3D晶体结构的图卷积神经网络',
                'input': 'CIF文件或晶体结构',
                'advantages': ['高精度', '物理意义明确', '适合精确预测'],
                'disadvantages': ['需要3D结构', '计算较慢', '内存占用高'],
                'best_for': ['已知结构的MOF', '高精度要求', '小规模预测'],
            },
            'moformer': {
                'name': 'MOFormer',
                'full_name': 'MOF Transformer',
                'description': '基于MOFid文本表示的Transformer模型',
                'input': 'MOFid文本字符串',
                'advantages': ['速度快', '不需要3D结构', '适合大规模筛选'],
                'disadvantages': ['精度略低', '需要MOFid表示'],
                'best_for': ['快速筛选', '大规模预测', '假设MOF'],
            },
            'ensemble': {
                'name': 'Ensemble',
                'full_name': '集成模型',
                'description': '结合CGCNN和MOFormer的集成模型',
                'input': '支持多种格式',
                'advantages': ['最高精度', '鲁棒性强', '综合两种方法优势'],
                'disadvantages': ['计算时间最长', '需要两种数据格式'],
                'best_for': ['最高精度需求', '重要决策', '有充足时间'],
            },
        }
        
        return model_info.get(model_name, {})
    
    def compare_models(self) -> pd.DataFrame:
        """
        比较不同模型
        
        Returns:
            模型比较表
        """
        comparison_data = {
            '模型': ['CGCNN', 'MOFormer', 'Ensemble'],
            '输入格式': ['CIF/结构', 'MOFid/文本', '多种'],
            '预测精度': ['高', '中', '最高'],
            '计算速度': ['慢', '快', '中'],
            '内存占用': ['高', '低', '高'],
            '适用场景': ['精确预测', '快速筛选', '重要决策'],
        }
        
        return pd.DataFrame(comparison_data)


