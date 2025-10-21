"""
数据加载器
Data Loader Module
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pymatgen.core import Structure

from .file_handler import FileHandler


class DataLoader:
    """MOF数据加载器"""
    
    def __init__(self, qmof_config: Dict[str, str] = None):
        """
        初始化数据加载器
        
        Args:
            qmof_config: QMOF数据路径配置
        """
        self.qmof_config = qmof_config or {}
        self.file_handler = FileHandler()
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        limit: Optional[int] = None
    ) -> Tuple[Any, str]:
        """
        从文件加载数据
        
        Args:
            file_path: 文件路径
            limit: 限制加载数量
            
        Returns:
            (数据, 数据类型)元组
        """
        file_path = Path(file_path)
        file_type = self.file_handler.detect_file_type(file_path)
        
        if file_type == 'cif':
            data = self._load_cif(file_path)
            return data, 'structure'
        
        elif file_type == 'json':
            data = self._load_json(file_path, limit)
            # 判断JSON类型
            data_subtype = self._detect_json_type(data)
            return data, data_subtype
        
        elif file_type == 'csv':
            data = self._load_csv(file_path, limit)
            return data, 'tabular'
        
        elif file_type == 'zip':
            # 假设是CIF文件的压缩包
            data = self._load_cif_zip(file_path, limit)
            return data, 'structure_batch'
        
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    
    def load_qmof_data(
        self,
        data_source: str = 'json',
        limit: Optional[int] = None
    ) -> Tuple[Any, str]:
        """
        加载QMOF数据集
        
        Args:
            data_source: 数据源类型 ('json', 'csv', 'structures', 'cif')
            limit: 限制加载数量
            
        Returns:
            (数据, 数据类型)元组
        """
        if data_source == 'json':
            file_path = self.qmof_config.get('qmof_json')
            if not file_path or not Path(file_path).exists():
                raise FileNotFoundError("QMOF JSON文件不存在，请检查配置")
            return self._load_json(file_path, limit), 'qmof_properties'
        
        elif data_source == 'csv':
            file_path = self.qmof_config.get('qmof_csv')
            if not file_path or not Path(file_path).exists():
                raise FileNotFoundError("QMOF CSV文件不存在，请检查配置")
            return self._load_csv(file_path, limit), 'qmof_properties'
        
        elif data_source == 'structures':
            file_path = self.qmof_config.get('qmof_structure_data')
            if not file_path or not Path(file_path).exists():
                raise FileNotFoundError("QMOF结构数据文件不存在，请检查配置")
            return self._load_structure_json(file_path, limit), 'qmof_structures'
        
        elif data_source == 'cif':
            # 检查是否已解压
            cif_dir = self.qmof_config.get('relaxed_structures_dir')
            if cif_dir and Path(cif_dir).exists():
                return self._load_cif_directory(cif_dir, limit), 'structure_batch'
            
            # 尝试解压ZIP
            zip_path = self.qmof_config.get('relaxed_structures_zip')
            if zip_path and Path(zip_path).exists():
                extract_dir = Path(zip_path).parent / 'relaxed_structures'
                if not extract_dir.exists():
                    self.file_handler.extract_zip(zip_path, extract_dir.parent)
                return self._load_cif_directory(extract_dir, limit), 'structure_batch'
            
            raise FileNotFoundError("QMOF CIF文件不存在")
        
        else:
            raise ValueError(f"不支持的数据源类型: {data_source}")
    
    def _load_cif(self, file_path: Path) -> Structure:
        """加载单个CIF文件"""
        return self.file_handler.read_cif(file_path)
    
    def _load_json(self, file_path: Path, limit: Optional[int] = None) -> Any:
        """加载JSON文件"""
        data = self.file_handler.read_json(file_path)
        
        if isinstance(data, list) and limit:
            data = data[:limit]
        
        return data
    
    def _load_csv(self, file_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
        """加载CSV文件"""
        df = self.file_handler.read_csv(file_path)
        
        if limit:
            df = df.head(limit)
        
        return df
    
    def _load_cif_zip(self, zip_path: Path, limit: Optional[int] = None) -> List[Structure]:
        """加载CIF压缩包"""
        # 解压到临时目录
        extract_dir = zip_path.parent / 'temp_extracted'
        self.file_handler.extract_zip(zip_path, extract_dir)
        
        # 加载CIF文件
        structures = self._load_cif_directory(extract_dir, limit)
        
        return structures
    
    def _load_cif_directory(
        self,
        directory: Path,
        limit: Optional[int] = None
    ) -> Dict[str, Structure]:
        """加载目录中的所有CIF文件"""
        cif_files = list(Path(directory).rglob('*.cif'))
        
        if limit:
            cif_files = cif_files[:limit]
        
        structures = {}
        for cif_file in cif_files:
            try:
                mof_id = cif_file.stem  # 文件名作为ID
                structure = self.file_handler.read_cif(cif_file)
                structures[mof_id] = structure
            except Exception as e:
                print(f"警告: 无法加载 {cif_file}: {str(e)}")
        
        return structures
    
    def _load_structure_json(
        self,
        file_path: Path,
        limit: Optional[int] = None
    ) -> Dict[str, Structure]:
        """加载QMOF结构数据JSON"""
        return self.file_handler.read_qmof_structure_data(file_path)
    
    def _detect_json_type(self, data: Any) -> str:
        """
        检测JSON数据类型
        
        Returns:
            'qmof_properties', 'qmof_structures', 'structure', 'mofid', 'unknown'
        """
        if isinstance(data, dict):
            # 单个结构
            if 'structure' in data or ('lattice' in data and 'sites' in data):
                return 'structure'
            
            # 单个MOF属性数据
            if 'qmof_id' in data or 'mofid' in data:
                return 'mofid'
            
            return 'unknown'
        
        elif isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            
            if isinstance(first_item, dict):
                # QMOF属性数据
                if 'qmof_id' in first_item and 'outputs' in first_item:
                    return 'qmof_properties'
                
                # QMOF结构数据
                if 'qmof_id' in first_item and 'structure' in first_item:
                    return 'qmof_structures'
                
                # 普通MOFid数据
                if 'mofid' in first_item or 'smiles' in first_item:
                    return 'mofid'
                
                # 结构数据
                if 'structure' in first_item or 'lattice' in first_item:
                    return 'structure'
        
        return 'unknown'
    
    def get_data_summary(self, data: Any, data_type: str) -> Dict[str, Any]:
        """
        获取数据摘要信息
        
        Args:
            data: 数据
            data_type: 数据类型
            
        Returns:
            数据摘要字典
        """
        summary = {
            'data_type': data_type,
            'count': 0,
            'features': [],
            'sample': None,
        }
        
        if data_type == 'tabular':
            if isinstance(data, pd.DataFrame):
                summary['count'] = len(data)
                summary['features'] = list(data.columns)
                summary['sample'] = data.head(3).to_dict('records')
        
        elif data_type in ['qmof_properties', 'mofid']:
            if isinstance(data, list):
                summary['count'] = len(data)
                if len(data) > 0:
                    summary['features'] = list(data[0].keys())
                    summary['sample'] = data[:3]
        
        elif data_type in ['qmof_structures', 'structure_batch']:
            if isinstance(data, dict):
                summary['count'] = len(data)
                summary['features'] = ['structure', 'lattice', 'sites']
                summary['sample'] = list(data.keys())[:3]
        
        elif data_type == 'structure':
            summary['count'] = 1
            summary['features'] = ['lattice', 'sites', 'composition']
            if isinstance(data, Structure):
                summary['sample'] = {
                    'formula': data.composition.formula,
                    'num_sites': len(data),
                }
        
        return summary


