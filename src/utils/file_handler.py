"""
文件处理模块
File Handler Module
"""

import json
import zipfile
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import pandas as pd
from pymatgen.core import Structure


class FileHandler:
    """文件处理类"""
    
    @staticmethod
    def read_cif(file_path: Union[str, Path]) -> Structure:
        """
        读取CIF文件
        
        Args:
            file_path: CIF文件路径
            
        Returns:
            Pymatgen Structure对象
        """
        try:
            structure = Structure.from_file(str(file_path))
            return structure
        except Exception as e:
            raise ValueError(f"无法读取CIF文件 {file_path}: {str(e)}")
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        读取JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            JSON数据字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise ValueError(f"无法读取JSON文件 {file_path}: {str(e)}")
    
    @staticmethod
    def read_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """
        读取CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            Pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise ValueError(f"无法读取CSV文件 {file_path}: {str(e)}")
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path]):
        """
        保存为JSON文件
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: Union[str, Path]):
        """
        保存为CSV文件
        
        Args:
            df: DataFrame
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
    
    @staticmethod
    def extract_zip(zip_path: Union[str, Path], extract_to: Union[str, Path]) -> Path:
        """
        解压ZIP文件
        
        Args:
            zip_path: ZIP文件路径
            extract_to: 解压目标路径
            
        Returns:
            解压后的目录路径
        """
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        return extract_to
    
    @staticmethod
    def detect_file_type(file_path: Union[str, Path]) -> str:
        """
        检测文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件类型: 'cif', 'json', 'csv', 'zip', 'unknown'
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            '.cif': 'cif',
            '.json': 'json',
            '.csv': 'csv',
            '.zip': 'zip',
        }
        
        return type_mapping.get(suffix, 'unknown')
    
    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            
        Returns:
            文件路径列表
        """
        directory = Path(directory)
        if not directory.exists():
            return []
        
        return list(directory.glob(pattern))
    
    @staticmethod
    def read_qmof_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        读取QMOF JSON文件
        
        Args:
            file_path: QMOF JSON文件路径
            
        Returns:
            MOF数据列表
        """
        data = FileHandler.read_json(file_path)
        
        # QMOF JSON可能是列表或字典
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("无法识别的QMOF JSON格式")
    
    @staticmethod
    def read_qmof_structure_data(file_path: Union[str, Path]) -> Dict[str, Structure]:
        """
        读取QMOF结构数据JSON
        
        Args:
            file_path: qmof_structure_data.json路径
            
        Returns:
            {qmof_id: Structure}字典
        """
        data = FileHandler.read_json(file_path)
        
        structures = {}
        for entry in data:
            qmof_id = entry.get('qmof_id')
            structure_dict = entry.get('structure')
            
            if qmof_id and structure_dict:
                try:
                    structure = Structure.from_dict(structure_dict)
                    structures[qmof_id] = structure
                except Exception as e:
                    print(f"警告: 无法解析结构 {qmof_id}: {str(e)}")
        
        return structures
    
    @staticmethod
    def validate_file(file_path: Union[str, Path]) -> bool:
        """
        验证文件是否存在且可读
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否有效
        """
        file_path = Path(file_path)
        return file_path.exists() and file_path.is_file()
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        
        return {
            "exists": True,
            "name": file_path.name,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "type": FileHandler.detect_file_type(file_path),
            "modified": stat.st_mtime,
        }


