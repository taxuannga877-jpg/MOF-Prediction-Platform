"""
日志系统
Logging System
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "MOF_Platform",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        console_output: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class PlatformLogger:
    """平台日志管理类"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建不同类型的日志文件
        timestamp = datetime.now().strftime("%Y%m%d")
        
        self.main_logger = setup_logger(
            "Main",
            self.log_dir / f"main_{timestamp}.log"
        )
        
        self.model_logger = setup_logger(
            "Model",
            self.log_dir / f"model_{timestamp}.log"
        )
        
        self.data_logger = setup_logger(
            "Data",
            self.log_dir / f"data_{timestamp}.log"
        )
        
    def log_info(self, message: str, logger_type: str = "main"):
        """记录信息日志"""
        logger = getattr(self, f"{logger_type}_logger", self.main_logger)
        logger.info(message)
        
    def log_warning(self, message: str, logger_type: str = "main"):
        """记录警告日志"""
        logger = getattr(self, f"{logger_type}_logger", self.main_logger)
        logger.warning(message)
        
    def log_error(self, message: str, logger_type: str = "main"):
        """记录错误日志"""
        logger = getattr(self, f"{logger_type}_logger", self.main_logger)
        logger.error(message)
        
    def log_debug(self, message: str, logger_type: str = "main"):
        """记录调试日志"""
        logger = getattr(self, f"{logger_type}_logger", self.main_logger)
        logger.debug(message)


