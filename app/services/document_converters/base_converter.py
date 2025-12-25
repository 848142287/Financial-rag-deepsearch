"""
文档转换器基类
定义转换器接口和通用功能
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class ConversionStatus(Enum):
    """转换状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConversionResult:
    """转换结果"""
    status: ConversionStatus
    output_path: Optional[str] = None
    output_format: Optional[str] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    conversion_time: Optional[float] = None
    file_size: Optional[int] = None


class BaseConverter(ABC):
    """文档转换器基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.converter_name = self.__class__.__name__
        self.temp_dir = self.config.get('temp_dir', '/tmp/document_conversion')

        # 确保临时目录存在
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def convert(
        self,
        input_path: str,
        output_format: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ConversionResult:
        """
        转换文档

        Args:
            input_path: 输入文件路径
            output_format: 目标格式
            output_path: 输出文件路径（可选）
            **kwargs: 其他转换参数

        Returns:
            ConversionResult: 转换结果
        """
        pass

    @abstractmethod
    def get_supported_input_formats(self) -> List[str]:
        """获取支持的输入格式"""
        pass

    @abstractmethod
    def get_supported_output_formats(self) -> List[str]:
        """获取支持的输出格式"""
        pass

    def validate_input_file(self, input_path: str) -> bool:
        """验证输入文件"""
        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            return False

        # 检查文件大小
        file_size = os.path.getsize(input_path)
        max_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB

        if file_size > max_size:
            logger.error(f"文件过大: {file_size} bytes, 最大支持: {max_size} bytes")
            return False

        # 检查文件扩展名
        file_ext = Path(input_path).suffix.lower()
        supported_formats = self.get_supported_input_formats()

        if file_ext not in supported_formats:
            logger.error(f"不支持的文件格式: {file_ext}")
            return False

        return True

    def generate_output_path(
        self,
        input_path: str,
        output_format: str,
        output_path: Optional[str] = None
    ) -> str:
        """生成输出文件路径"""
        if output_path:
            return output_path

        # 在临时目录中生成输出路径
        input_file = Path(input_path)
        output_filename = f"{input_file.stem}_converted.{output_format}"
        output_path = os.path.join(self.temp_dir, output_filename)

        return output_path

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """获取文件元数据"""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'extension': Path(file_path).suffix.lower(),
                'mime_type': self._get_mime_type(file_path)
            }
        except Exception as e:
            logger.error(f"获取文件元数据失败: {str(e)}")
            return {}

    def _get_mime_type(self, file_path: str) -> str:
        """获取MIME类型"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'

    async def cleanup_temp_files(self, file_paths: List[str]):
        """清理临时文件"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"已删除临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败 {file_path}: {str(e)}")