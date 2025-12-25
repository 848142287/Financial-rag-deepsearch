"""
文档转换器工厂
根据需求自动选择合适的转换器
"""

import logging
from typing import Dict, Any, Optional, List, Type
from pathlib import Path

from .base_converter import BaseConverter, ConversionResult
from .libreoffice_converter import LibreOfficeConverter
from .reportlab_converter import ReportLabConverter

logger = logging.getLogger(__name__)


class ConverterFactory:
    """文档转换器工厂"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._converters: Dict[str, BaseConverter] = {}
        self._register_converters()

    def _register_converters(self):
        """注册所有可用的转换器"""
        # 注册LibreOffice转换器
        try:
            self._converters['libreoffice'] = LibreOfficeConverter(
                self.config.get('libreoffice', {})
            )
            logger.info("LibreOffice转换器注册成功")
        except Exception as e:
            logger.warning(f"LibreOffice转换器注册失败: {str(e)}")

        # 注册ReportLab转换器
        try:
            self._converters['reportlab'] = ReportLabConverter(
                self.config.get('reportlab', {})
            )
            logger.info("ReportLab转换器注册成功")
        except Exception as e:
            logger.warning(f"ReportLab转换器注册失败: {str(e)}")

    def get_converter(self, converter_name: str) -> Optional[BaseConverter]:
        """获取指定的转换器"""
        return self._converters.get(converter_name)

    def get_available_converters(self) -> List[str]:
        """获取可用的转换器列表"""
        return list(self._converters.keys())

    def select_best_converter(
        self,
        input_format: str,
        output_format: str
    ) -> Optional[BaseConverter]:
        """根据输入输出格式选择最佳转换器"""
        converter_scores = {}

        for name, converter in self._converters.items():
            input_formats = converter.get_supported_input_formats()
            output_formats = converter.get_supported_output_formats()

            # 检查是否支持输入输出格式
            supports_input = input_format.lower() in input_formats
            supports_output = output_format.lower() in output_formats

            if supports_input and supports_output:
                # 计算转换器优先级分数
                score = self._calculate_converter_score(name, input_format, output_format)
                converter_scores[name] = score

        if not converter_scores:
            return None

        # 返回分数最高的转换器
        best_converter_name = max(converter_scores, key=converter_scores.get)
        return self._converters[best_converter_name]

    def _calculate_converter_score(
        self,
        converter_name: str,
        input_format: str,
        output_format: str
    ) -> int:
        """计算转换器适用性分数"""
        base_score = 0

        # LibreOffice在Office文档转换方面表现最好
        if converter_name == 'libreoffice':
            if input_format in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
                base_score += 10
            if output_format == 'pdf':
                base_score += 5

        # ReportLab在文本到PDF转换方面表现好
        elif converter_name == 'reportlab':
            if input_format in ['.txt', '.md', '.html']:
                base_score += 10
            if output_format == 'pdf':
                base_score += 10

        return base_score

    async def convert_document(
        self,
        input_path: str,
        output_format: str,
        output_path: Optional[str] = None,
        preferred_converter: Optional[str] = None
    ) -> ConversionResult:
        """转换文档"""
        input_format = Path(input_path).suffix.lower()

        # 选择转换器
        if preferred_converter:
            converter = self.get_converter(preferred_converter)
            if not converter:
                return ConversionResult(
                    status="failed",
                    error_message=f"指定的转换器 {preferred_converter} 不可用"
                )
        else:
            converter = self.select_best_converter(input_format, output_format)
            if not converter:
                return ConversionResult(
                    status="failed",
                    error_message=f"没有找到支持 {input_format} -> {output_format} 转换的转换器"
                )

        logger.info(f"使用 {converter.converter_name} 转换文档")

        # 执行转换
        return await converter.convert(input_path, output_format, output_path)

    async def batch_convert(
        self,
        input_files: List[str],
        output_format: str,
        output_dir: Optional[str] = None,
        preferred_converter: Optional[str] = None
    ) -> List[ConversionResult]:
        """批量转换文档"""
        results = []

        for input_file in input_files:
            output_path = None
            if output_dir:
                filename = Path(input_file).stem + f".{output_format}"
                output_path = str(Path(output_dir) / filename)

            result = await self.convert_document(
                input_file,
                output_format,
                output_path,
                preferred_converter
            )
            results.append(result)

        return results

    def get_conversion_capabilities(self) -> Dict[str, Any]:
        """获取转换能力信息"""
        capabilities = {
            'available_converters': {},
            'supported_conversions': []
        }

        for name, converter in self._converters.items():
            input_formats = converter.get_supported_input_formats()
            output_formats = converter.get_supported_output_formats()

            capabilities['available_converters'][name] = {
                'input_formats': input_formats,
                'output_formats': output_formats
            }

            # 添加支持的转换路径
            for input_fmt in input_formats:
                for output_fmt in output_formats:
                    capabilities['supported_conversions'].append({
                        'from': input_fmt,
                        'to': output_fmt,
                        'converter': name
                    })

        return capabilities

    async def auto_detect_and_convert(
        self,
        input_path: str,
        target_format: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> ConversionResult:
        """自动检测并转换文档"""
        input_format = Path(input_path).suffix.lower()

        # 如果没有指定目标格式，根据输入格式推荐一个
        if not target_format:
            target_format = self._recommend_output_format(input_format)

        if not target_format:
            return ConversionResult(
                status="failed",
                error_message="无法确定目标格式，请手动指定"
            )

        return await self.convert_document(input_path, target_format, None)

    def _recommend_output_format(self, input_format: str) -> Optional[str]:
        """根据输入格式推荐输出格式"""
        recommendations = {
            '.doc': 'pdf',
            '.docx': 'pdf',
            '.xls': 'xlsx',
            '.xlsx': 'pdf',
            '.ppt': 'pdf',
            '.pptx': 'pdf',
            '.txt': 'pdf',
            '.md': 'pdf',
            '.html': 'pdf'
        }

        return recommendations.get(input_format.lower())