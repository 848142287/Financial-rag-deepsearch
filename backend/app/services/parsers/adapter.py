"""
文件解析适配器
负责路由到对应的文件解析器
"""

import os
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import logging

from .base import BaseFileParser, ParseResult, ParserError, UnsupportedFileTypeError
from .registry import ParserRegistry

logger = logging.getLogger(__name__)


class FileParserAdapter:
    """文件解析适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文件解析适配器

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.registry = ParserRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)

        # 自动注册所有解析器
        self._register_parsers()

    def _register_parsers(self):
        """注册所有解析器"""
        try:
            # 导入所有解析器类（核心解析器）
            from .csv_parser import CSVParser
            from .enhanced_doc_parser import EnhancedDocParser  # 增强Word解析器
            from .enhanced_excel_parser import EnhancedExcelParser  # 增强Excel解析器
            from .ppt_parser_wrapper import PPTParserWrapper  # 增强PPT解析器
            from .advanced.pymupdf4llm_parser import PyMuPDF4LLMParser  # 增强PDF解析器

            # 注册增强解析器（支持多模态分析和语义分块）
            self.registry.register(CSVParser(self.config))
            self.registry.register(EnhancedDocParser(self.config))  # 增强Word解析器（支持多模态）
            self.registry.register(EnhancedExcelParser(self.config))  # 增强Excel解析器（支持语义分块）
            self.registry.register(PPTParserWrapper(self.config))  # 增强PPT解析器（支持多模态）
            self.registry.register(PyMuPDF4LLMParser(self.config))  # 增强PDF解析器（RAG优化）

            self.logger.info(f"Registered {len(self.registry.get_all_parsers())} parsers")
            self.logger.info("Enhanced parsers: PDF (RAG-optimized), DOCX (multimodal), Excel (semantic), PPT (multimodal)")

        except ImportError as e:
            self.logger.warning(f"Failed to import some parsers: {str(e)}")

    async def parse_file(self, file_path: str, **kwargs) -> ParseResult:
        """
        解析文件

        Args:
            file_path: 文件路径
            **kwargs: 解析参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 验证文件
            is_valid, error_msg = self._validate_file(file_path)
            if not is_valid:
                return ParseResult(
                    content="",
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )

            # 获取文件扩展名
            file_extension = Path(file_path).suffix.lower()

            # 查找合适的解析器
            parser = self._get_parser(file_path, file_extension)

            if not parser:
                error_msg = f"No parser found for file extension: {file_extension}"
                self.logger.error(error_msg)
                return ParseResult(
                    content="",
                    metadata={'error': error_msg, 'file_extension': file_extension},
                    success=False,
                    error_message=error_msg
                )

            # 执行解析
            self.logger.info(f"Parsing file {file_path} with {parser.parser_name}")
            result = await parser.parse_with_metadata(file_path, **kwargs)

            # 添加适配器级别的元数据
            result.metadata.update({
                'adapter_version': '1.0.0',
                'routing_info': {
                    'file_extension': file_extension,
                    'selected_parser': parser.parser_name,
                    'fallback_used': False
                }
            })

            return result

        except Exception as e:
            error_msg = f"Failed to parse file {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return ParseResult(
                content="",
                metadata={'error': str(e)},
                success=False,
                error_message=error_msg
            )

    async def parse_file_with_fallback(
        self,
        file_path: str,
        fallback_parser: str = 'text',
        **kwargs
    ) -> ParseResult:
        """
        带回退机制的文件解析

        Args:
            file_path: 文件路径
            fallback_parser: 回退解析器名称
            **kwargs: 解析参数

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 首先尝试正常解析
            result = await self.parse_file(file_path, **kwargs)

            # 如果成功，直接返回
            if result.success:
                return result

            # 如果失败，尝试使用回退解析器
            self.logger.warning(f"Primary parser failed for {file_path}, trying fallback: {fallback_parser}")

            fallback_parser_instance = self.registry.get_parser(fallback_parser)
            if not fallback_parser_instance:
                error_msg = f"Fallback parser '{fallback_parser}' not found"
                self.logger.error(error_msg)
                return result  # 返回原始错误结果

            # 尝试使用回退解析器
            fallback_result = await fallback_parser_instance.parse_with_metadata(file_path, **kwargs)

            # 添加回退信息到元数据
            fallback_result.metadata.update({
                'fallback_used': True,
                'fallback_parser': fallback_parser,
                'original_error': result.error_message
            })

            return fallback_result

        except Exception as e:
            error_msg = f"Fallback parsing also failed for {file_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return ParseResult(
                content="",
                metadata={'error': str(e), 'fallback_failed': True},
                success=False,
                error_message=error_msg
            )

    def get_supported_formats(self) -> List[str]:
        """
        获取支持的文件格式列表

        Returns:
            List[str]: 支持的文件扩展名列表
        """
        extensions = set()
        for parser in self.registry.get_all_parsers():
            extensions.update(parser.supported_extensions)
        return sorted(list(extensions))

    def get_parser_for_format(self, file_extension: str) -> Optional[str]:
        """
        获取指定格式的解析器名称

        Args:
            file_extension: 文件扩展名

        Returns:
            Optional[str]: 解析器名称
        """
        parser = self._get_parser("", file_extension.lower())
        return parser.parser_name if parser else None

    def register_custom_parser(self, parser: BaseFileParser):
        """
        注册自定义解析器

        Args:
            parser: 自定义解析器实例
        """
        self.registry.register(parser)
        self.logger.info(f"Registered custom parser: {parser.parser_name}")

    def unregister_parser(self, parser_name: str) -> bool:
        """
        注销解析器

        Args:
            parser_name: 解析器名称

        Returns:
            bool: 是否成功注销
        """
        return self.registry.unregister(parser_name)

    def _validate_file(self, file_path: str) -> tuple[bool, Optional[str]]:
        """
        验证文件

        Args:
            file_path: 文件路径

        Returns:
            tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"

            # 检查是否为文件
            if not os.path.isfile(file_path):
                return False, f"Path is not a file: {file_path}"

            # 检查文件是否可读
            if not os.access(file_path, os.R_OK):
                return False, f"File not readable: {file_path}"

            # 检查文件大小
            file_size = os.path.getsize(file_path)
            max_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 默认100MB

            if file_size > max_size:
                return False, f"File too large: {file_size} bytes (max: {max_size})"

            if file_size == 0:
                return False, "File is empty"

            return True, None

        except Exception as e:
            return False, f"File validation error: {str(e)}"

    def _get_parser(self, file_path: str, file_extension: str) -> Optional[BaseFileParser]:
        """
        获取适合的解析器

        Args:
            file_path: 文件路径
            file_extension: 文件扩展名

        Returns:
            Optional[BaseFileParser]: 解析器实例
        """
        # 优先通过文件扩展名查找
        if file_extension:
            parser = self.registry.get_parser_by_extension(file_extension)
            if parser and parser.can_parse(file_path, file_extension):
                return parser

        # 如果扩展名匹配失败，尝试让每个解析器检查文件
        for parser in self.registry.get_all_parsers():
            if parser.can_parse(file_path, file_extension):
                return parser

        return None

    def get_parser_info(self) -> Dict[str, Any]:
        """
        获取适配器信息

        Returns:
            Dict[str, Any]: 适配器信息
        """
        return {
            'adapter_name': 'FileParserAdapter',
            'version': '1.0.0',
            'supported_formats': self.get_supported_formats(),
            'registered_parsers': [
                parser.get_parser_info()
                for parser in self.registry.get_all_parsers()
            ],
            'config': self.config
        }

    async def batch_parse(
        self,
        file_paths: List[str],
        max_concurrent: int = 5,
        **kwargs
    ) -> List[ParseResult]:
        """
        批量解析文件

        Args:
            file_paths: 文件路径列表
            max_concurrent: 最大并发数
            **kwargs: 解析参数

        Returns:
            List[ParseResult]: 解析结果列表
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def parse_single(file_path: str) -> ParseResult:
            async with semaphore:
                return await self.parse_file(file_path, **kwargs)

        # 创建任务
        tasks = [parse_single(path) for path in file_paths]

        # 执行任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Batch parsing error for {file_paths[i]}: {str(result)}"
                self.logger.error(error_msg)
                processed_results.append(ParseResult(
                    content="",
                    metadata={'error': str(result)},
                    success=False,
                    error_message=error_msg
                ))
            else:
                processed_results.append(result)

        return processed_results

    def analyze_file_type(self, file_path: str) -> Dict[str, Any]:
        """
        分析文件类型

        Args:
            file_path: 文件路径

        Returns:
            Dict[str, Any]: 文件类型分析结果
        """
        import magic
        import mimetypes

        try:
            file_path_obj = Path(file_path)
            file_extension = file_path_obj.suffix.lower()
            file_size = file_path_obj.stat().st_size

            # 使用python-magic检测文件类型
            mime_type = None
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except:
                # 如果python-magic不可用，使用mimetypes
                mime_type, _ = mimetypes.guess_type(file_path)

            # 查找匹配的解析器
            parser = self._get_parser(file_path, file_extension)
            parser_info = parser.get_parser_info() if parser else None

            return {
                'file_path': file_path,
                'file_extension': file_extension,
                'file_size': file_size,
                'mime_type': mime_type,
                'supported_parser': parser.parser_name if parser else None,
                'parser_info': parser_info,
                'can_parse': parser is not None
            }

        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'can_parse': False
            }