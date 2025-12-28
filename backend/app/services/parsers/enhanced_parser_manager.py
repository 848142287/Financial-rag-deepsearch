"""
增强的解析器管理器
智能选择最佳解析器，集成所有解析功能

特点：
- 自动根据文档类型选择最佳解析器
- 支持fallback机制
- 性能监控和优化
- 向下兼容现有系统
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import asyncio

from .registry import ParserRegistry
from .base import BaseFileParser, ParseResult

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """文档类型"""
    FINANCIAL_REPORT = "financial_report"  # 财务报告
    ACADEMIC_PAPER = "academic_paper"      # 学术论文
    TECHNICAL_DOC = "technical_doc"        # 技术文档
    CONTRACT = "contract"                  # 合同文档
    SIMPLE_DOC = "simple_doc"              # 简单文档
    UNKNOWN = "unknown"                    # 未知类型


class ParserStrategy(Enum):
    """解析策略"""
    FAST = "fast"              # 快速模式：PyMuPDF4LLM
    PRECISE = "precise"        # 精确模式：MinerU
    VLM = "vlm"                # VLM模式：VLM解析器
    AUTO = "auto"              # 自动模式：智能选择


class EnhancedParserManager:
    """
    增强的解析器管理器

    负责管理所有解析器，并根据策略自动选择最佳解析器
    """

    # 解析器推荐配置
    PARSER_RECOMMENDATIONS = {
        DocumentType.FINANCIAL_REPORT: {
            'primary': 'MinerUParser',     # 首选：MinerU（表格识别强）
            'secondary': 'PyMuPDF4LLMParser',  # 备选：PyMuPDF4LLM
            'fallback': 'VLMPreciseParser'     # 兜底：VLM
        },
        DocumentType.ACADEMIC_PAPER: {
            'primary': 'VLMPreciseParser',     # 首选：VLM（公式识别强）
            'secondary': 'MinerUParser',
            'fallback': 'PyMuPDF4LLMParser'
        },
        DocumentType.TECHNICAL_DOC: {
            'primary': 'PyMuPDF4LLMParser',    # 首选：PyMuPDF4LLM（快速且准确）
            'secondary': 'MinerUParser',
            'fallback': 'VLMPreciseParser'
        },
        DocumentType.CONTRACT: {
            'primary': 'MinerUParser',         # 首选：MinerU
            'secondary': 'VLMPreciseParser',
            'fallback': 'PyMuPDF4LLMParser'
        },
        DocumentType.SIMPLE_DOC: {
            'primary': 'PyMuPDF4LLMParser',    # 首选：PyMuPDF4LLM（最快）
            'secondary': 'MinerUParser',
            'fallback': 'VLMPreciseParser'
        },
        DocumentType.UNKNOWN: {
            'primary': 'PyMuPDF4LLMParser',
            'secondary': 'MinerUParser',
            'fallback': 'VLMPreciseParser'
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增强解析器管理器

        参数:
            config: 配置字典，可包含：
                - parser_configs: 各解析器的配置
                - default_strategy: 默认解析策略
                - enable_fallback: 是否启用fallback
                - performance_monitoring: 是否启用性能监控
        """
        self.config = config or {}
        self.registry = ParserRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)

        # 配置
        self.default_strategy = ParserStrategy(
            self.config.get('default_strategy', 'auto')
        )
        self.enable_fallback = self.config.get('enable_fallback', True)
        self.performance_monitoring = self.config.get('performance_monitoring', True)

        # 性能统计
        self.performance_stats = {}

        # 初始化所有可用解析器
        self._initialize_parsers()

    def _initialize_parsers(self):
        """初始化所有可用的解析器"""
        parser_configs = self.config.get('parser_configs', {})

        # PyMuPDF4LLM解析器
        try:
            from .advanced.pymupdf4llm_parser import PyMuPDF4LLMParser
            pymupdf4llm_config = parser_configs.get('pymupdf4llm', {})
            pymupdf4llm_parser = PyMuPDF4LLMParser(config=pymupdf4llm_config)
            self.registry.register(pymupdf4llm_parser)
            self.logger.info("PyMuPDF4LLM解析器已注册")
        except Exception as e:
            self.logger.warning(f"PyMuPDF4LLM解析器注册失败: {e}")

        # MinerU解析器
        try:
            from .advanced.mineru_parser import MinerUParser
            mineru_config = parser_configs.get('mineru', {})
            mineru_parser = MinerUParser(config=mineru_config)
            self.registry.register(mineru_parser)
            self.logger.info("MinerU解析器已注册")
        except Exception as e:
            self.logger.warning(f"MinerU解析器注册失败: {e}")

        # VLM精确解析器
        try:
            from .advanced.vlm_parser import VLMPreciseParser
            vlm_config = parser_configs.get('vlm', {})
            # 注入model_manager
            vlm_config['model_manager'] = self.config.get('model_manager')
            vlm_parser = VLMPreciseParser(config=vlm_config)
            self.registry.register(vlm_parser)
            self.logger.info("VLM精确解析器已注册")
        except Exception as e:
            self.logger.warning(f"VLM精确解析器注册失败: {e}")

        # 其他现有解析器...
        self.logger.info(f"总共注册了 {self.registry.get_parser_count()} 个解析器")

    async def parse_document(
        self,
        file_path: str,
        strategy: Optional[ParserStrategy] = None,
        document_type: Optional[DocumentType] = None,
        **kwargs
    ) -> ParseResult:
        """
        解析文档（智能选择解析器）

        参数:
            file_path: 文件路径
            strategy: 解析策略（如果为None，使用默认策略）
            document_type: 文档类型（如果为None，自动检测）
            **kwargs: 传递给解析器的额外参数

        返回:
            ParseResult: 解析结果
        """
        # 确定策略
        if strategy is None:
            strategy = self.default_strategy

        # 检测文档类型
        if document_type is None:
            document_type = self._detect_document_type(file_path)

        # 选择解析器
        parser = await self._select_parser(
            file_path,
            strategy,
            document_type
        )

        if parser is None:
            return ParseResult(
                content="",
                metadata={'error': 'No suitable parser found'},
                success=False,
                error_message='No suitable parser found'
            )

        # 执行解析
        return await self._parse_with_fallback(
            parser,
            file_path,
            document_type,
            **kwargs
        )

    async def _select_parser(
        self,
        file_path: str,
        strategy: ParserStrategy,
        document_type: DocumentType
    ) -> Optional[BaseFileParser]:
        """选择最佳解析器"""

        if strategy == ParserStrategy.AUTO:
            # 自动模式：根据文档类型选择
            recommendations = self.PARSER_RECOMMENDATIONS.get(
                document_type,
                self.PARSER_RECOMMENDATIONS[DocumentType.UNKNOWN]
            )

            # 尝试首选解析器
            primary_parser = self._find_parser_by_class_name(recommendations['primary'])
            if primary_parser and primary_parser.can_parse(file_path):
                self.logger.info(f"使用首选解析器: {primary_parser.parser_name}")
                return primary_parser

            # 尝试备选解析器
            secondary_parser = self._find_parser_by_class_name(recommendations['secondary'])
            if secondary_parser and secondary_parser.can_parse(file_path):
                self.logger.info(f"使用备选解析器: {secondary_parser.parser_name}")
                return secondary_parser

            # 使用fallback
            fallback_parser = self._find_parser_by_class_name(recommendations['fallback'])
            if fallback_parser and fallback_parser.can_parse(file_path):
                self.logger.info(f"使用fallback解析器: {fallback_parser.parser_name}")
                return fallback_parser

        elif strategy == ParserStrategy.FAST:
            # 快速模式：优先PyMuPDF4LLM
            parser = self.registry.get_parser_by_name('PyMuPDF4LLM')
            if parser and parser.can_parse(file_path):
                return parser

        elif strategy == ParserStrategy.PRECISE:
            # 精确模式：优先MinerU
            parser = self.registry.get_parser_by_name('MinerU')
            if parser and parser.can_parse(file_path):
                return parser

        elif strategy == ParserStrategy.VLM:
            # VLM模式
            parser = self.registry.get_parser_by_name('VLMPrecise')
            if parser and parser.can_parse(file_path):
                return parser

        # 如果没有找到指定解析器，使用任何可用的
        parsers = self.registry.get_parsers_by_extension(Path(file_path).suffix)
        if parsers:
            return parsers[0]

        return None

    def _find_parser_by_class_name(self, class_name: str) -> Optional[BaseFileParser]:
        """根据类名查找解析器"""
        for parser in self.registry.get_all_parsers():
            if parser.__class__.__name__ == class_name:
                return parser
        return None

    async def _parse_with_fallback(
        self,
        parser: BaseFileParser,
        file_path: str,
        document_type: DocumentType,
        **kwargs
    ) -> ParseResult:
        """使用fallback机制解析文档"""

        # 首次尝试
        result = await parser.parse_with_metadata(file_path, **kwargs)

        if result.success:
            # 记录性能
            if self.performance_monitoring:
                self._record_performance(parser.parser_name, result)
            return result

        # 如果失败且启用了fallback
        if not self.enable_fallback:
            return result

        self.logger.warning(f"解析器 {parser.parser_name} 失败，尝试fallback")

        # 获取fallback解析器
        recommendations = self.PARSER_RECOMMENDATIONS.get(
            document_type,
            self.PARSER_RECOMMENDATIONS[DocumentType.UNKNOWN]
        )

        # 尝试fallback解析器
        fallback_parsers = [
            recommendations['secondary'],
            recommendations['fallback']
        ]

        for fallback_class_name in fallback_parsers:
            fallback_parser = self._find_parser_by_class_name(fallback_class_name)
            if fallback_parser and fallback_parser.can_parse(file_path):
                self.logger.info(f"尝试fallback解析器: {fallback_parser.parser_name}")

                fallback_result = await fallback_parser.parse_with_metadata(file_path, **kwargs)

                if fallback_result.success:
                    self.logger.info(f"Fallback解析器 {fallback_parser.parser_name} 成功")
                    if self.performance_monitoring:
                        self._record_performance(fallback_parser.parser_name, fallback_result)
                    return fallback_result

        # 所有fallback都失败
        return result

    def _detect_document_type(self, file_path: str) -> DocumentType:
        """检测文档类型（基于文件名和启发式规则）"""

        filename = Path(file_path).name.lower()

        # 财务报告
        if any(keyword in filename for keyword in [
            '财务', '报表', '年报', '季报', '审计',
            'financial', 'report', 'annual', 'quarterly'
        ]):
            return DocumentType.FINANCIAL_REPORT

        # 学术论文
        if any(keyword in filename for keyword in [
            '论文', '研究', '期刊',
            'paper', 'research', 'journal', 'thesis'
        ]):
            return DocumentType.ACADEMIC_PAPER

        # 技术文档
        if any(keyword in filename for keyword in [
            '技术', '文档', '手册', '指南',
            'technical', 'manual', 'guide', 'documentation'
        ]):
            return DocumentType.TECHNICAL_DOC

        # 合同
        if any(keyword in filename for keyword in [
            '合同', '协议', '条款',
            'contract', 'agreement'
        ]):
            return DocumentType.CONTRACT

        # 默认为简单文档
        return DocumentType.SIMPLE_DOC

    def _record_performance(self, parser_name: str, result: ParseResult):
        """记录性能统计"""
        if parser_name not in self.performance_stats:
            self.performance_stats[parser_name] = {
                'total_calls': 0,
                'success_calls': 0,
                'total_time': 0,
                'avg_time': 0
            }

        stats = self.performance_stats[parser_name]
        stats['total_calls'] += 1

        if result.success:
            stats['success_calls'] += 1
            if result.parse_time:
                stats['total_time'] += result.parse_time
                stats['avg_time'] = stats['total_time'] / stats['success_calls']

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats

    def get_registry(self) -> ParserRegistry:
        """获取解析器注册表"""
        return self.registry


# 便捷函数
async def parse_document_auto(
    file_path: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ParseResult:
    """
    自动解析文档（使用最佳解析器）

    参数:
        file_path: 文件路径
        config: 配置字典
        **kwargs: 额外参数

    返回:
        ParseResult: 解析结果

    示例:
        result = await parse_document_auto("document.pdf")
        if result.success:
            print(result.content)
    """
    manager = EnhancedParserManager(config)
    return await manager.parse_document(file_path, **kwargs)
