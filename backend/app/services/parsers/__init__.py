"""
文件解析器模块
提供多种文件格式的解析功能
"""

from .adapter import FileParserAdapter
from .base import BaseFileParser
from .csv_parser import CSVParser
from .enhanced_doc_parser import EnhancedDocParser  # 增强Word解析器
from .legacy_doc_parser import LegacyDocParser  # 旧版Word解析器
from .enhanced_excel_parser import EnhancedExcelParser  # 增强Excel解析器
from .ppt_parser_wrapper import PPTParserWrapper  # 增强PPT解析器
from .markdown_parser import MarkdownParser  # Markdown解析器
from .text_parser import TextParser  # 文本解析器
from .advanced.pymupdf4llm_parser import PyMuPDF4LLMParser  # 增强PDF解析器
from .registry import ParserRegistry

__all__ = [
    'FileParserAdapter',
    'BaseFileParser',
    'CSVParser',
    'EnhancedDocParser',  # 增强Word解析器（支持多模态分析）
    'LegacyDocParser',  # 旧版Word解析器（支持.doc格式）
    'EnhancedExcelParser',  # 增强Excel解析器（支持语义分块）
    'PPTParserWrapper',  # 增强PPT解析器（支持多模态分析）
    'MarkdownParser',  # Markdown解析器（支持结构化提取和智能分块）
    'TextParser',  # 文本解析器（支持语言检测和段落分块）
    'PyMuPDF4LLMParser',  # 增强PDF解析器（RAG优化）
    'ParserRegistry'
]