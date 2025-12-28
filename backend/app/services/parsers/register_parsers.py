"""
解析器注册辅助脚本
用于注册和管理所有文档解析器
"""

from .registry import ParserRegistry
from .markdown_parser import MarkdownParser
from .text_parser import TextParser
from .csv_parser import CSVParser
from .enhanced_doc_parser import EnhancedDocParser
from .legacy_doc_parser import LegacyDocParser
from .enhanced_excel_parser import EnhancedExcelParser
from .ppt_parser_wrapper import PPTParserWrapper
from .advanced.pymupdf4llm_parser import PyMuPDF4LLMParser
import logging

logger = logging.getLogger(__name__)


def register_all_parsers(config: dict = None) -> ParserRegistry:
    """
    注册所有解析器到注册表

    Args:
        config: 解析器配置字典

    Returns:
        ParserRegistry: 包含所有解析器的注册表
    """
    config = config or {}
    registry = ParserRegistry()

    # Markdown解析器
    md_parser = MarkdownParser(config.get('markdown', {}))
    registry.register(md_parser)
    logger.info(f"Registered {md_parser.parser_name} for {md_parser.supported_extensions}")

    # 文本解析器
    txt_parser = TextParser(config.get('text', {}))
    registry.register(txt_parser)
    logger.info(f"Registered {txt_parser.parser_name} for {txt_parser.supported_extensions}")

    # CSV解析器
    csv_parser = CSVParser(config.get('csv', {}))
    registry.register(csv_parser)
    logger.info(f"Registered {csv_parser.parser_name} for {csv_parser.supported_extensions}")

    # Word解析器
    doc_parser = EnhancedDocParser(config.get('word', {}))
    registry.register(doc_parser)
    logger.info(f"Registered {doc_parser.parser_name} for {doc_parser.supported_extensions}")

    # 旧版Word解析器
    legacy_doc_parser = LegacyDocParser(config.get('legacy_word', {}))
    registry.register(legacy_doc_parser)
    logger.info(f"Registered {legacy_doc_parser.parser_name} for {legacy_doc_parser.supported_extensions}")

    # Excel解析器
    excel_parser = EnhancedExcelParser(config.get('excel', {}))
    registry.register(excel_parser)
    logger.info(f"Registered {excel_parser.parser_name} for {excel_parser.supported_extensions}")

    # PPT解析器
    ppt_parser = PPTParserWrapper(config.get('ppt', {}))
    registry.register(ppt_parser)
    logger.info(f"Registered {ppt_parser.parser_name} for {ppt_parser.supported_extensions}")

    # PDF解析器
    pdf_parser = PyMuPDF4LLMParser(config.get('pdf', {}))
    registry.register(pdf_parser)
    logger.info(f"Registered {pdf_parser.parser_name} for {pdf_parser.supported_extensions}")

    logger.info(f"Total parsers registered: {registry.get_parser_count()}")
    logger.info(f"Supported extensions: {registry.get_supported_extensions()}")

    return registry


def get_default_registry() -> ParserRegistry:
    """
    获取默认配置的解析器注册表

    Returns:
        ParserRegistry: 解析器注册表
    """
    # 默认配置
    default_config = {
        'markdown': {
            'extract_metadata': True,
            'preserve_html': False,
            'chunk_config': {
                'max_chunk_size': 2000,
                'chunk_overlap': 200,
                'min_chunk_size': 100
            }
        },
        'text': {
            'detect_language': True,
            'preserve_whitespace': False,
            'chunk_by_paragraph': True
        },
        'csv': {
            'encoding': 'utf-8',
            'delimiter': ','
        },
        'excel': {
            'extract_all_sheets': True
        },
        'word': {
            'extract_images': True
        },
        'legacy_word': {
            'extract_images': False,
            'enable_ocr': False
        },
        'ppt': {
            'extract_images': True
        },
        'pdf': {
            'extract_images': True,
            'ocr_enabled': False
        }
    }

    return register_all_parsers(default_config)


if __name__ == "__main__":
    # 测试注册所有解析器
    logging.basicConfig(level=logging.INFO)

    registry = get_default_registry()

    print("\n" + "="*60)
    print("解析器注册表统计")
    print("="*60)
    print(f"总解析器数量: {registry.get_parser_count()}")
    print(f"支持的扩展名数量: {registry.get_extension_count()}")
    print(f"\n支持的文件类型:")
    for ext_info in registry.list_extensions():
        print(f"  {ext_info['extension']} -> {ext_info['parser_names']}")
