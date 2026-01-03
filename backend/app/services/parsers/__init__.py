"""
文档解析器模块
提供统一的文档解析接口

主要组件：
- base: 基类和数据类定义
- formats: 文档格式解析器（PDF、Office等）
- factory: Parser工厂模式（推荐使用）
- text_parser, csv_parser: 简单格式解析器

使用示例：
    from app.services.parsers.factory import ParserFactory

    # 方式1: 使用工厂（推荐）
    parser = ParserFactory.get_parser('/path/to/file.pdf')
    result = await parser.parse('/path/to/file.pdf')

    # 方式2: 使用便捷函数
    from app.services.parsers.factory import parse_file
    result = await parse_file('/path/to/file.pdf')
"""

# 基类和数据类

# Parser工厂（推荐使用）

# 格式解析器

# 简单解析器

__all__ = [
    # 基类
    'BaseFileParser',
    'BaseDocumentParser',

    # 数据类
    'ParseResult',
    'DocumentChunk',
    'DocumentMetadata',
    'SectionData',
    'TableData',
    'ImageData',

    # 异常
    'ParserError',
    'UnsupportedFileTypeError',
    'FileParsingError',
    'FileValidationError',

    # 工厂模式（推荐）
    'ParserFactory',
    'get_parser',
    'parse_file',
    'is_supported_file',

    # 格式解析器
    'UnifiedPDFParser',
    'UnifiedExcelParser',
    'UnifiedPPTParser',
    'UnifiedWordParser',
    'TextParser',
    'CSVParser',
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'Financial RAG Team'
