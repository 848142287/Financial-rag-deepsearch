"""
文档格式解析器模块
包含各种文档格式的专用解析器

优化版本：
- 重命名refactored/为formats/，语义更清晰
- 统一导入路径，使用app.services.parsers.base
- 移除_refactored后缀，使用简洁的命名
"""

__all__ = [
    'UnifiedPDFParser',
    'UnifiedExcelParser',
    'UnifiedPPTParser',
    'UnifiedWordParser'
]
