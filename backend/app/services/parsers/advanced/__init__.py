"""
高级解析器模块
包含MinerU、PyMuPDF4LLM、VLM等先进解析器
"""

# 导入所有高级解析器
try:
    from .mineru_parser import MinerUParser
except ImportError:
    MinerUParser = None

try:
    from .pymupdf4llm_parser import PyMuPDF4LLMParser
except ImportError:
    PyMuPDF4LLMParser = None

try:
    from .vlm_parser import VLMPreciseParser
except ImportError:
    VLMPreciseParser = None

try:
    from .multimodal_parser import MultimodalParser
except ImportError:
    MultimodalParser = None

try:
    from .image_parser import ImageParser
except ImportError:
    ImageParser = None

try:
    from .chart_parser import ChartParser
except ImportError:
    ChartParser = None

try:
    from .formula_parser import FormulaParser
except ImportError:
    FormulaParser = None

__all__ = [
    'MinerUParser',
    'PyMuPDF4LLMParser',
    'VLMPreciseParser',
    'MultimodalParser',
    'ImageParser',
    'ChartParser',
    'FormulaParser'
]