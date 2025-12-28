"""
文档分析增强模块 - Document Analysis Enhancement Module
提供智能内容过滤、语义标注、深度分析等功能
"""

from .intelligent_content_filter import (
    IntelligentContentFilter,
    FilterStatistics
)

from .semantic_unit_annotator import (
    SemanticUnitAnnotator,
    SemanticAnnotation,
    SemanticType
)

from .enhanced_table_analyzer import (
    EnhancedTableAnalyzer,
    TableInsight,
    TableRecommendation
)

from .financial_chart_analyzer import (
    FinancialChartAnalyzer,
    FinancialMetrics
)

from .statistical_chart_analyzer import (
    StatisticalChartAnalyzer
)

from .cross_element_linker import (
    CrossElementLinker,
    ConsistencyValidator
)

from .enhanced_parse_result import (
    EnhancedParseResult,
    DocumentOverview,
    CoreInsights
)

from .document_analysis_pipeline import (
    DocumentAnalysisPipeline,
    PipelineConfig
)

__all__ = [
    # 智能内容过滤
    'IntelligentContentFilter',
    'FilterStatistics',

    # 语义单元标注
    'SemanticUnitAnnotator',
    'SemanticAnnotation',
    'SemanticType',

    # 表格深度分析
    'EnhancedTableAnalyzer',
    'TableInsight',
    'TableRecommendation',

    # 金融图表分析
    'FinancialChartAnalyzer',
    'FinancialMetrics',

    # 统计图表分析
    'StatisticalChartAnalyzer',

    # 跨元素关联
    'CrossElementLinker',
    'ConsistencyValidator',

    # 增强结果
    'EnhancedParseResult',
    'DocumentOverview',
    'CoreInsights',

    # 管道
    'DocumentAnalysisPipeline',
    'PipelineConfig'
]
