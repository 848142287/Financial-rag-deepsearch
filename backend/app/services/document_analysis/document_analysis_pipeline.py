"""
文档分析管道 - Document Analysis Pipeline
集成所有增强功能的统一接口
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .intelligent_content_filter import IntelligentContentFilter
from .semantic_unit_annotator import SemanticUnitAnnotator
from .enhanced_table_analyzer import EnhancedTableAnalyzer
from .financial_chart_analyzer import FinancialChartAnalyzer
from .statistical_chart_analyzer import StatisticalChartAnalyzer
from .cross_element_linker import CrossElementLinker, ConsistencyValidator
from .enhanced_parse_result import EnhancedParseResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """管道配置"""
    # 内容过滤配置
    enable_filtering: bool = True
    min_content_length: int = 10
    min_content_density: float = 0.3

    # 语义标注配置
    enable_annotation: bool = True
    use_llm_for_annotation: bool = False

    # 表格分析配置
    enable_table_analysis: bool = True

    # 图表分析配置
    enable_chart_analysis: bool = True

    # 关联建立配置
    enable_linking: bool = True

    # 一致性验证配置
    enable_validation: bool = True


class DocumentAnalysisPipeline:
    """文档分析管道"""

    def __init__(self, config: PipelineConfig = None):
        """
        初始化管道

        Args:
            config: 管道配置
        """
        self.config = config or PipelineConfig()

        # 初始化各个分析器
        if self.config.enable_filtering:
            self.content_filter = IntelligentContentFilter({
                'min_length': self.config.min_content_length,
                'min_density': self.config.min_content_density
            })

        if self.config.enable_annotation:
            self.semantic_annotator = SemanticUnitAnnotator({
                'use_llm': self.config.use_llm_for_annotation
            })

        if self.config.enable_table_analysis:
            self.table_analyzer = EnhancedTableAnalyzer()

        if self.config.enable_chart_analysis:
            self.financial_chart_analyzer = FinancialChartAnalyzer()
            self.statistical_chart_analyzer = StatisticalChartAnalyzer()

        if self.config.enable_linking:
            self.element_linker = CrossElementLinker()

        if self.config.enable_validation:
            self.consistency_validator = ConsistencyValidator()

    async def analyze_document(
        self,
        elements: List[Any],
        parse_result: Any = None
    ) -> EnhancedParseResult:
        """
        分析文档

        Args:
            elements: 文档元素列表
            parse_result: 原始解析结果

        Returns:
            EnhancedParseResult: 增强解析结果
        """
        import time
        start_time = time.time()

        try:
            # 准备基础结果
            if parse_result:
                base_result = EnhancedParseResult(
                    content=parse_result.content,
                    metadata=parse_result.metadata,
                    success=parse_result.success,
                    error_message=parse_result.error_message,
                    parse_time=parse_result.parse_time
                )
            else:
                base_result = EnhancedParseResult(
                    content="",
                    metadata={},
                    success=True
                )

            # 第一步：内容过滤
            if self.config.enable_filtering:
                elements, filter_stats = self.content_filter.filter_document_elements(elements)
                base_result.filter_statistics = {
                    'total_elements': filter_stats.total_elements,
                    'filtered_elements': filter_stats.filtered_elements,
                    'kept_elements': filter_stats.kept_elements,
                    'noise_types_removed': filter_stats.noise_types_removed
                }
                logger.info(f"Content filtering: {filter_stats.kept_elements}/{filter_stats.total_elements} elements kept")

            # 第二步：语义标注
            if self.config.enable_annotation:
                elements, annotation_stats = await self.semantic_annotator.annotate_document(elements)
                base_result.annotation_statistics = annotation_stats
                logger.info(f"Semantic annotation: {annotation_stats['annotated_paragraphs']} paragraphs annotated")

            # 第三步：表格深度分析
            table_analyses = []
            if self.config.enable_table_analysis:
                for element in elements:
                    if getattr(element, 'element_type', None) == 'table':
                        metadata = getattr(element, 'metadata', {})
                        table_data = metadata.get('data', []) if metadata else []

                        if table_data:
                            analysis = self.table_analyzer.analyze_table_deeply(table_data, metadata)
                            table_analyses.append(analysis)

                            # 将分析结果存回元素元数据
                            if hasattr(element, 'metadata'):
                                element.metadata['deep_analysis'] = analysis

                base_result.table_analyses = table_analyses
                logger.info(f"Table analysis: {len(table_analyses)} tables analyzed")

            # 第四步：图表分析
            chart_analyses = []
            if self.config.enable_chart_analysis:
                for element in elements:
                    elem_type = getattr(element, 'element_type', None)

                    if elem_type in ['image', 'chart']:
                        metadata = getattr(element, 'metadata', {})

                        # 根据图表类型选择分析器
                        image_type = metadata.get('image_type', '')

                        if image_type == 'chart':
                            # 金融图表分析
                            # 这里需要实际的数据点，暂时用空列表
                            chart_analysis = self.financial_chart_analyzer.analyze_financial_chart(
                                metadata.get('chart_info', {})
                            )
                            chart_analyses.append({
                                'type': 'financial',
                                'analysis': chart_analysis
                            })

                        elif metadata.get('ocr_text'):  # 有OCR文字的图表
                            # 统计图表分析
                            chart_analysis = self.statistical_chart_analyzer.analyze_statistical_chart(
                                metadata.get('chart_info', {})
                            )
                            chart_analyses.append({
                                'type': 'statistical',
                                'analysis': chart_analysis
                            })

                base_result.chart_analyses = chart_analyses
                logger.info(f"Chart analysis: {len(chart_analyses)} charts analyzed")

            # 第五步：跨元素关联建立
            if self.config.enable_linking:
                content_network = self.element_linker.build_content_network(elements)
                base_result.cross_references = content_network
                logger.info("Cross-element linking completed")

            # 第六步：一致性验证
            if self.config.enable_validation:
                validation_results = self.consistency_validator.validate_document(elements)
                base_result.validation_results = validation_results
                logger.info(f"Consistency validation: {validation_results.get('overall_rating', 'Unknown')} rating")

            # 第七步：生成核心洞察
            insights = self._generate_core_insights(
                base_result,
                elements,
                table_analyses,
                chart_analyses
            )
            base_result.insights = insights

            # 更新元数据
            base_result.semantic_structure = self._build_semantic_structure(elements)
            base_result.content_quality = self._assess_content_quality(base_result)

            # 更新解析时间
            total_time = time.time() - start_time
            base_result.parse_time = total_time

            logger.info(f"Document analysis completed in {total_time:.2f}s")

            return base_result

        except Exception as e:
            logger.error(f"Error in document analysis pipeline: {e}", exc_info=True)

            # 返回部分结果
            if parse_result:
                return EnhancedParseResult(
                    content=parse_result.content,
                    metadata=parse_result.metadata,
                    success=False,
                    error_message=f"Analysis failed: {str(e)}",
                    parse_time=time.time() - start_time
                )
            else:
                return EnhancedParseResult(
                    content="",
                    metadata={},
                    success=False,
                    error_message=str(e),
                    parse_time=time.time() - start_time
                )

    def _build_semantic_structure(self, elements: List[Any]) -> Dict[str, Any]:
        """构建语义结构"""
        structure = {
            'total_elements': len(elements),
            'element_types': {},
            'semantic_types': {},
            'hierarchy': []
        }

        # 统计元素类型
        for element in elements:
            elem_type = getattr(element, 'element_type', 'unknown')
            structure['element_types'][elem_type] = \
                structure['element_types'].get(elem_type, 0) + 1

            # 统计语义类型
            metadata = getattr(element, 'metadata', {})
            if metadata and 'semantic_annotation' in metadata:
                sem_type = metadata['semantic_annotation'].get('semantic_type', 'unknown')
                structure['semantic_types'][sem_type] = \
                    structure['semantic_types'].get(sem_type, 0) + 1

        return structure

    def _assess_content_quality(self, result: EnhancedParseResult) -> Dict[str, Any]:
        """评估内容质量"""
        quality = {
            'completeness': 'A',  # 简化实现
            'consistency': result.validation_results.get('overall_rating', 'B') if result.validation_results else 'B',
            'clarity': 'B',
            'overall_score': 0.85
        }

        return quality

    def _generate_core_insights(
        self,
        result: EnhancedParseResult,
        elements: List[Any],
        table_analyses: List[Dict],
        chart_analyses: List[Dict]
    ) -> Dict[str, Any]:
        """生成核心洞察"""
        insights = {
            'key_findings': [],
            'data_insights': {},
            'action_items': []
        }

        # 从表格分析中提取洞察
        for table_analysis in table_analyses:
            table_insights = table_analysis.get('insights', [])
            for insight in table_insights[:2]:  # 每个表格最多2个洞察
                if insight.get('importance') == '高':
                    insights['key_findings'].append({
                        'finding': insight.get('description', ''),
                        'source': 'table_analysis',
                        'importance': insight.get('importance', '中')
                    })

            # 业务逻辑建议
            business_logic = table_analysis.get('business_logic', {})
            action_suggestions = business_logic.get('action_suggestions', [])
            insights['action_items'].extend(action_suggestions[:2])

        # 从图表分析中提取洞察
        for chart_analysis in chart_analyses:
            if chart_analysis['type'] == 'financial':
                fin_analysis = chart_analysis['analysis']
                interpretation = fin_analysis.get('interpretation', [])
                insights['key_findings'].extend([
                    {
                        'finding': interp,
                        'source': 'financial_chart',
                        'importance': '高'
                    }
                    for interp in interpretation[:2]
                ])

        # 数据洞察汇总
        insights['data_insights'] = {
            'tables_analyzed': len(table_analyses),
            'charts_analyzed': len(chart_analyses),
            'total_insights': len(insights['key_findings']),
            'total_recommendations': len(insights['action_items'])
        }

        return insights
