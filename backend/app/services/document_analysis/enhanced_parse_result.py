"""
增强解析结果 - Enhanced Parse Result
包含所有深度分析结果的扩展数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class EnhancedParseResult:
    """增强解析结果"""

    # 原有字段（保持兼容性）
    content: str
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    parse_time: float = 0

    # 新增字段 - 深度分析结果
    semantic_structure: Optional[Dict[str, Any]] = None  # 语义结构
    content_quality: Optional[Dict[str, Any]] = None     # 内容质量评估
    cross_references: Optional[List[Dict]] = None        # 交叉引用
    insights: Optional[Dict[str, Any]] = None            # 核心洞察
    validation_results: Optional[Dict[str, Any]] = None  # 验证结果

    # 详细分析结果
    filter_statistics: Optional[Dict[str, Any]] = None   # 过滤统计
    annotation_statistics: Optional[Dict[str, Any]] = None  # 标注统计
    table_analyses: Optional[List[Dict]] = None          # 表格分析
    chart_analyses: Optional[List[Dict]] = None          # 图表分析

    # 时间戳
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'basic_info': {
                'success': self.success,
                'error_message': self.error_message,
                'parse_time': self.parse_time,
                'analysis_timestamp': self.analysis_timestamp
            },
            'content': self.content,
            'metadata': self.metadata,
            'semantic_structure': self.semantic_structure,
            'content_quality': self.content_quality,
            'cross_references': self.cross_references,
            'insights': self.insights,
            'validation_results': self.validation_results,
            'filter_statistics': self.filter_statistics,
            'annotation_statistics': self.annotation_statistics,
            'table_analyses': self.table_analyses,
            'chart_analyses': self.chart_analyses
        }

    def get_summary(self) -> str:
        """获取结果摘要"""
        summary = f"""
=== 文档分析摘要 ===

基础信息:
  状态: {'成功' if self.success else '失败'}
  解析时间: {self.parse_time:.2f}秒
  分析时间: {self.analysis_timestamp}

"""

        if self.content_quality:
            summary += f"内容质量:\n"
            for key, value in self.content_quality.items():
                summary += f"  {key}: {value}\n"

        if self.validation_results:
            overall_score = self.validation_results.get('overall_consistency_score', 'N/A')
            overall_rating = self.validation_results.get('overall_rating', 'N/A')
            summary += f"\n一致性验证:\n"
            summary += f"  总体评分: {overall_score}\n"
            summary += f"  总体评级: {overall_rating}\n"

        if self.insights:
            summary += f"\n核心洞察:\n"
            key_findings = self.insights.get('key_findings', [])
            for i, finding in enumerate(key_findings[:3], 1):
                summary += f"  {i}. {finding}\n"

        return summary

    def get_json_report(self) -> str:
        """获取JSON格式的完整报告"""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class DocumentOverview:
    """文档概览"""
    basic_info: Dict[str, Any]
    content_distribution: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    processing_time: str


@dataclass
class CoreInsights:
    """核心洞察"""
    key_findings: List[Dict[str, Any]]
    data_insights: Dict[str, Any]
    action_items: List[str]
