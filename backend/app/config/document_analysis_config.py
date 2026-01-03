"""
文档分析配置
基于提示词规范的配置化方案
"""

from pydantic import BaseModel, Field

class SemanticUnitConfig(BaseModel):
    """语义单元配置"""
    enabled: bool = Field(default=True, description="是否启用语义单元标注")
    min_confidence: float = Field(default=0.3, description="最低置信度阈值")
    min_paragraph_length: int = Field(default=10, description="最小段落长度")

class ChartAnalysisConfig(BaseModel):
    """图表分析配置"""
    enabled_deep_analysis: bool = Field(default=True, description="是否启用深度图表分析")
    analyze_flowchart: bool = Field(default=True, description="是否分析流程图")
    analyze_axes: bool = Field(default=True, description="是否详细分析坐标轴")
    detect_inflection_points: bool = Field(default=True, description="是否检测拐点")
    inflection_threshold: float = Field(default=0.1, description="拐点检测阈值")

class FormulaConfig(BaseModel):
    """公式分析配置"""
    enabled_interpreter: bool = Field(default=True, description="是否启用公式解释器")
    extract_variables: bool = Field(default=True, description="是否提取变量")
    mathematical_meaning: bool = Field(default=True, description="是否分析数学意义")
    application_interpretation: bool = Field(default=True, description="是否解读应用场景")

class TableAnalysisConfig(BaseModel):
    """表格分析配置"""
    generate_comprehensive_insights: bool = Field(default=True, description="是否生成综合洞察")
    dimension_analysis: bool = Field(default=True, description="是否分析维度结构")
    pattern_recognition: bool = Field(default=True, description="是否识别模式")
    decision_support: bool = Field(default=True, description="是否提供决策支持")

class QualityCheckConfig(BaseModel):
    """质量检查配置"""
    text_accuracy_min: float = Field(default=0.95, description="OCR准确率最低要求")
    data_consistency_required: bool = Field(default=True, description="是否要求数据一致性")
    check_hyperlinks: bool = Field(default=True, description="是否检查超链接有效性")
    hierarchy_correctness: bool = Field(default=True, description="是否检查层次结构正确性")

class ReportFormatConfig(BaseModel):
    """报告格式配置"""
    template: str = Field(default="comprehensive_analysis", description="报告模板类型")
    include_overview: bool = Field(default=True, description="是否包含概览")
    include_structure: bool = Field(default=True, description="是否包含结构导航")
    include_deep_analysis: bool = Field(default=True, description="是否包含深度分析")
    include_core_insights: bool = Field(default=True, description="是否包含核心洞察")
    format_output: str = Field(default="markdown", description="输出格式: markdown, json, html")

class DocumentAnalysisConfig(BaseModel):
    """文档分析总配置"""
    
    # 功能开关
    semantic_unit: SemanticUnitConfig = Field(default_factory=SemanticUnitConfig)
    chart_analysis: ChartAnalysisConfig = Field(default_factory=ChartAnalysisConfig)
    formula: FormulaConfig = Field(default_factory=FormulaConfig)
    table_analysis: TableAnalysisConfig = Field(default_factory=TableAnalysisConfig)
    quality_check: QualityCheckConfig = Field(default_factory=QualityCheckConfig)
    report_format: ReportFormatConfig = Field(default_factory=ReportFormatConfig)
    
    # 语义单元类型定义
    semantic_unit_types: Dict[str, str] = Field(
        default={
            "discourse": "论述段落",
            "data_statement": "数据陈述",
            "case_description": "案例描述",
            "definition": "定义说明",
            "conclusion": "结论总结",
            "recommendation": "建议措施"
        },
        description="语义单元类型映射"
    )
    
    # 逻辑关系类型
    logical_relation_types: Dict[str, str] = Field(
        default={
            "causal": "因果关系",
            "parallel": "并列关系",
            "contrast": "对比关系",
            "supplementary": "补充关系",
            "sequential": "顺序关系"
        },
        description="逻辑关系类型映射"
    )
    
    # 图像分析阈值
    image_confidence_threshold: float = Field(default=0.85, description="图像识别置信度阈值")
    ocr_accuracy_threshold: float = Field(default=0.95, description="OCR准确率阈值")
    
    # 智能过滤规则
    filter_rules: List[str] = Field(
        default=[
            "duplicate_headers_footers",
            "page_numbers",
            "print_markers",
            "decorative_images",
            "blank_lines",
            "template_text"
        ],
        description="智能过滤规则列表"
    )
    
    # 图表分析配置
    financial_metrics_enabled: bool = Field(default=True, description="是否启用金融专业指标")
    advanced_chart_insights_enabled: bool = Field(default=True, description="是否启用高级图表洞察")
    
    # 关联分析配置
    enable_logic_proof_chain: bool = Field(default=True, description="是否启用逻辑证明链")
    enable_visual_reinforcement_chain: bool = Field(default=True, description="是否启用视觉强化链")
    
    # 智能摘要配置
    enable_intelligent_summary: bool = Field(default=True, description="是否启用智能摘要")
    max_core_arguments: int = Field(default=3, description="核心论点最大数量")
    max_key_evidence: int = Field(default=5, description="关键证据最大数量")
    
    class Config:
        arbitrary_types_allowed = True

# 默认配置实例
default_config = DocumentAnalysisConfig()

def get_config() -> DocumentAnalysisConfig:
    """获取文档分析配置"""
    return default_config

def get_custom_config(**kwargs) -> DocumentAnalysisConfig:
    """
    获取自定义配置
    
    Args:
        **kwargs: 配置参数覆盖
    
    Returns:
        自定义配置实例
    """
    config_dict = default_config.dict()
    config_dict.update(kwargs)
    return DocumentAnalysisConfig(**config_dict)
