"""
Agentic RAG 生成阶段
基于高质量检索结果生成可靠答案
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re

from .planner import QueryType, QueryComplexity
from .executor import ExecutionResult, FusedResult

logger = logging.getLogger(__name__)


class GenerationTemplate(Enum):
    """生成模板类型"""
    FACT_FINDING = "fact_finding"
    COMPARISON_ANALYSIS = "comparison_analysis"
    TREND_PREDICTION = "trend_prediction"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    COMPREHENSIVE_RESEARCH = "comprehensive_research"
    DEFINITION_EXPLANATION = "definition_explanation"


@dataclass
class GenerationContext:
    """生成上下文"""
    query: str
    query_type: QueryType
    query_complexity: QueryComplexity
    fused_results: List[FusedResult]
    constraints: Dict[str, Any]
    domain: str = "general"


@dataclass
class FactCheckResult:
    """事实检查结果"""
    is_factual: bool
    confidence: float
    issues: List[str]
    corrections: List[str]


@dataclass
class GenerationResult:
    """生成结果"""
    answer: str
    sources: List[str]
    fact_check: FactCheckResult
    metadata: Dict[str, Any]
    generation_time: float
    template_used: str
    quality_score: float


class AgenticRAGGenerator:
    """Agentic RAG 生成器"""

    def __init__(self):
        # 初始化模板
        self.templates = self._initialize_templates()

        # 金融特定约束
        self.financial_constraints = {
            "required_disclaimers": [
                "投资有风险，入市需谨慎",
                "以上分析仅供参考，不构成投资建议"
            ],
            "compliance_checks": [
                "no_price_predictions",
                "no_investment_advice",
                "include_risk_warning"
            ]
        }

        # 事实检查规则
        self.fact_check_rules = {
            "numeric_consistency": True,
            "source_citation": True,
            "logical_consistency": True,
            "compliance_check": True
        }

    def _initialize_templates(self) -> Dict[GenerationTemplate, str]:
        """初始化生成模板"""
        templates = {
            GenerationTemplate.FACT_FINDING: """
基于以下检索结果，回答用户的问题。请确保答案准确、简洁，并引用信息来源。

用户问题：{query}

检索结果：
{context}

请提供：
1. 直接答案
2. 支持数据
3. 信息来源

答案：
""",

            GenerationTemplate.COMPARISON_ANALYSIS: """
基于以下检索结果，进行详细的比较分析。请客观地比较不同对象的特点。

用户问题：{query}

检索结果：
{context}

请提供结构化的比较分析：
1. 比较维度
2. 各对象特点
3. 优缺点分析
4. 综合评价

比较分析：
""",

            GenerationTemplate.TREND_PREDICTION: """
基于以下历史数据和趋势信息，进行趋势分析预测。请注意区分事实和预测。

用户问题：{query}

检索结果：
{context}

请提供：
1. 历史趋势回顾
2. 当前状况分析
3. 未来趋势预测（需明确标注预测性质）
4. 影响因素分析
5. 相关风险提示

趋势分析：
""",

            GenerationTemplate.RELATIONSHIP_ANALYSIS: """
基于以下信息，分析各个要素之间的关系和影响。

用户问题：{query}

检索结果：
{context}

请提供：
1. 主要关系识别
2. 影响机制分析
3. 相互作用说明
4. 因果关系梳理

关系分析：
""",

            GenerationTemplate.COMPREHENSIVE_RESEARCH: """
基于以下全面的研究资料，进行深入的综合分析。

用户问题：{query}

检索结果：
{context}

请提供全面深入的分析报告：
1. 研究背景
2. 核心问题分析
3. 多角度论证
4. 数据支撑
5. 结论和建议
6. 研究局限说明

综合分析：
""",

            GenerationTemplate.DEFINITION_EXPLANATION: """
基于以下资料，详细解释相关概念和定义。

用户问题：{query}

检索结果：
{context}

请提供：
1. 概念定义
2. 核心特征
3. 应用场景
4. 相关示例
5. 注意事项

解释说明：
"""
        }

        return templates

    async def generate_answer(self, execution_result: ExecutionResult, plan_context: Dict[str, Any]) -> GenerationResult:
        """
        生成答案

        Args:
            execution_result: 执行结果
            plan_context: 计划上下文

        Returns:
            GenerationResult: 生成结果
        """
        start_time = time.time()

        try:
            logger.info(f"开始生成答案，计划ID: {execution_result.plan_id}")

            # 1. 准备生成上下文
            generation_context = self._prepare_generation_context(execution_result, plan_context)

            # 2. 选择生成模板
            template = self._select_template(generation_context)

            # 3. 构建上下文
            formatted_context = self._build_context(generation_context.fused_results)

            # 4. 设置约束
            constraints = self._set_constraints(generation_context)

            # 5. 生成答案
            raw_answer = await self._generate_raw_answer(template, generation_context.query, formatted_context)

            # 6. 事实检查
            fact_check = await self._perform_fact_check(raw_answer, generation_context)

            # 7. 格式化输出
            formatted_answer = self._format_output(raw_answer, generation_context, fact_check)

            # 8. 质量评估
            quality_score = self._assess_quality(formatted_answer, generation_context, fact_check)

            generation_time = time.time() - start_time

            result = GenerationResult(
                answer=formatted_answer,
                sources=[r.source for r in generation_context.fused_results],
                fact_check=fact_check,
                metadata={
                    "template": template.name,
                    "constraints": constraints,
                    "context_count": len(generation_context.fused_results),
                    "domain": generation_context.domain
                },
                generation_time=generation_time,
                template_used=template.name,
                quality_score=quality_score
            )

            logger.info(f"答案生成完成，质量分数: {quality_score:.2f}")
            return result

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"答案生成失败: {str(e)}")

            return GenerationResult(
                answer="抱歉，答案生成过程中出现错误，请稍后重试。",
                sources=[],
                fact_check=FactCheckResult(is_factual=False, confidence=0.0, issues=["生成失败"], corrections=[]),
                metadata={"error": str(e)},
                generation_time=generation_time,
                template_used="error",
                quality_score=0.0
            )

    def _prepare_generation_context(self, execution_result: ExecutionResult, plan_context: Dict[str, Any]) -> GenerationContext:
        """准备生成上下文"""
        query_analysis = plan_context.get("query_analysis")

        return GenerationContext(
            query=execution_result.original_query,
            query_type=query_analysis.query_type if query_analysis else QueryType.FACT_FINDING,
            query_complexity=query_analysis.complexity if query_analysis else QueryComplexity.SIMPLE,
            fused_results=execution_result.fused_results,
            constraints=plan_context.get("constraints", {}),
            domain=query_analysis.domain if query_analysis else "general"
        )

    def _select_template(self, context: GenerationContext) -> GenerationTemplate:
        """选择生成模板"""
        template_mapping = {
            QueryType.FACT_FINDING: GenerationTemplate.FACT_FINDING,
            QueryType.COMPARISON_ANALYSIS: GenerationTemplate.COMPARISON_ANALYSIS,
            QueryType.TREND_PREDICTION: GenerationTemplate.TREND_PREDICTION,
            QueryType.RELATIONSHIP_ANALYSIS: GenerationTemplate.RELATIONSHIP_ANALYSIS,
            QueryType.COMPREHENSIVE_RESEARCH: GenerationTemplate.COMPREHENSIVE_RESEARCH,
            QueryType.DEFINITION_EXPLANATION: GenerationTemplate.DEFINITION_EXPLANATION
        }

        return template_mapping.get(context.query_type, GenerationTemplate.FACT_FINDING)

    def _build_context(self, fused_results: List[FusedResult]) -> str:
        """构建上下文"""
        context_parts = []

        for i, result in enumerate(fused_results, 1):
            context_part = f"""
【信息来源 {i}】
来源：{', '.join(result.sources)}
可信度：{result.overall_score:.2f}
内容：{result.content}
方法贡献：{result.method_contributions}
"""
            context_parts.append(context_part)

        return '\n'.join(context_parts)

    def _set_constraints(self, context: GenerationContext) -> Dict[str, Any]:
        """设置约束条件"""
        constraints = context.constraints.copy()

        # 金融领域特殊约束
        if context.domain == "financial":
            constraints["compliance_required"] = True
            constraints["risk_warning_required"] = True
            constraints["no_price_prediction"] = True
            constraints["no_investment_advice"] = True

        # 根据复杂度调整
        if context.query_complexity == QueryComplexity.COMPLEX:
            constraints["max_length"] = 2000
            constraints["require_detailed_analysis"] = True
        elif context.query_complexity == QueryComplexity.SIMPLE:
            constraints["max_length"] = 500
            constraints["require_concise_answer"] = True

        return constraints

    async def _generate_raw_answer(self, template: GenerationTemplate, query: str, context: str) -> str:
        """生成原始答案"""
        try:
            # 这里应该调用LLM
            # 由于依赖问题，使用模拟实现
            return self._simulate_llm_generation(template, query, context)

        except Exception as e:
            logger.error(f"LLM生成失败: {str(e)}")
            raise

    def _simulate_llm_generation(self, template: GenerationTemplate, query: str, context: str) -> str:
        """模拟LLM生成"""
        # 简单的模拟实现
        template_str = self.templates.get(template, self.templates[GenerationTemplate.FACT_FINDING])

        # 基于查询类型生成不同类型的答案
        if template == GenerationTemplate.FACT_FINDING:
            answer = f"基于检索到的信息，{query}的答案是：根据相关数据显示..."
        elif template == GenerationTemplate.COMPARISON_ANALYSIS:
            answer = f"经过比较分析，各对象的主要差异和特点如下：\n1. 特点对比...\n2. 性能比较..."
        elif template == GenerationTemplate.TREND_PREDICTION:
            answer = f"基于历史数据和当前趋势分析，未来发展趋势预测如下：\n注意：以下为趋势分析预测，投资有风险..."
        else:
            answer = f"基于检索信息，针对{query}的分析如下：\n1. 主要发现...\n2. 详细分析..."

        return answer[:800]  # 限制长度

    async def _perform_fact_check(self, answer: str, context: GenerationContext) -> FactCheckResult:
        """执行事实检查"""
        issues = []
        corrections = []

        # 数值一致性检查
        numeric_issues = self._check_numeric_consistency(answer, context.fused_results)
        issues.extend(numeric_issues)

        # 来源引用检查
        citation_issues = self._check_citation_consistency(answer, context.fused_results)
        issues.extend(citation_issues)

        # 逻辑一致性检查
        logic_issues = self._check_logical_consistency(answer)
        issues.extend(logic_issues)

        # 合规性检查
        compliance_issues = self._check_compliance(answer, context)
        issues.extend(compliance_issues)

        is_factual = len(issues) == 0
        confidence = max(0.1, 1.0 - len(issues) * 0.2)  # 简单的置信度计算

        return FactCheckResult(
            is_factual=is_factual,
            confidence=confidence,
            issues=issues,
            corrections=corrections
        )

    def _check_numeric_consistency(self, answer: str, results: List[FusedResult]) -> List[str]:
        """检查数值一致性"""
        issues = []
        # 提取答案中的数值
        answer_numbers = re.findall(r'\d+(?:\.\d+)?%?|\d+(?:,\d{3})*', answer)

        # 简化实现：检查是否有明显的数值错误
        for num_str in answer_numbers:
            try:
                num_value = float(num_str.replace('%', '').replace(',', ''))
                if num_value > 10000 and '%' not in num_str:  # 检查异常大的数值
                    issues.append(f"数值 {num_str} 可能存在异常")
            except ValueError:
                issues.append(f"数值格式错误: {num_str}")

        return issues

    def _check_citation_consistency(self, answer: str, results: List[FusedResult]) -> List[str]:
        """检查引用一致性"""
        issues = []
        source_count = len(set(r.source for r in results))

        if source_count == 0:
            issues.append("缺少信息来源引用")
        elif "来源" not in answer and "参考" not in answer and "数据来源" not in answer:
            issues.append("答案中未明确引用信息来源")

        return issues

    def _check_logical_consistency(self, answer: str) -> List[str]:
        """检查逻辑一致性"""
        issues = []

        # 检查矛盾表述
        if "上涨" in answer and "下跌" in answer:
            # 简化的矛盾检测
            if answer.count("上涨") > 3 and answer.count("下跌") > 3:
                issues.append("可能存在矛盾表述")

        return issues

    def _check_compliance(self, answer: str, context: GenerationContext) -> List[str]:
        """检查合规性"""
        issues = []

        if context.domain == "financial":
            # 检查是否包含投资建议
            if any(word in answer for word in ["建议购买", "推荐买入", "强烈建议", "必买"]):
                issues.append("包含直接投资建议")

            # 检查是否有风险提示
            if "风险" not in answer and "谨慎" not in answer and "仅供参考" not in answer:
                issues.append("缺少风险提示")

        return issues

    def _format_output(self, answer: str, context: GenerationContext, fact_check: FactCheckResult) -> str:
        """格式化输出"""
        # 基本格式化
        formatted_answer = answer.strip()

        # 添加来源信息
        if context.fused_results:
            sources = list(set(r.source for r in context.fused_results))
            source_text = f"\n\n信息来源：{', '.join(sources[:3])}"  # 最多显示3个来源
            formatted_answer += source_text

        # 添加合规信息（金融领域）
        if context.domain == "financial":
            disclaimer = "\n\n风险提示：以上信息仅供参考，不构成投资建议，投资有风险，入市需谨慎。"
            formatted_answer += disclaimer

        # 添加质量信息（如果有问题）
        if fact_check.issues:
            quality_note = f"\n\n注意：本回答可能存在{len(fact_check.issues)}个需要关注的问题。"
            formatted_answer += quality_note

        return formatted_answer

    def _assess_quality(self, answer: str, context: GenerationContext, fact_check: FactCheckResult) -> float:
        """评估答案质量"""
        quality_score = 0.5  # 基础分数

        # 长度合理性
        if 100 <= len(answer) <= 2000:
            quality_score += 0.1

        # 事实检查结果
        quality_score += fact_check.confidence * 0.3

        # 来源引用
        if "来源" in answer or "参考" in answer:
            quality_score += 0.1

        # 结构完整性
        if any(marker in answer for marker in ["1.", "2.", "3.", "首先", "其次", "最后"]):
            quality_score += 0.1

        # 领域特定质量
        if context.domain == "financial":
            if "风险" in answer:
                quality_score += 0.1
            if "数据" in answer or "统计" in answer:
                quality_score += 0.1

        return min(quality_score, 1.0)


# 全局生成器实例
rag_generator = AgenticRAGGenerator()


# 导入time模块（在文件开头）
import time