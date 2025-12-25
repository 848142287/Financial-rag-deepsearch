"""
Agentic RAG生成阶段
基于高质量检索结果生成可靠答案
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
from datetime import datetime

from app.services.llm_service import LLMService
from app.services.progress_tracker import progress_tracker, TaskStatus, TaskStep

logger = logging.getLogger(__name__)


class AnswerTemplate(Enum):
    """答案模板类型"""
    FACTUAL = "factual"              # 事实型答案
    COMPARISON = "comparison"        # 比较型答案
    ANALYSIS = "analysis"           # 分析型答案
    PREDICTION = "prediction"       # 预测型答案
    CAUSAL = "causal"              # 因果型答案
    LIST = "list"                  # 列表型答案
    DEFINITION = "definition"       # 定义型答案


@dataclass
class GenerationConstraints:
    """生成约束"""
    max_length: int = 2000
    min_length: int = 100
    compliance_level: str = "strict"  # strict, moderate, relaxed
    format_type: str = "paragraph"    # paragraph, bullet_points, table
    language: str = "zh-CN"
    style: str = "professional"       # professional, casual, academic
    include_sources: bool = True
    include_confidence: bool = False


@dataclass
class GeneratedAnswer:
    """生成的答案"""
    task_id: str
    plan_id: str
    answer: str
    template_used: AnswerTemplate
    constraints: GenerationConstraints
    sources: List[Dict[str, Any]]
    confidence_score: float
    factual_score: float
    compliance_score: float
    metadata: Dict[str, Any]
    generation_time_ms: float
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['template_used'] = self.template_used.value
        data['created_at'] = self.created_at.isoformat()
        return data


class GenerationPhase:
    """生成阶段处理器"""

    def __init__(self):
        self.llm_service = LLMService()

        # 答案模板
        self.templates = {
            AnswerTemplate.FACTUAL: """
请基于以下信息回答问题：

问题：{query}

参考信息：
{context}

请提供一个准确、简洁的答案。答案应该：
1. 直接回答问题
2. 基于提供的参考信息
3. 语言清晰、专业
4. 包含关键数据和事实

答案：
""",

            AnswerTemplate.COMPARISON: """
请基于以下信息进行对比分析：

问题：{query}

参考信息：
{context}

请提供一个结构化的对比分析，包括：
1. 对比的主要维度
2. 各项的具体差异
3. 优缺点分析
4. 总结性结论

答案：
""",

            AnswerTemplate.ANALYSIS: """
请基于以下信息进行深入分析：

问题：{query}

参考信息：
{context}

请提供一个全面的分析报告，包括：
1. 关键要素识别
2. 深入的分析和解读
3. 影响因素探讨
4. 结论和建议

答案：
""",

            AnswerTemplate.PREDICTION: """
请基于以下信息进行预测分析：

问题：{query}

参考信息：
{context}

请提供预测性分析，包括：
1. 历史趋势总结
2. 关键影响因素
3. 未来趋势预测
4. 风险提示和不确定性说明

答案：
""",

            AnswerTemplate.CAUSAL: """
请基于以下信息分析因果关系：

问题：{query}

参考信息：
{context}

请提供因果分析，包括：
1. 直接原因识别
2. 间接原因分析
3. 影响机制说明
4. 结果和后果

答案：
""",

            AnswerTemplate.LIST: """
请基于以下信息列出相关项目：

问题：{query}

参考信息：
{context}

请提供一个清晰的列表，包括：
1. 逐项列出
2. 每项的简要说明
3. 相关数据或特征

答案：
""",

            AnswerTemplate.DEFINITION: """
请基于以下信息提供定义解释：

问题：{query}

参考信息：
{context}

请提供清晰的定义，包括：
1. 核心概念解释
2. 关键特征描述
3. 相关背景信息
4. 实例或应用场景

答案：
"""
        }

        # 合规性规则
        self.compliance_rules = {
            'strict': [
                "答案必须基于提供的事实信息",
                "不包含未经证实的推测",
                "必须标明信息来源",
                "不提供投资建议"
            ],
            'moderate': [
                "答案主要基于事实信息",
                "可以有合理的推断",
                "关键信息需要来源",
                "避免明确的投资建议"
            ],
            'relaxed': [
                "答案基于相关信息",
                "可以包含专业见解",
                "保持客观中立"
            ]
        }

        # 事实性检查规则
        self.factual_check_rules = {
            'must_have_evidence': "答案中的关键事实必须有证据支持",
            'no_contradiction': "答案不能与提供的参考信息矛盾",
            'verify_numbers': "所有数据应与参考信息一致",
            'source_clarity': "来源信息必须清晰明确"
        }

    async def generate_answer(
        self,
        plan,
        fused_result,
        task_id: Optional[str] = None
    ) -> GeneratedAnswer:
        """
        生成答案

        Args:
            plan: 检索计划
            fused_result: 融合后的检索结果
            task_id: 任务ID

        Returns:
            生成的答案
        """
        start_time = datetime.now()

        try:
            logger.info(f"Generating answer for plan: {plan.task_id}")

            # 更新进度
            if task_id:
                await progress_tracker.update_progress(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    current_step=TaskStep.SYNTHESIS,
                    progress_percentage=75.0,
                    message="准备生成答案"
                )

            # 1. 上下文准备
            context = await self._prepare_context(fused_result, plan)

            # 2. 模板选择
            template = self._select_template(plan.query_type)

            # 3. 约束设置
            constraints = self._set_constraints(plan, context)

            # 4. LLM生成答案
            raw_answer = await self._generate_with_llm(
                plan.processed_query,
                context,
                template,
                constraints
            )

            # 5. 事实性检查
            checked_answer, factual_score = await self._factual_check(
                raw_answer, fused_result
            )

            # 6. 合规性检查
            compliant_answer, compliance_score = await self._compliance_check(
                checked_answer, constraints
            )

            # 7. 格式化输出
            final_answer = await self._format_output(
                compliant_answer, constraints
            )

            # 8. 计算置信度
            confidence_score = await self._calculate_confidence(
                final_answer, fused_result, factual_score, compliance_score
            )

            # 创建答案对象
            answer = GeneratedAnswer(
                task_id=task_id or "unknown",
                plan_id=plan.task_id,
                answer=final_answer,
                template_used=template,
                constraints=constraints,
                sources=fused_result.final_results[:5],  # 前5个来源
                confidence_score=confidence_score,
                factual_score=factual_score,
                compliance_score=compliance_score,
                metadata={
                    'context_length': len(context),
                    'template_name': template.value,
                    'retrieval_quality': fused_result.quality_score
                },
                generation_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                created_at=datetime.now()
            )

            # 更新进度
            if task_id:
                await progress_tracker.update_progress(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    current_step=TaskStep.FINALIZING,
                    progress_percentage=100.0,
                    message="答案生成完成",
                    details={
                        'answer_length': len(final_answer),
                        'confidence': confidence_score,
                        'sources_count': len(answer.sources)
                    }
                )

            logger.info(f"Answer generation completed with confidence: {confidence_score:.2f}")
            return answer

        except Exception as e:
            logger.error(f"Error in Generation Phase: {str(e)}")
            if task_id:
                await progress_tracker.fail_task(task_id, str(e))
            raise

    async def _prepare_context(
        self,
        fused_result,
        plan
    ) -> str:
        """准备上下文"""
        context_parts = []

        # 按相关性排序结果
        sorted_results = sorted(
            fused_result.final_results,
            key=lambda x: x.get('score', 0),
            reverse=True
        )

        # 选择最相关的结果
        max_context_items = min(plan.estimated_results, len(sorted_results))
        selected_results = sorted_results[:max_context_items]

        # 构建上下文
        for i, result in enumerate(selected_results):
            content = result.get('content', '')
            score = result.get('score', 0)
            sources = result.get('sources', [])

            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."

            context_part = f"[来源 {i+1}] (相关性: {score:.2f})\n{content}"
            if sources:
                context_part += f"\n来源: {', '.join(sources)}"

            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def _select_template(self, query_type) -> AnswerTemplate:
        """选择答案模板"""
        template_map = {
            'factual': AnswerTemplate.FACTUAL,
            'definition': AnswerTemplate.DEFINITION,
            'list': AnswerTemplate.LIST,
            'comparison': AnswerTemplate.COMPARISON,
            'causal': AnswerTemplate.CAUSAL,
            'trend_prediction': AnswerTemplate.PREDICTION,
            'analytical': AnswerTemplate.ANALYSIS,
            'evaluative': AnswerTemplate.ANALYSIS,
            'predictive': AnswerTemplate.PREDICTION
        }

        return template_map.get(query_type.value, AnswerTemplate.FACTUAL)

    def _set_constraints(self, plan, context: str) -> GenerationConstraints:
        """设置生成约束"""
        constraints = GenerationConstraints()

        # 根据复杂度调整长度
        if plan.complexity_level > 0.7:
            constraints.max_length = 3000
            constraints.min_length = 200
        elif plan.complexity_level < 0.3:
            constraints.max_length = 1000
            constraints.min_length = 50

        # 根据查询类型调整格式
        if plan.query_type.value in ['comparison', 'list']:
            constraints.format_type = 'bullet_points'
        elif plan.query_type.value == 'definition':
            constraints.format_type = 'paragraph'
        elif plan.query_type.value == 'analytical':
            constraints.format_type = 'paragraph'

        # 金融场景默认严格合规
        constraints.compliance_level = 'strict'
        constraints.include_sources = True
        constraints.include_confidence = plan.complexity_level > 0.5

        return constraints

    async def _generate_with_llm(
        self,
        query: str,
        context: str,
        template: AnswerTemplate,
        constraints: GenerationConstraints
    ) -> str:
        """使用LLM生成答案"""
        # 构建prompt
        template_str = self.templates[template]

        # 添加约束说明
        constraint_prompt = self._build_constraint_prompt(constraints)

        full_prompt = template_str.format(
            query=query,
            context=context
        ) + "\n\n" + constraint_prompt

        # 调用LLM
        response = await self.llm_service.generate_completion(
            prompt=full_prompt,
            max_tokens=min(constraints.max_length // 2, 2000),
            temperature=0.3,  # 较低的温度保证稳定性
            top_p=0.9
        )

        return response.strip()

    def _build_constraint_prompt(self, constraints: GenerationConstraints) -> str:
        """构建约束提示"""
        prompt_parts = []

        # 合规性约束
        compliance_rules = self.compliance_rules.get(constraints.compliance_level, [])
        if compliance_rules:
            prompt_parts.append("合规要求：")
            for rule in compliance_rules:
                prompt_parts.append(f"- {rule}")

        # 格式约束
        format_instructions = {
            'paragraph': "请以段落形式回答，保持逻辑清晰",
            'bullet_points': "请使用项目符号（•）列出要点",
            'table': "请使用表格形式呈现对比信息"
        }
        if constraints.format_type in format_instructions:
            prompt_parts.append(f"格式要求：{format_instructions[constraints.format_type]}")

        # 长度约束
        if constraints.max_length:
            prompt_parts.append(f"答案长度不超过 {constraints.max_length} 字符")

        # 来源要求
        if constraints.include_sources:
            prompt_parts.append("请在答案中适当引用来源信息")

        return "\n".join(prompt_parts)

    async def _factual_check(
        self,
        answer: str,
        fused_result
    ) -> Tuple[str, float]:
        """事实性检查"""
        try:
            # 提取答案中的关键信息
            key_facts = self._extract_key_facts(answer)

            # 验证每个事实
            verified_facts = []
            for fact in key_facts:
                if self._verify_fact_in_context(fact, fused_result):
                    verified_facts.append(fact)

            # 计算事实性分数
            if key_facts:
                factual_score = len(verified_facts) / len(key_facts)
            else:
                factual_score = 1.0  # 没有关键事实，默认满分

            # 如果分数过低，标记需要人工审核
            if factual_score < 0.7:
                answer = self._add_factual_warning(answer)

            return answer, factual_score

        except Exception as e:
            logger.error(f"Factual check failed: {str(e)}")
            return answer, 0.5

    async def _compliance_check(
        self,
        answer: str,
        constraints: GenerationConstraints
    ) -> Tuple[str, float]:
        """合规性检查"""
        try:
            issues = []

            # 检查是否包含投资建议
            if constraints.compliance_level == 'strict':
                investment_keywords = ['建议', '推荐', '买入', '卖出', '持有', 'should', 'recommend']
                if any(keyword in answer for keyword in investment_keywords):
                    issues.append("包含潜在的投资建议")

            # 检查是否包含未经证实的推测
            speculation_keywords = ['可能', '或许', '大概', 'probably', 'maybe']
            if constraints.compliance_level == 'strict' and any(keyword in answer for keyword in speculation_keywords):
                issues.append("包含未经证实的推测")

            # 计算合规分数
            if issues:
                compliance_score = max(0.5, 1.0 - len(issues) * 0.2)
                # 添加合规警告
                answer = self._add_compliance_warning(answer, issues)
            else:
                compliance_score = 1.0

            return answer, compliance_score

        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}")
            return answer, 0.5

    async def _format_output(
        self,
        answer: str,
        constraints: GenerationConstraints
    ) -> str:
        """格式化输出"""
        # 添加来源引用
        if constraints.include_sources:
            answer = self._add_source_citations(answer)

        # 添加置信度信息
        if constraints.include_confidence:
            answer = self._add_confidence_note(answer)

        # 清理格式
        answer = self._clean_format(answer)

        return answer

    async def _calculate_confidence(
        self,
        answer: str,
        fused_result,
        factual_score: float,
        compliance_score: float
    ) -> float:
        """计算置信度"""
        # 检索质量权重
        retrieval_weight = 0.3
        factual_weight = 0.4
        compliance_weight = 0.3

        confidence = (
            fused_result.quality_score * retrieval_weight +
            factual_score * factual_weight +
            compliance_score * compliance_weight
        )

        return min(1.0, max(0.0, confidence))

    def _extract_key_facts(self, text: str) -> List[str]:
        """提取关键事实"""
        # 简单的事实提取：识别数字、百分比、日期等
        facts = []

        # 提取百分比
        percentages = re.findall(r'\d+\.?\d*%', text)
        facts.extend(percentages)

        # 提取金额
        amounts = re.findall(r'[\d,]+\.?\d*\s*(万|亿|千万)?元', text)
        facts.extend(amounts)

        # 提取日期
        dates = re.findall(r'\d{4}年|\d{1,2}月\d{1,2}日', text)
        facts.extend(dates)

        # 提取公司/股票代码
        companies = re.findall(r'[A-Z]{2,}', text)
        facts.extend(companies)

        return list(set(facts))  # 去重

    def _verify_fact_in_context(self, fact: str, fused_result) -> bool:
        """在上下文中验证事实"""
        context_text = ' '.join([
            r.get('content', '') for r in fused_result.final_results
        ])
        return fact in context_text

    def _add_factual_warning(self, answer: str) -> str:
        """添加事实性警告"""
        return answer + "\n\n[注意：部分信息需要进一步核实]"

    def _add_compliance_warning(self, answer: str, issues: List[str]) -> str:
        """添加合规警告"""
        warning = "\n\n[合规提示：" + "; ".join(issues) + "]"
        return answer + warning

    def _add_source_citations(self, answer: str) -> str:
        """添加来源引用"""
        # 简化的来源添加
        return answer + "\n\n来源：基于公开的金融研究报告和数据"

    def _add_confidence_note(self, answer: str) -> str:
        """添加置信度说明"""
        return answer + "\n\n注：以上分析基于现有信息，实际情况可能有所不同"

    def _clean_format(self, text: str) -> str:
        """清理格式"""
        # 移除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除首尾空格
        text = text.strip()
        return text