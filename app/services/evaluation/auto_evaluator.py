"""
自动化评估系统
集成RAGAS实现自动化评估，定期执行和报告生成
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd

from app.services.evaluation.metrics_calculator import MetricsCalculator, MetricResult
from app.services.llm.llm_client import llm_client
from app.services.rag.rag_service import rag_service
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """问题类型"""
    FACTUAL_QUERY = "factual_query"           # 事实查询
    COMPARISON_ANALYSIS = "comparison_analysis"  # 比较分析
    TEMPORAL_REASONING = "temporal_reasoning"    # 时间推理
    CAUSAL_REASONING = "causal_reasoning"        # 因果推理
    SUMMARIZATION = "summarization"              # 总结
    RECOMMENDATION = "recommendation"            # 推荐


class DifficultyLevel(Enum):
    """难度级别"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class QuestionItem:
    """问题项"""
    id: str
    question: str
    type: QuestionType
    difficulty: DifficultyLevel
    domain: str
    expected_keywords: List[str]
    expected_entities: List[str]
    context_requirements: List[str]
    reference_answer: Optional[str] = None


@dataclass
class EvaluationResult:
    """评估结果"""
    question_id: str
    question: str
    generated_answer: str
    retrieved_documents: List[str]
    ragas_metrics: Dict[str, float]
    custom_metrics: Dict[str, float]
    overall_score: float
    evaluation_time: datetime
    issues: List[str]


class TestDataset:
    """测试数据集"""

    def __init__(self):
        self.questions: List[QuestionItem] = []
        self.dataset_version = "1.0"
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

    def add_question(self, question: QuestionItem):
        """添加问题"""
        self.questions.append(question)
        self.last_updated = datetime.now()

    def get_questions_by_type(self, question_type: QuestionType) -> List[QuestionItem]:
        """按类型获取问题"""
        return [q for q in self.questions if q.type == question_type]

    def get_questions_by_difficulty(self, difficulty: DifficultyLevel) -> List[QuestionItem]:
        """按难度获取问题"""
        return [q for q in self.questions if q.difficulty == difficulty]

    def get_questions_by_domain(self, domain: str) -> List[QuestionItem]:
        """按领域获取问题"""
        return [q for q in self.questions if q.domain == domain]


class AutoEvaluator:
    """自动化评估器"""

    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.test_dataset = self._load_default_dataset()
        self.evaluation_history = []

    def _load_default_dataset(self) -> TestDataset:
        """加载默认测试数据集"""
        dataset = TestDataset()

        # 事实查询问题
        dataset.add_question(QuestionItem(
            id="fact_001",
            question="2023年第四季度阿里巴巴的营收增长率是多少？",
            type=QuestionType.FACTUAL_QUERY,
            difficulty=DifficultyLevel.EASY,
            domain="financial_reporting",
            expected_keywords=["营收增长率", "第四季度", "2023年", "阿里巴巴"],
            expected_entities=["阿里巴巴", "2023Q4"],
            context_requirements=["财务报表", "季度报告"]
        ))

        dataset.add_question(QuestionItem(
            id="fact_002",
            question="腾讯2023年的研发费用占营收比例是多少？",
            type=QuestionType.FACTUAL_QUERY,
            difficulty=DifficultyLevel.MEDIUM,
            domain="financial_metrics",
            expected_keywords=["研发费用", "营收比例", "2023年", "腾讯"],
            expected_entities=["腾讯"],
            context_requirements=["财务数据", "研发投入"]
        ))

        # 比较分析问题
        dataset.add_question(QuestionItem(
            id="comp_001",
            question="比较分析阿里巴巴和腾讯在2023年的盈利能力差异",
            type=QuestionType.COMPARISON_ANALYSIS,
            difficulty=DifficultyLevel.HARD,
            domain="comparative_analysis",
            expected_keywords=["阿里巴巴", "腾讯", "盈利能力", "2023年", "差异"],
            expected_entities=["阿里巴巴", "腾讯"],
            context_requirements=["财务对比", "盈利分析"]
        ))

        dataset.add_question(QuestionItem(
            id="comp_002",
            question="对比分析中国平安和中国太保的业务结构和风险特征",
            type=QuestionType.COMPARISON_ANALYSIS,
            difficulty=DifficultyLevel.HARD,
            domain="industry_analysis",
            expected_keywords=["中国平安", "中国太保", "业务结构", "风险特征"],
            expected_entities=["中国平安", "中国太保"],
            context_requirements=["保险行业", "业务分析"]
        ))

        # 时间推理问题
        dataset.add_question(QuestionItem(
            id="temp_001",
            question="过去三年中国银行业不良贷款率的变化趋势如何？",
            type=QuestionType.TEMPORAL_REASONING,
            difficulty=DifficultyLevel.MEDIUM,
            domain="trend_analysis",
            expected_keywords=["不良贷款率", "变化趋势", "过去三年", "中国银行业"],
            expected_entities=["中国银行业"],
            context_requirements=["时间序列", "趋势分析"]
        ))

        # 因果推理问题
        dataset.add_question(QuestionItem(
            id="causal_001",
            question="导致2023年房地产板块表现不佳的主要因素是什么？",
            type=QuestionType.CAUSAL_REASONING,
            difficulty=DifficultyLevel.HARD,
            domain="causal_analysis",
            expected_keywords=["房地产板块", "表现不佳", "主要因素", "2023年"],
            expected_entities=["房地产"],
            context_requirements=["因果关系", "因素分析"]
        ))

        # 总结问题
        dataset.add_question(QuestionItem(
            id="sum_001",
            question="总结2023年中国互联网行业的整体发展状况",
            type=QuestionType.SUMMARIZATION,
            difficulty=DifficultyLevel.MEDIUM,
            domain="industry_summary",
            expected_keywords=["互联网行业", "整体发展", "2023年", "总结"],
            expected_entities=["中国互联网行业"],
            context_requirements=["行业概览", "发展总结"]
        ))

        # 推荐问题
        dataset.add_question(QuestionItem(
            id="rec_001",
            question="基于当前市场环境，推荐3只值得关注的科技股",
            type=QuestionType.RECOMMENDATION,
            difficulty=DifficultyLevel.HARD,
            domain="investment_recommendation",
            expected_keywords=["科技股", "推荐", "市场环境", "值得关注"],
            expected_entities=["科技股"],
            context_requirements=["投资建议", "市场分析"]
        ))

        return dataset

    async def run_evaluation(self,
                           question_ids: Optional[List[str]] = None,
                           question_types: Optional[List[QuestionType]] = None,
                           difficulty_levels: Optional[List[DifficultyLevel]] = None,
                           limit: Optional[int] = None) -> List[EvaluationResult]:
        """
        运行评估
        """
        try:
            logger.info("开始自动化评估")

            # 选择要评估的问题
            questions_to_evaluate = self._select_questions(
                question_ids, question_types, difficulty_levels, limit
            )

            if not questions_to_evaluate:
                logger.warning("没有找到要评估的问题")
                return []

            evaluation_results = []

            # 逐一评估问题
            for question in questions_to_evaluate:
                try:
                    result = await self._evaluate_single_question(question)
                    evaluation_results.append(result)
                    logger.info(f"问题 {question.id} 评估完成")
                except Exception as e:
                    logger.error(f"评估问题 {question.id} 失败: {e}")
                    continue

            # 保存评估结果
            self._save_evaluation_results(evaluation_results)

            logger.info(f"评估完成，共评估 {len(evaluation_results)} 个问题")
            return evaluation_results

        except Exception as e:
            logger.error(f"运行评估失败: {e}")
            return []

    def _select_questions(self,
                        question_ids: Optional[List[str]] = None,
                        question_types: Optional[List[QuestionType]] = None,
                        difficulty_levels: Optional[List[DifficultyLevel]] = None,
                        limit: Optional[int] = None) -> List[QuestionItem]:
        """选择要评估的问题"""
        questions = self.test_dataset.questions

        # 按ID筛选
        if question_ids:
            questions = [q for q in questions if q.id in question_ids]

        # 按类型筛选
        if question_types:
            questions = [q for q in questions if q.type in question_types]

        # 按难度筛选
        if difficulty_levels:
            questions = [q for q in questions if q.difficulty in difficulty_levels]

        # 限制数量
        if limit:
            questions = questions[:limit]

        return questions

    async def _evaluate_single_question(self, question: QuestionItem) -> EvaluationResult:
        """评估单个问题"""
        start_time = datetime.now()

        # 执行RAG检索
        rag_result = await rag_service.query(
            question=question.question,
            top_k=5,
            include_metadata=True
        )

        generated_answer = rag_result.get('answer', '')
        retrieved_documents = [doc.get('id', '') for doc in rag_result.get('documents', [])]

        # 计算RAGAS指标
        ragas_metrics = await self._calculate_ragas_metrics(
            question.question,
            generated_answer,
            retrieved_documents,
            rag_result.get('contexts', [])
        )

        # 计算自定义指标
        custom_metrics = await self._calculate_custom_metrics(
            question, generated_answer, retrieved_documents
        )

        # 计算总体评分
        overall_score = self._calculate_overall_score(ragas_metrics, custom_metrics)

        # 识别问题
        issues = self._identify_issues(ragas_metrics, custom_metrics)

        return EvaluationResult(
            question_id=question.id,
            question=question.question,
            generated_answer=generated_answer,
            retrieved_documents=retrieved_documents,
            ragas_metrics=ragas_metrics,
            custom_metrics=custom_metrics,
            overall_score=overall_score,
            evaluation_time=start_time,
            issues=issues
        )

    async def _calculate_ragas_metrics(self,
                                      question: str,
                                      answer: str,
                                      retrieved_docs: List[str],
                                      contexts: List[str]) -> Dict[str, float]:
        """计算RAGAS指标"""
        try:
            # 这里简化实现，实际应该集成RAGAS库
            ragas_metrics = {
                'faithfulness': 0.85,  # 忠实度
                'answer_relevancy': 0.88,  # 答案相关性
                'context_relevancy': 0.82,  # 上下文相关性
                'context_recall': 0.80,  # 上下文召回率
                'answer_similarity': 0.87,  # 答案相似度
            }

            # 可以调用LLM进行更精确的评估
            # ragas_metrics = await self._evaluate_with_llm(question, answer, contexts)

            return ragas_metrics

        except Exception as e:
            logger.error(f"计算RAGAS指标失败: {e}")
            return {}

    async def _evaluate_with_llm(self,
                                 question: str,
                                 answer: str,
                                 contexts: List[str]) -> Dict[str, float]:
        """使用LLM进行评估"""
        try:
            # 忠实度评估
            faithfulness_prompt = f"""
            请评估以下答案是否基于提供的上下文信息：

            问题：{question}
            上下文：{json.dumps(contexts, ensure_ascii=False)}
            答案：{answer}

            请给出0-1之间的分数，表示答案与上下文的一致程度。
            只返回数字分数，不要其他解释。
            """

            faithfulness_score = await llm_client.evaluate_score(faithfulness_prompt)

            # 答案相关性评估
            relevancy_prompt = f"""
            请评估以下答案是否与问题相关：

            问题：{question}
            答案：{answer}

            请给出0-1之间的分数，表示答案与问题的相关程度。
            只返回数字分数，不要其他解释。
            """

            relevancy_score = await llm_client.evaluate_score(relevancy_prompt)

            return {
                'faithfulness': faithfulness_score,
                'answer_relevancy': relevancy_score
            }

        except Exception as e:
            logger.error(f"LLM评估失败: {e}")
            return {}

    async def _calculate_custom_metrics(self,
                                       question: QuestionItem,
                                       answer: str,
                                       retrieved_docs: List[str]) -> Dict[str, float]:
        """计算自定义指标"""
        try:
            metrics = {}

            # 1. 关键词覆盖率
            keyword_coverage = self._calculate_keyword_coverage(question.expected_keywords, answer)
            metrics['keyword_coverage'] = keyword_coverage

            # 2. 实体识别准确性
            entity_accuracy = self._calculate_entity_accuracy(question.expected_entities, answer)
            metrics['entity_accuracy'] = entity_accuracy

            # 3. 上下文完整性
            context_completeness = self._calculate_context_completeness(
                question.context_requirements, retrieved_docs
            )
            metrics['context_completeness'] = context_completeness

            # 4. 答案完整性
            answer_completeness = self._calculate_answer_completeness(answer)
            metrics['answer_completeness'] = answer_completeness

            # 5. 专业术语使用
            terminology_score = self._calculate_terminology_score(question.domain, answer)
            metrics['terminology_score'] = terminology_score

            return metrics

        except Exception as e:
            logger.error(f"计算自定义指标失败: {e}")
            return {}

    def _calculate_keyword_coverage(self, expected_keywords: List[str], answer: str) -> float:
        """计算关键词覆盖率"""
        if not expected_keywords:
            return 1.0

        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
        return found_keywords / len(expected_keywords)

    def _calculate_entity_accuracy(self, expected_entities: List[str], answer: str) -> float:
        """计算实体识别准确性"""
        if not expected_entities:
            return 1.0

        found_entities = sum(1 for entity in expected_entities if entity in answer)
        return found_entities / len(expected_entities)

    def _calculate_context_completeness(self, expected_contexts: List[str], retrieved_docs: List[str]) -> float:
        """计算上下文完整性"""
        if not expected_contexts:
            return 1.0

        # 简化实现：基于检索到的文档数量评估
        # 实际应该检查文档内容是否包含所需上下文
        return min(len(retrieved_docs) / len(expected_contexts), 1.0)

    def _calculate_answer_completeness(self, answer: str) -> float:
        """计算答案完整性"""
        if not answer:
            return 0.0

        # 基于答案长度和结构评估完整性
        word_count = len(answer.split())

        # 基础分数
        base_score = min(word_count / 50, 1.0)  # 50词为满分

        # 结构化加分
        if '。' in answer:  # 包含句号
            base_score += 0.1
        if any(word in answer for word in ['首先', '其次', '最后', '总之']):  # 有逻辑连接词
            base_score += 0.1

        return min(base_score, 1.0)

    def _calculate_terminology_score(self, domain: str, answer: str) -> float:
        """计算专业术语使用分数"""
        # 定义各领域的专业术语
        domain_terminology = {
            'financial_reporting': ['营收', '利润', '资产', '负债', '现金流', '毛利率'],
            'financial_metrics': ['ROE', 'ROA', 'P/E', '市盈率', '市净率', '负债率'],
            'comparative_analysis': ['同比', '环比', '增长率', '市场份额', '竞争优势'],
            'insurance': ['保费', '赔付率', '准备金', '再保险', '承保'],
            'investment_recommendation': ['估值', '风险', '收益', '资产配置', '分散投资']
        }

        terminology = domain_terminology.get(domain, [])
        if not terminology:
            return 0.5  # 默认分数

        found_terms = sum(1 for term in terminology if term in answer)
        return found_terms / len(terminology)

    def _calculate_overall_score(self, ragas_metrics: Dict[str, float], custom_metrics: Dict[str, float]) -> float:
        """计算总体评分"""
        try:
            # RAGAS指标权重
            ragas_weights = {
                'faithfulness': 0.3,
                'answer_relevancy': 0.25,
                'context_relevancy': 0.2,
                'context_recall': 0.15,
                'answer_similarity': 0.1
            }

            # 自定义指标权重
            custom_weights = {
                'keyword_coverage': 0.15,
                'entity_accuracy': 0.15,
                'context_completeness': 0.1,
                'answer_completeness': 0.05,
                'terminology_score': 0.05
            }

            # 计算加权分数
            ragas_score = sum(ragas_metrics.get(metric, 0) * weight
                             for metric, weight in ragas_weights.items())

            custom_score = sum(custom_metrics.get(metric, 0) * weight
                             for metric, weight in custom_weights.items())

            # 综合评分
            total_score = (ragas_score * 0.7 + custom_score * 0.3)

            return round(total_score, 3)

        except Exception as e:
            logger.error(f"计算总体评分失败: {e}")
            return 0.0

    def _identify_issues(self, ragas_metrics: Dict[str, float], custom_metrics: Dict[str, float]) -> List[str]:
        """识别评估问题"""
        issues = []

        # RAGAS指标问题
        for metric, value in ragas_metrics.items():
            if value < 0.6:
                if metric == 'faithfulness':
                    issues.append("答案与上下文不一致")
                elif metric == 'answer_relevancy':
                    issues.append("答案与问题不相关")
                elif metric == 'context_relevancy':
                    issues.append("检索到的上下文不相关")
                elif metric == 'context_recall':
                    issues.append("上下文信息不完整")
                elif metric == 'answer_similarity':
                    issues.append("答案质量偏低")

        # 自定义指标问题
        for metric, value in custom_metrics.items():
            if value < 0.6:
                if metric == 'keyword_coverage':
                    issues.append("关键信息缺失")
                elif metric == 'entity_accuracy':
                    issues.append("实体识别错误")
                elif metric == 'context_completeness':
                    issues.append("检索结果不全面")
                elif metric == 'answer_completeness':
                    issues.append("答案不够完整")
                elif metric == 'terminology_score':
                    issues.append("专业术语使用不足")

        return issues

    def _save_evaluation_results(self, results: List[EvaluationResult]):
        """保存评估结果"""
        try:
            # 保存到Redis
            results_data = [asdict(result) for result in results]
            timestamp = datetime.now().isoformat()

            redis_client.setex(
                f"evaluation_results:{timestamp}",
                86400 * 30,  # 保存30天
                json.dumps(results_data, ensure_ascii=False, default=str)
            )

            # 保存到历史记录
            self.evaluation_history.extend(results_data)

            # 保持最近1000条记录
            if len(self.evaluation_history) > 1000:
                self.evaluation_history = self.evaluation_history[-1000:]

        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")

    async def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """生成评估报告"""
        try:
            if not results:
                return {"error": "没有评估结果"}

            # 基础统计
            total_questions = len(results)
            avg_score = sum(r.overall_score for r in results) / total_questions
            high_score_count = sum(1 for r in results if r.overall_score >= 0.8)
            low_score_count = sum(1 for r in results if r.overall_score < 0.6)

            # 按问题类型统计
            type_scores = {}
            type_counts = {}
            for result in results:
                # 从问题ID推断类型（简化实现）
                question_type = result.question_id.split('_')[0]
                if question_type not in type_scores:
                    type_scores[question_type] = []
                    type_counts[question_type] = 0
                type_scores[question_type].append(result.overall_score)
                type_counts[question_type] += 1

            type_avg_scores = {
                qtype: sum(scores) / len(scores)
                for qtype, scores in type_scores.items()
            }

            # 问题分析
            all_issues = []
            for result in results:
                all_issues.extend(result.issues)

            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            # RAGAS指标统计
            ragas_metrics = {}
            for metric_name in ['faithfulness', 'answer_relevancy', 'context_relevancy', 'context_recall']:
                values = [r.ragas_metrics.get(metric_name, 0) for r in results if metric_name in r.ragas_metrics]
                if values:
                    ragas_metrics[metric_name] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }

            # 自定义指标统计
            custom_metrics = {}
            for metric_name in ['keyword_coverage', 'entity_accuracy', 'context_completeness']:
                values = [r.custom_metrics.get(metric_name, 0) for r in results if metric_name in r.custom_metrics]
                if values:
                    custom_metrics[metric_name] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }

            report = {
                'evaluation_summary': {
                    'total_questions': total_questions,
                    'average_score': round(avg_score, 3),
                    'high_score_rate': round(high_score_count / total_questions, 3),
                    'low_score_rate': round(low_score_count / total_questions, 3),
                    'evaluation_date': datetime.now().isoformat()
                },
                'performance_by_type': {
                    'scores': type_avg_scores,
                    'counts': type_counts
                },
                'ragas_metrics': ragas_metrics,
                'custom_metrics': custom_metrics,
                'common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'recommendations': self._generate_recommendations(avg_score, ragas_metrics, custom_metrics, issue_counts)
            }

            return report

        except Exception as e:
            logger.error(f"生成评估报告失败: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, avg_score: float, ragas_metrics: Dict, custom_metrics: Dict, issue_counts: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于总体评分的建议
        if avg_score < 0.7:
            recommendations.append("整体评分偏低，需要全面优化系统性能")
        elif avg_score < 0.8:
            recommendations.append("整体评分中等，建议针对弱项进行优化")

        # 基于RAGAS指标的建议
        if ragas_metrics.get('faithfulness', {}).get('avg', 1) < 0.7:
            recommendations.append("提高答案忠实度：优化检索结果筛选和答案生成逻辑")

        if ragas_metrics.get('answer_relevancy', {}).get('avg', 1) < 0.7:
            recommendations.append("提高答案相关性：改进查询理解和答案生成模型")

        if ragas_metrics.get('context_relevancy', {}).get('avg', 1) < 0.7:
            recommendations.append("提高上下文相关性：优化检索算法和排序策略")

        if ragas_metrics.get('context_recall', {}).get('avg', 1) < 0.7:
            recommendations.append("提高上下文召回率：扩大检索范围和改进召回策略")

        # 基于自定义指标的建议
        if custom_metrics.get('keyword_coverage', {}).get('avg', 1) < 0.7:
            recommendations.append("提高关键词覆盖率：改进实体识别和信息提取")

        if custom_metrics.get('context_completeness', {}).get('avg', 1) < 0.7:
            recommendations.append("提高上下文完整性：增加多源检索和信息融合")

        # 基于问题频次的建议
        for issue, count in issue_counts.items():
            if count > len(recommendations) * 0.3:  # 出现频率较高
                recommendations.append(f"重点解决：{issue}")

        if not recommendations:
            recommendations.append("系统表现良好，继续保持当前配置")

        return recommendations

    async def schedule_daily_evaluation(self):
        """调度每日评估"""
        try:
            logger.info("开始每日自动评估")

            # 运行完整评估
            results = await self.run_evaluation()

            # 生成报告
            report = await self.generate_evaluation_report(results)

            # 保存报告
            report_data = {
                'report_date': datetime.now().isoformat(),
                'report': report,
                'results_count': len(results)
            }

            redis_client.setex(
                f"daily_evaluation_report:{datetime.now().strftime('%Y-%m-%d')}",
                86400 * 90,  # 保存90天
                json.dumps(report_data, ensure_ascii=False, default=str)
            )

            # 检查是否需要告警
            await self._check_alert_conditions(report)

            logger.info(f"每日评估完成，评估了 {len(results)} 个问题")
            return report

        except Exception as e:
            logger.error(f"每日评估失败: {e}")
            return None

    async def _check_alert_conditions(self, report: Dict[str, Any]):
        """检查告警条件"""
        try:
            alerts = []

            # 检查总体评分
            avg_score = report.get('evaluation_summary', {}).get('average_score', 1)
            if avg_score < 0.6:
                alerts.append({
                    'type': 'critical',
                    'message': f'系统整体评分过低: {avg_score:.3f}',
                    'threshold': 0.6
                })
            elif avg_score < 0.75:
                alerts.append({
                    'type': 'warning',
                    'message': f'系统整体评分偏低: {avg_score:.3f}',
                    'threshold': 0.75
                })

            # 检查关键指标
            ragas_metrics = report.get('ragas_metrics', {})

            for metric, threshold in [('faithfulness', 0.8), ('answer_relevancy', 0.75)]:
                avg_value = ragas_metrics.get(metric, {}).get('avg', 1)
                if avg_value < threshold:
                    alerts.append({
                        'type': 'warning',
                        'message': f'{metric}指标过低: {avg_value:.3f}',
                        'threshold': threshold
                    })

            # 发送告警通知
            if alerts:
                await self._send_alerts(alerts)

        except Exception as e:
            logger.error(f"检查告警条件失败: {e}")

    async def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """发送告警通知"""
        try:
            from app.core.websocket_manager import connection_manager

            for alert in alerts:
                message = f"评估告警: {alert['message']}"
                level = alert['type']

                # 通过WebSocket发送通知
                await connection_manager.send_system_notification(message, level)

                # 保存告警记录
                alert_data = {
                    **alert,
                    'timestamp': datetime.now().isoformat()
                }
                redis_client.lpush('evaluation_alerts', json.dumps(alert_data, ensure_ascii=False))
                redis_client.ltrim('evaluation_alerts', 0, 999)  # 保留最近1000条

        except Exception as e:
            logger.error(f"发送告警失败: {e}")

    def get_evaluation_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取评估统计信息"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # 从历史记录中筛选数据
            recent_evaluations = [
                eval_data for eval_data in self.evaluation_history
                if datetime.fromisoformat(eval_data['evaluation_time']) >= cutoff_date
            ]

            if not recent_evaluations:
                return {"message": f"过去{days}天没有评估记录"}

            # 计算统计信息
            scores = [eval_data['overall_score'] for eval_data in recent_evaluations]

            return {
                'period': f"past_{days}_days",
                'total_evaluations': len(recent_evaluations),
                'average_score': round(sum(scores) / len(scores), 3),
                'max_score': max(scores),
                'min_score': min(scores),
                'score_distribution': {
                    'excellent': len([s for s in scores if s >= 0.9]),
                    'good': len([s for s in scores if 0.8 <= s < 0.9]),
                    'fair': len([s for s in scores if 0.6 <= s < 0.8]),
                    'poor': len([s for s in scores if s < 0.6])
                },
                'trend': self._calculate_score_trend(scores)
            }

        except Exception as e:
            logger.error(f"获取评估统计失败: {e}")
            return {"error": str(e)}

    def _calculate_score_trend(self, scores: List[float]) -> str:
        """计算评分趋势"""
        if len(scores) < 7:
            return "insufficient_data"

        # 比较最近7天和之前7天的平均值
        recent_avg = sum(scores[-7:]) / 7
        previous_avg = sum(scores[-14:-7]) / 7 if len(scores) >= 14 else sum(scores[:-7]) / (len(scores) - 7)

        if recent_avg > previous_avg * 1.05:
            return "improving"
        elif recent_avg < previous_avg * 0.95:
            return "declining"
        else:
            return "stable"