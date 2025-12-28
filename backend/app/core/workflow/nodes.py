"""
工作流节点定义
预定义的常用工作流节点
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

from .workflow_engine import WorkflowNode, NodeType
from app.services.search_service import SearchService
from app.services.document_service import DocumentService
from app.services.evaluation.ragas_evaluator import RAGASEvaluator

logger = logging.getLogger(__name__)


class PredefinedNodes:
    """预定义节点类"""

    @staticmethod
    def create_query_analysis_node() -> WorkflowNode:
        """创建查询分析节点"""
        async def analyze_query(state: Dict[str, Any]) -> Dict[str, Any]:
            """分析用户查询"""
            query = state.get('query', '')
            if not query:
                return {'error': '查询不能为空'}

            # 查询分析逻辑
            analysis = {
                'original_query': query,
                'query_type': 'question',  # question, command, search
                'intent': 'information_seeking',
                'complexity': 'medium',
                'entities': [],
                'keywords': [],
                'time_expressions': [],
                'financial_terms': [],
                'confidence': 0.8
            }

            # 提取关键词
            keywords = extract_keywords(query)
            analysis['keywords'] = keywords

            # 识别金融术语
            financial_terms = identify_financial_terms(query)
            analysis['financial_terms'] = financial_terms

            # 检测时间表达式
            time_expressions = extract_time_expressions(query)
            analysis['time_expressions'] = time_expressions

            return {
                'query_analysis': analysis,
                'processed_query': query,
                'search_strategy': recommend_search_strategy(analysis)
            }

        return WorkflowNode(
            id="query_analysis",
            name="查询分析",
            node_type=NodeType.TRANSFORM,
            description="分析用户查询，提取关键信息",
            function=analyze_query,
            config={
                'enable_nlp': True,
                'extract_entities': True
            }
        )

    @staticmethod
    def create_search_strategy_node() -> WorkflowNode:
        """创建搜索策略选择节点"""
        async def select_strategy(state: Dict[str, Any]) -> Dict[str, Any]:
            """选择搜索策略"""
            query_analysis = state.get('query_analysis', {})
            search_strategy = query_analysis.get('search_strategy', 'intelligent')

            strategies = {
                'intelligent': {
                    'name': '智能搜索',
                    'methods': ['vector', 'graph', 'keyword'],
                    'weights': {'vector': 0.4, 'graph': 0.3, 'keyword': 0.3},
                    'max_results': 10
                },
                'deep': {
                    'name': '深度搜索',
                    'methods': ['vector', 'graph', 'keyword'],
                    'max_iterations': 3,
                    'convergence_threshold': 0.8
                },
                'fast': {
                    'name': '快速搜索',
                    'methods': ['keyword'],
                    'max_results': 5
                }
            }

            selected_strategy = strategies.get(search_strategy, strategies['intelligent'])

            return {
                'search_strategy_config': selected_strategy,
                'search_methods': selected_strategy['methods'],
                'max_results': selected_strategy.get('max_results', 10)
            }

        return WorkflowNode(
            id="search_strategy",
            name="搜索策略选择",
            node_type=NodeType.TRANSFORM,
            description="根据查询分析结果选择最佳搜索策略",
            function=select_strategy
        )

    @staticmethod
    def create_retrieval_node() -> WorkflowNode:
        """创建检索节点"""
        async def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
            """执行文档检索"""
            query = state.get('processed_query', '')
            search_config = state.get('search_strategy_config', {})
            search_methods = state.get('search_methods', ['vector'])

            search_service = SearchService()
            results = {}

            # 执行多策略检索
            if 'vector' in search_methods:
                vector_results = await search_service.vector_search(query, **search_config)
                results['vector'] = vector_results

            if 'graph' in search_methods:
                graph_results = await search_service.graph_search(query, **search_config)
                results['graph'] = graph_results

            if 'keyword' in search_methods:
                keyword_results = await search_service.keyword_search(query, **search_config)
                results['keyword'] = keyword_results

            # 结果融合
            fused_results = search_service.fuse_results(results, search_config.get('weights', {}))

            return {
                'search_results': results,
                'fused_results': fused_results,
                'total_results': len(fused_results)
            }

        return WorkflowNode(
            id="retrieval",
            name="文档检索",
            node_type=NodeType.AGENT,
            description="执行多策略文档检索",
            agent="search_agent",
            config={
                'enable_fusion': True,
                'max_results_per_method': 20
            }
        )

    @staticmethod
    def create_answer_generation_node() -> WorkflowNode:
        """创建答案生成节点"""
        async def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
            """生成答案"""
            query = state.get('processed_query', '')
            fused_results = state.get('fused_results', [])
            query_analysis = state.get('query_analysis', {})

            if not fused_results:
                return {
                    'answer': '抱歉，没有找到相关的信息来回答您的问题。',
                    'answer_type': 'no_results'
                }

            # 准备上下文
            contexts = [result.get('content', '') for result in fused_results[:5]]

            # 生成答案
            answer_data = {
                'question': query,
                'contexts': contexts,
                'query_type': query_analysis.get('query_type', 'question'),
                'intent': query_analysis.get('intent', 'information_seeking')
            }

            # 调用LLM生成答案
            answer = await generate_llm_answer(answer_data)

            return {
                'answer': answer,
                'answer_type': 'generated',
                'source_count': len(contexts),
                'generation_time': datetime.now().isoformat()
            }

        return WorkflowNode(
            id="answer_generation",
            name="答案生成",
            node_type=NodeType.AGENT,
            description="基于检索结果生成答案",
            agent="llm_agent",
            config={
                'model': 'deepseek-chat',
                'temperature': 0.1,
                'max_tokens': 2000
            }
        )

    @staticmethod
    def create_answer_evaluation_node() -> WorkflowNode:
        """创建答案评估节点"""
        async def evaluate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
            """评估答案质量"""
            query = state.get('processed_query', '')
            answer = state.get('answer', '')
            fused_results = state.get('fused_results', [])
            contexts = [result.get('content', '') for result in fused_results[:5]]

            if not answer or not contexts:
                return {'evaluation_score': 0.0, 'evaluation_details': '无答案或上下文'}

            # 使用RAGAS评估
            evaluator = RAGASEvaluator()
            evaluation = await evaluator.evaluate(
                question=query,
                answer=answer,
                contexts=contexts
            )

            return {
                'evaluation_score': evaluation.overall_score,
                'evaluation_details': {
                    'faithfulness': evaluation.faithfulness_score,
                    'answer_relevance': evaluation.answer_relevance_score,
                    'context_relevance': evaluation.context_relevance_score,
                    'context_recall': evaluation.context_recall_score,
                    'answer_correctness': evaluation.answer_correctness_score,
                    'aspect_critique': evaluation.aspect_critique_score
                },
                'evaluation_time': datetime.now().isoformat()
            }

        return WorkflowNode(
            id="answer_evaluation",
            name="答案评估",
            node_type=NodeType.TOOL,
            description="评估生成答案的质量",
            tool="ragas_evaluator",
            config={
                'aspects': ['faithfulness', 'relevance', 'correctness']
            }
        )

    @staticmethod
    def create_result_refinement_node() -> WorkflowNode:
        """创建结果优化节点"""
        async def refine_result(state: Dict[str, Any]) -> Dict[str, Any]:
            """优化最终结果"""
            query = state.get('processed_query', '')
            answer = state.get('answer', '')
            fused_results = state.get('fused_results', [])
            evaluation_score = state.get('evaluation_score', 0.0)

            # 如果评估分数过低，尝试优化答案
            if evaluation_score < 0.6:
                refined_answer = await refine_low_quality_answer(query, answer, fused_results)
            else:
                refined_answer = answer

            # 格式化最终输出
            final_result = {
                'query': query,
                'answer': refined_answer,
                'sources': [
                    {
                        'id': result.get('id'),
                        'title': result.get('title', ''),
                        'content_snippet': result.get('content', '')[:200] + '...',
                        'score': result.get('score', 0.0),
                        'source_type': result.get('source_type', 'document')
                    }
                    for result in fused_results[:5]
                ],
                'metadata': {
                    'search_strategy': state.get('search_strategy_config', {}).get('name'),
                    'total_results': len(fused_results),
                    'evaluation_score': evaluation_score,
                    'generation_time': state.get('generation_time'),
                    'refined': evaluation_score < 0.6
                }
            }

            return {
                'final_result': final_result,
                'refined': evaluation_score < 0.6
            }

        return WorkflowNode(
            id="result_refinement",
            name="结果优化",
            node_type=NodeType.TRANSFORM,
            description="优化和格式化最终结果",
            function=refine_result
        )

    @staticmethod
    def create_quality_check_node() -> WorkflowNode:
        """创建质量检查节点"""
        async def quality_check(state: Dict[str, Any]) -> bool:
            """检查结果质量"""
            answer = state.get('answer', '')
            evaluation_score = state.get('evaluation_score', 0.0)

            # 质量检查条件
            has_answer = bool(answer and len(answer) > 20)
            good_quality = evaluation_score >= 0.6

            return has_answer and good_quality

        return WorkflowNode(
            id="quality_check",
            name="质量检查",
            node_type=NodeType.CONDITION,
            description="检查生成结果的质量",
            function=quality_check
        )


# 辅助函数
def extract_keywords(query: str) -> List[str]:
    """提取关键词"""
    # 简单的关键词提取，实际应该使用NLP库
    import re
    words = re.findall(r'\b\w+\b', query.lower())
    # 过滤停用词
    stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '呢', '吗', '吧'}
    keywords = [word for word in words if len(word) > 1 and word not in stop_words]
    return keywords[:10]  # 返回前10个关键词


def identify_financial_terms(query: str) -> List[str]:
    """识别金融术语"""
    financial_terms = [
        '银行', '贷款', '存款', '利率', '利息', '投资', '理财', '基金', '股票',
        '债券', '保险', '信托', '期货', '外汇', '黄金', '房地产', 'GDP', 'CPI',
        '净利润', '营业收入', '资产负债', '现金流', '市值', '股息', '分红'
    ]

    found_terms = []
    query_lower = query.lower()
    for term in financial_terms:
        if term in query_lower:
            found_terms.append(term)

    return found_terms


def extract_time_expressions(query: str) -> List[str]:
    """提取时间表达式"""
    import re
    time_patterns = [
        r'\d{4}年',
        r'\d{1,2}月',
        r'\d{1,2}日',
        r'今年|去年|前年',
        r'第一季度|第二季度|第三季度|第四季度',
        r'上半年|下半年',
        r'最近|近期|本年|本季度'
    ]

    time_expressions = []
    for pattern in time_patterns:
        matches = re.findall(pattern, query)
        time_expressions.extend(matches)

    return time_expressions


def recommend_search_strategy(analysis: Dict[str, Any]) -> str:
    """推荐搜索策略"""
    complexity = analysis.get('complexity', 'medium')
    financial_terms = analysis.get('financial_terms', [])
    time_expressions = analysis.get('time_expressions', [])

    # 根据分析结果推荐策略
    if complexity == 'high' or len(financial_terms) > 2:
        return 'deep'
    elif len(time_expressions) > 0 or len(financial_terms) > 0:
        return 'intelligent'
    else:
        return 'fast'


async def generate_llm_answer(answer_data: Dict[str, Any]) -> str:
    """生成LLM答案"""
    # 这里应该调用实际的LLM API
    # 暂时返回模拟答案
    question = answer_data.get('question', '')
    contexts = answer_data.get('contexts', [])

    if not contexts:
        return f"抱歉，我没有找到关于'{question}'的相关信息。"

    # 模拟答案生成
    answer = f"根据相关资料，关于'{question}'的回答如下：\n\n"
    answer += "基于检索到的信息，可以总结出以下几点：\n"
    answer += "1. 相关数据显示了重要趋势\n"
    answer += "2. 分析表明存在关键影响因素\n"
    answer += "3. 建议关注后续发展动态\n\n"
    answer += "详细信息请参考相关报告和文档。"

    return answer


async def refine_low_quality_answer(query: str, answer: str, results: List[Dict[str, Any]]) -> str:
    """优化低质量答案"""
    # 简单的答案优化逻辑
    if len(answer) < 100:
        # 答案太短，尝试扩展
        refined = f"{answer}\n\n根据更多相关资料显示，这个问题涉及多个方面。"
        refined += "建议您参考相关报告获取更详细的信息。"
        return refined
    return answer