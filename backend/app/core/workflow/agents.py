"""
智能体定义
预定义的工作流智能体
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.schema import SystemMessage, HumanMessage
    from langchain.tools import BaseTool
    from langchain.chat_models.base import BaseChatModel
except ImportError:
    # 如果LangChain未安装，提供模拟类
    AgentExecutor = object
    create_openai_tools_agent = object
    SystemMessage = object
    HumanMessage = object
    BaseTool = object
    BaseChatModel = object

from app.core.config import settings
from app.services.search_service import SearchService
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class BaseAgent:
    """基础智能体类"""

    def __init__(self, name: str, description: str, tools: List[BaseTool] = None):
        self.name = name
        self.description = description
        self.tools = tools or []
        self.llm_service = LLMService()

    async def arun(self, input_data: Dict[str, Any]) -> Any:
        """异步运行智能体"""
        raise NotImplementedError

    def run(self, input_data: Dict[str, Any]) -> Any:
        """同步运行智能体"""
        raise NotImplementedError


class SearchAgent(BaseAgent):
    """搜索智能体"""

    def __init__(self):
        super().__init__(
            name="search_agent",
            description="专门用于文档搜索的智能体",
            tools=[SearchTool(), VectorSearchTool(), GraphSearchTool()]
        )

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行搜索任务"""
        query = input_data.get('query', '')
        search_config = input_data.get('search_config', {})
        search_methods = input_data.get('search_methods', ['vector'])

        if not query:
            return {'error': '查询不能为空'}

        search_service = SearchService()
        results = {}

        # 执行多种搜索方法
        for method in search_methods:
            try:
                if method == 'vector':
                    method_results = await search_service.vector_search(query, **search_config)
                    results['vector'] = method_results
                elif method == 'graph':
                    method_results = await search_service.graph_search(query, **search_config)
                    results['graph'] = method_results
                elif method == 'keyword':
                    method_results = await search_service.keyword_search(query, **search_config)
                    results['keyword'] = method_results

            except Exception as e:
                logger.error(f"搜索方法 {method} 执行失败: {e}")
                results[method] = {'error': str(e)}

        return results

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行搜索任务"""
        return asyncio.run(self.arun(input_data))


class LLMAgent(BaseAgent):
    """LLM智能体"""

    def __init__(self, model_name: str = None):
        super().__init__(
            name="llm_agent",
            description="基于大语言模型的智能体"
        )
        self.model_name = model_name or settings.DEFAULT_LLM_MODEL

    async def arun(self, input_data: Dict[str, Any]) -> str:
        """执行LLM生成任务"""
        query = input_data.get('query', '')
        contexts = input_data.get('contexts', [])
        instruction = input_data.get('instruction', '')
        temperature = input_data.get('temperature', 0.1)
        max_tokens = input_data.get('max_tokens', 2000)

        # 构建提示
        prompt = self._build_prompt(query, contexts, instruction)

        # 调用LLM
        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"生成答案时出现错误: {str(e)}"

    def _build_prompt(self, query: str, contexts: List[str], instruction: str) -> str:
        """构建提示词"""
        prompt = f"请基于以下信息回答用户问题：\n\n"

        if instruction:
            prompt += f"特殊要求：{instruction}\n\n"

        prompt += "参考信息：\n"
        for i, context in enumerate(contexts, 1):
            prompt += f"{i}. {context}\n"

        prompt += f"\n用户问题：{query}\n\n"
        prompt += "请提供准确、详细、有帮助的回答。如果信息不足，请说明需要补充哪些信息。"

        return prompt

    def run(self, input_data: Dict[str, Any]) -> str:
        """同步执行LLM任务"""
        return asyncio.run(self.arun(input_data))


class DocumentAgent(BaseAgent):
    """文档处理智能体"""

    def __init__(self):
        super().__init__(
            name="document_agent",
            description="专门用于文档处理的智能体",
            tools=[DocumentUploadTool(), DocumentExtractTool(), DocumentAnalyzeTool()]
        )

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行文档处理任务"""
        task_type = input_data.get('task_type', '')
        doc_id = input_data.get('doc_id', '')
        file_path = input_data.get('file_path', '')
        options = input_data.get('options', {})

        document_service = DocumentService()

        try:
            if task_type == 'upload':
                result = await document_service.upload_document(file_path, **options)
            elif task_type == 'extract':
                result = await document_service.extract_content(doc_id, **options)
            elif task_type == 'analyze':
                result = await document_service.analyze_document(doc_id, **options)
            else:
                result = {'error': f'不支持的任务类型: {task_type}'}

            return result

        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return {'error': str(e)}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行文档处理任务"""
        return asyncio.run(self.arun(input_data))


class EvaluationAgent(BaseAgent):
    """评估智能体"""

    def __init__(self):
        super().__init__(
            name="evaluation_agent",
            description="专门用于质量评估的智能体",
            tools=[RAGASEvaluationTool(), QualityCheckTool()]
        )

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行评估任务"""
        evaluation_type = input_data.get('evaluation_type', 'ragas')
        question = input_data.get('question', '')
        answer = input_data.get('answer', '')
        contexts = input_data.get('contexts', [])
        ground_truth = input_data.get('ground_truth', '')

        if evaluation_type == 'ragas':
            from app.services.evaluation.ragas_evaluator import RAGASEvaluator
            evaluator = RAGASEvaluator()

            try:
                result = await evaluator.evaluate(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth
                )
                return result.dict()
            except Exception as e:
                logger.error(f"RAGAS评估失败: {e}")
                return {'error': str(e)}

        elif evaluation_type == 'quality':
            # 简单的质量检查
            quality_score = self._simple_quality_check(answer, contexts)
            return {
                'quality_score': quality_score,
                'evaluation_type': 'simple_quality'
            }

        else:
            return {'error': f'不支持的评估类型: {evaluation_type}'}

    def _simple_quality_check(self, answer: str, contexts: List[str]) -> float:
        """简单的质量检查"""
        if not answer:
            return 0.0

        score = 0.0

        # 长度检查
        if 50 <= len(answer) <= 1000:
            score += 0.3
        elif len(answer) > 1000:
            score += 0.2

        # 相关性检查（简单版本）
        if contexts and any(context in answer for context in contexts[:3]):
            score += 0.4

        # 结构检查
        if '：' in answer or '.' in answer or '、' in answer:
            score += 0.3

        return min(score, 1.0)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行评估任务"""
        return asyncio.run(self.arun(input_data))


class AnalysisAgent(BaseAgent):
    """分析智能体"""

    def __init__(self):
        super().__init__(
            name="analysis_agent",
            description="专门用于数据分析的智能体"
        )

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行分析任务"""
        analysis_type = input_data.get('analysis_type', '')
        data = input_data.get('data', {})
        parameters = input_data.get('parameters', {})

        try:
            if analysis_type == 'query_analysis':
                result = self._analyze_query(data.get('query', ''))
            elif analysis_type == 'result_analysis':
                result = self._analyze_results(data.get('results', []))
            elif analysis_type == 'trend_analysis':
                result = self._analyze_trends(data, parameters)
            else:
                result = {'error': f'不支持的分析类型: {analysis_type}'}

            return result

        except Exception as e:
            logger.error(f"分析任务失败: {e}")
            return {'error': str(e)}

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询"""
        from .nodes import extract_keywords, identify_financial_terms, extract_time_expressions

        return {
            'keywords': extract_keywords(query),
            'financial_terms': identify_financial_terms(query),
            'time_expressions': extract_time_expressions(query),
            'query_length': len(query),
            'complexity': 'high' if len(query) > 50 else 'medium' if len(query) > 20 else 'low'
        }

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析结果"""
        if not results:
            return {'total_results': 0, 'analysis': '无结果'}

        # 分析结果质量
        scores = [r.get('score', 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0

        # 分析来源多样性
        sources = set(r.get('source_type', 'unknown') for r in results)
        source_diversity = len(sources)

        return {
            'total_results': len(results),
            'avg_score': avg_score,
            'max_score': max(scores) if scores else 0,
            'source_diversity': source_diversity,
            'sources': list(sources),
            'quality_rating': 'high' if avg_score > 0.8 else 'medium' if avg_score > 0.5 else 'low'
        }

    def _analyze_trends(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """分析趋势"""
        # 简单的趋势分析
        return {
            'trend_direction': 'stable',
            'confidence': 0.7,
            'key_factors': ['数据不足'],
            'recommendations': ['需要更多历史数据进行分析']
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行分析任务"""
        return asyncio.run(self.arun(input_data))


# 工具类定义
class SearchTool(BaseTool):
    """搜索工具"""
    name = "search"
    description = "执行文档搜索"

    def _run(self, query: str) -> str:
        return f"搜索: {query}"

    async def _arun(self, query: str) -> str:
        return await self._run(query)


class VectorSearchTool(BaseTool):
    """向量搜索工具"""
    name = "vector_search"
    description = "执行向量相似度搜索"

    def _run(self, query: str) -> str:
        return f"向量搜索: {query}"

    async def _arun(self, query: str) -> str:
        return await self._run(query)


class GraphSearchTool(BaseTool):
    """图搜索工具"""
    name = "graph_search"
    description = "执行图数据库搜索"

    def _run(self, query: str) -> str:
        return f"图搜索: {query}"

    async def _arun(self, query: str) -> str:
        return await self._run(query)


class DocumentUploadTool(BaseTool):
    """文档上传工具"""
    name = "upload_document"
    description = "上传并处理文档"

    def _run(self, file_path: str) -> str:
        return f"上传文档: {file_path}"

    async def _arun(self, file_path: str) -> str:
        return await self._run(file_path)


class DocumentExtractTool(BaseTool):
    """文档提取工具"""
    name = "extract_document"
    description = "提取文档内容"

    def _run(self, doc_id: str) -> str:
        return f"提取文档: {doc_id}"

    async def _arun(self, doc_id: str) -> str:
        return await self._run(doc_id)


class DocumentAnalyzeTool(BaseTool):
    """文档分析工具"""
    name = "analyze_document"
    description = "分析文档内容"

    def _run(self, doc_id: str) -> str:
        return f"分析文档: {doc_id}"

    async def _arun(self, doc_id: str) -> str:
        return await self._run(doc_id)


class RAGASEvaluationTool(BaseTool):
    """RAGAS评估工具"""
    name = "ragas_evaluation"
    description = "使用RAGAS进行答案质量评估"

    def _run(self, evaluation_data: str) -> str:
        return f"RAGAS评估: {evaluation_data}"

    async def _arun(self, evaluation_data: str) -> str:
        return await self._run(evaluation_data)


class QualityCheckTool(BaseTool):
    """质量检查工具"""
    name = "quality_check"
    description = "检查内容质量"

    def _run(self, content: str) -> str:
        return f"质量检查: {content[:50]}..."

    async def _arun(self, content: str) -> str:
        return await self._run(content)


# 预定义智能体实例
PREDEFINED_AGENTS = {
    'search_agent': SearchAgent(),
    'llm_agent': LLMAgent(),
    'document_agent': DocumentAgent(),
    'evaluation_agent': EvaluationAgent(),
    'analysis_agent': AnalysisAgent()
}


def get_agent(agent_name: str) -> Optional[BaseAgent]:
    """获取预定义智能体"""
    return PREDEFINED_AGENTS.get(agent_name)


def register_agent(name: str, agent: BaseAgent) -> None:
    """注册自定义智能体"""
    PREDEFINED_AGENTS[name] = agent
    logger.info(f"注册智能体: {name}")