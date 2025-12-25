"""
工作流工具定义
预定义的工作流工具
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import os
from pathlib import Path

try:
    from langchain.tools import BaseTool
except ImportError:
    # 如果LangChain未安装，提供模拟类
    BaseTool = object

from app.services.search_service import SearchService
from app.services.document_service import DocumentService
from app.services.evaluation.ragas_evaluator import RAGASEvaluator
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """搜索工具"""
    name = "search"
    description = "执行文档搜索，支持关键词搜索"

    def __init__(self):
        self.search_service = SearchService()

    def _run(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """同步执行搜索"""
        return asyncio.run(self._arun(query, max_results))

    async def _arun(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """异步执行搜索"""
        try:
            results = await self.search_service.keyword_search(
                query=query,
                limit=max_results
            )
            return results
        except Exception as e:
            logger.error(f"搜索工具执行失败: {e}")
            return []


class VectorSearchTool(BaseTool):
    """向量搜索工具"""
    name = "vector_search"
    description = "执行向量相似度搜索"

    def __init__(self):
        self.search_service = SearchService()

    def _run(self, query: str, top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """同步执行向量搜索"""
        return asyncio.run(self._arun(query, top_k, threshold))

    async def _arun(self, query: str, top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """异步执行向量搜索"""
        try:
            results = await self.search_service.vector_search(
                query=query,
                top_k=top_k,
                threshold=threshold
            )
            return results
        except Exception as e:
            logger.error(f"向量搜索工具执行失败: {e}")
            return []


class GraphSearchTool(BaseTool):
    """图搜索工具"""
    name = "graph_search"
    description = "执行图数据库搜索，查找实体关系"

    def __init__(self):
        self.search_service = SearchService()

    def _run(self, query: str, max_depth: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
        """同步执行图搜索"""
        return asyncio.run(self._arun(query, max_depth, limit))

    async def _arun(self, query: str, max_depth: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
        """异步执行图搜索"""
        try:
            results = await self.search_service.graph_search(
                query=query,
                max_depth=max_depth,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"图搜索工具执行失败: {e}")
            return []


class DocumentUploadTool(BaseTool):
    """文档上传工具"""
    name = "document_upload"
    description = "上传并处理文档文件"

    def __init__(self):
        self.document_service = DocumentService()

    def _run(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """同步执行文档上传"""
        return asyncio.run(self._arun(file_path, metadata))

    async def _arun(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """异步执行文档上传"""
        try:
            if not os.path.exists(file_path):
                return {'error': f'文件不存在: {file_path}'}

            result = await self.document_service.upload_document(
                file_path=file_path,
                metadata=metadata or {}
            )
            return result
        except Exception as e:
            logger.error(f"文档上传工具执行失败: {e}")
            return {'error': str(e)}


class DocumentExtractTool(BaseTool):
    """文档内容提取工具"""
    name = "document_extract"
    description = "提取文档内容"

    def __init__(self):
        self.document_service = DocumentService()

    def _run(self, doc_id: str, extract_type: str = "content") -> Dict[str, Any]:
        """同步执行文档提取"""
        return asyncio.run(self._arun(doc_id, extract_type))

    async def _arun(self, doc_id: str, extract_type: str = "content") -> Dict[str, Any]:
        """异步执行文档提取"""
        try:
            result = await self.document_service.extract_content(
                doc_id=doc_id,
                extract_type=extract_type
            )
            return result
        except Exception as e:
            logger.error(f"文档提取工具执行失败: {e}")
            return {'error': str(e)}


class DocumentAnalyzeTool(BaseTool):
    """文档分析工具"""
    name = "document_analyze"
    description = "分析文档内容，提取实体和关系"

    def __init__(self):
        self.document_service = DocumentService()

    def _run(self, doc_id: str, analysis_type: str = "entity") -> Dict[str, Any]:
        """同步执行文档分析"""
        return asyncio.run(self._arun(doc_id, analysis_type))

    async def _arun(self, doc_id: str, analysis_type: str = "entity") -> Dict[str, Any]:
        """异步执行文档分析"""
        try:
            result = await self.document_service.analyze_document(
                doc_id=doc_id,
                analysis_type=analysis_type
            )
            return result
        except Exception as e:
            logger.error(f"文档分析工具执行失败: {e}")
            return {'error': str(e)}


class RAGASEvaluationTool(BaseTool):
    """RAGAS评估工具"""
    name = "ragas_evaluation"
    description = "使用RAGAS框架评估答案质量"

    def __init__(self):
        self.evaluator = RAGASEvaluator()

    def _run(self, question: str, answer: str, contexts: List[str], ground_truth: str = "") -> Dict[str, Any]:
        """同步执行RAGAS评估"""
        return asyncio.run(self._arun(question, answer, contexts, ground_truth))

    async def _arun(self, question: str, answer: str, contexts: List[str], ground_truth: str = "") -> Dict[str, Any]:
        """异步执行RAGAS评估"""
        try:
            result = await self.evaluator.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            )
            return result.dict()
        except Exception as e:
            logger.error(f"RAGAS评估工具执行失败: {e}")
            return {'error': str(e)}


class CacheTool(BaseTool):
    """缓存工具"""
    name = "cache"
    description = "操作缓存数据"

    def _run(self, operation: str, key: str, value: str = None, namespace: str = "default") -> Any:
        """同步执行缓存操作"""
        return asyncio.run(self._arun(operation, key, value, namespace))

    async def _arun(self, operation: str, key: str, value: str = None, namespace: str = "default") -> Any:
        """异步执行缓存操作"""
        try:
            if operation == "get":
                return await cache_manager.get(key, namespace)
            elif operation == "set":
                return await cache_manager.set(key, value, namespace=namespace)
            elif operation == "delete":
                return await cache_manager.delete(key, namespace)
            elif operation == "exists":
                return await cache_manager.exists(key, namespace)
            else:
                return {'error': f'不支持的操作: {operation}'}
        except Exception as e:
            logger.error(f"缓存工具执行失败: {e}")
            return {'error': str(e)}


class DataTransformTool(BaseTool):
    """数据转换工具"""
    name = "data_transform"
    description = "转换和格式化数据"

    def _run(self, data: str, transform_type: str, params: str = "{}") -> str:
        """同步执行数据转换"""
        return asyncio.run(self._arun(data, transform_type, params))

    async def _arun(self, data: str, transform_type: str, params: str = "{}") -> str:
        """异步执行数据转换"""
        try:
            # 解析参数
            try:
                params_dict = json.loads(params)
            except:
                params_dict = {}

            # 解析数据
            try:
                if isinstance(data, str):
                    data_obj = json.loads(data)
                else:
                    data_obj = data
            except:
                data_obj = {"raw_data": str(data)}

            # 执行转换
            if transform_type == "format_json":
                return json.dumps(data_obj, indent=2, ensure_ascii=False)
            elif transform_type == "extract_keys":
                keys = list(data_obj.keys()) if isinstance(data_obj, dict) else []
                return json.dumps(keys, ensure_ascii=False)
            elif transform_type == "flatten":
                return self._flatten_data(data_obj)
            elif transform_type == "filter":
                filter_key = params_dict.get("key")
                if filter_key and isinstance(data_obj, dict):
                    filtered_data = {k: v for k, v in data_obj.items() if filter_key in str(k)}
                    return json.dumps(filtered_data, ensure_ascii=False)
                return json.dumps(data_obj, ensure_ascii=False)
            else:
                return json.dumps(data_obj, ensure_ascii=False)

        except Exception as e:
            logger.error(f"数据转换工具执行失败: {e}")
            return json.dumps({'error': str(e)}, ensure_ascii=False)

    def _flatten_data(self, data: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """扁平化数据结构"""
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.extend(self._flatten_data(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.extend(self._flatten_data(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
        else:
            return {parent_key: data} if parent_key else {"value": data}

        return dict(items)


class QualityCheckTool(BaseTool):
    """质量检查工具"""
    name = "quality_check"
    description = "检查内容质量"

    def _run(self, content: str, check_type: str = "general") -> Dict[str, Any]:
        """同步执行质量检查"""
        return asyncio.run(self._arun(content, check_type))

    async def _arun(self, content: str, check_type: str = "general") -> Dict[str, Any]:
        """异步执行质量检查"""
        try:
            if check_type == "general":
                return self._general_quality_check(content)
            elif check_type == "readability":
                return self._readability_check(content)
            elif check_type == "completeness":
                return self._completeness_check(content)
            else:
                return {'error': f'不支持的检查类型: {check_type}'}
        except Exception as e:
            logger.error(f"质量检查工具执行失败: {e}")
            return {'error': str(e)}

    def _general_quality_check(self, content: str) -> Dict[str, Any]:
        """通用质量检查"""
        score = 0.0
        issues = []

        # 长度检查
        if len(content) == 0:
            issues.append("内容为空")
        elif len(content) < 50:
            issues.append("内容过短")
            score += 0.2
        elif 50 <= len(content) <= 1000:
            score += 0.4
        elif len(content) > 1000:
            score += 0.3

        # 结构检查
        if any(punct in content for punct in ['。', '！', '？', '.', '!', '?']):
            score += 0.3
        else:
            issues.append("缺少句子结束标点")

        # 格式检查
        if '：' in content or ':' in content:
            score += 0.2

        return {
            'score': min(score, 1.0),
            'quality_level': 'high' if score > 0.8 else 'medium' if score > 0.5 else 'low',
            'issues': issues,
            'length': len(content)
        }

    def _readability_check(self, content: str) -> Dict[str, Any]:
        """可读性检查"""
        sentences = content.count('。') + content.count('！') + content.count('？') + \
                   content.count('.') + content.count('!') + content.count('?')
        words = len(content.replace(' ', ''))

        if sentences == 0:
            avg_words_per_sentence = words
        else:
            avg_words_per_sentence = words / sentences

        readability_score = 1.0
        if avg_words_per_sentence > 50:
            readability_score = 0.3
        elif avg_words_per_sentence > 30:
            readability_score = 0.6
        elif avg_words_per_sentence > 15:
            readability_score = 0.8

        return {
            'readability_score': readability_score,
            'sentences': sentences,
            'words': words,
            'avg_words_per_sentence': avg_words_per_sentence,
            'suggestion': '句子偏长，建议适当分句' if avg_words_per_sentence > 20 else '可读性良好'
        }

    def _completeness_check(self, content: str) -> Dict[str, Any]:
        """完整性检查"""
        completeness_score = 0.0
        missing_elements = []

        # 检查是否包含数字
        if any(char.isdigit() for char in content):
            completeness_score += 0.3
        else:
            missing_elements.append("缺少数据支撑")

        # 检查是否包含时间信息
        time_indicators = ['年', '月', '日', '2023', '2024', '2025']
        if any(indicator in content for indicator in time_indicators):
            completeness_score += 0.3
        else:
            missing_elements.append("缺少时间信息")

        # 检查是否包含专业术语
        professional_terms = ['分析', '报告', '数据', '统计', '趋势', '增长', '下降']
        if any(term in content for term in professional_terms):
            completeness_score += 0.4
        else:
            missing_elements.append("缺少专业分析")

        return {
            'completeness_score': completeness_score,
            'missing_elements': missing_elements,
            'is_complete': completeness_score > 0.6
        }


class FileOperationTool(BaseTool):
    """文件操作工具"""
    name = "file_operation"
    description = "执行文件操作"

    def _run(self, operation: str, file_path: str, content: str = None) -> Dict[str, Any]:
        """同步执行文件操作"""
        return asyncio.run(self._arun(operation, file_path, content))

    async def _arun(self, operation: str, file_path: str, content: str = None) -> Dict[str, Any]:
        """异步执行文件操作"""
        try:
            if operation == "read":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return {'content': f.read()}
            elif operation == "write":
                if content is None:
                    return {'error': '写入内容不能为空'}
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {'success': True, 'message': '文件写入成功'}
            elif operation == "exists":
                return {'exists': os.path.exists(file_path)}
            elif operation == "size":
                if os.path.exists(file_path):
                    return {'size': os.path.getsize(file_path)}
                else:
                    return {'error': '文件不存在'}
            elif operation == "delete":
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return {'success': True, 'message': '文件删除成功'}
                else:
                    return {'error': '文件不存在'}
            else:
                return {'error': f'不支持的操作: {operation}'}
        except Exception as e:
            logger.error(f"文件操作工具执行失败: {e}")
            return {'error': str(e)}


# 预定义工具实例
PREDEFINED_TOOLS = {
    'search': SearchTool(),
    'vector_search': VectorSearchTool(),
    'graph_search': GraphSearchTool(),
    'document_upload': DocumentUploadTool(),
    'document_extract': DocumentExtractTool(),
    'document_analyze': DocumentAnalyzeTool(),
    'ragas_evaluation': RAGASEvaluationTool(),
    'cache': CacheTool(),
    'data_transform': DataTransformTool(),
    'quality_check': QualityCheckTool(),
    'file_operation': FileOperationTool()
}


def get_tool(tool_name: str) -> Optional[BaseTool]:
    """获取预定义工具"""
    return PREDEFINED_TOOLS.get(tool_name)


def register_tool(name: str, tool: BaseTool) -> None:
    """注册自定义工具"""
    PREDEFINED_TOOLS[name] = tool
    logger.info(f"注册工具: {name}")


def list_tools() -> List[str]:
    """列出所有可用工具"""
    return list(PREDEFINED_TOOLS.keys())