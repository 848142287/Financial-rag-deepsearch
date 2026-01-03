"""
LTR中间件 - 自动为现有RAG服务添加LTR能力
通过装饰器模式，无需修改原有代码
"""

import functools
from app.core.structured_logging import get_structured_logger
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.services.agentic_rag.enhanced_retrieval_manager import get_enhanced_retrieval_with_ltr

logger = get_structured_logger(__name__)


class LTRMiddleware:
    """LTR中间件"""

    def __init__(self):
        self.enhanced_retrieval = get_enhanced_retrieval_with_ltr()
        self.enabled = True
        self.auto_collect_training_data = True

    async def enhance_rag_response(
        self,
        original_func,
        query: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        增强RAG响应，添加LTR排序

        Args:
            original_func: 原始RAG函数
            query: 用户查询
            *args, **kwargs: 其他参数

        Returns:
            增强后的响应
        """
        # 1. 调用原始函数
        response = await original_func(query, *args, **kwargs)

        # 2. 如果响应包含检索结果，应用LTR重排序
        if self._should_apply_ltr(response):
            try:
                # 提取检索结果
                documents = self._extract_documents(response)

                if documents:
                    # 应用LTR排序
                    user_id = kwargs.get('user_id') or response.get('user_id')
                    session_id = kwargs.get('session_id') or response.get('session_id')

                    reranked_docs = await self.enhanced_retrieval.retrieve_with_ltr(
                        query=query,
                        initial_results=documents,
                        user_id=user_id,
                        session_id=session_id,
                        top_k=len(documents)
                    )

                    # 更新响应
                    response = self._update_response_with_ltr(response, reranked_docs)

                    # 记录用于训练
                    if self.auto_collect_training_data:
                        await self.enhanced_retrieval.log_retrieval_for_training(
                            query=query,
                            retrieved_docs=reranked_docs,
                            user_id=user_id,
                            session_id=session_id,
                            retrieval_context={'method': 'ltr_middleware'}
                        )

                    logger.info("LTR中间件已应用排序")

            except Exception as e:
                logger.warning(f"LTR中间件应用失败: {e}，使用原始响应")

        return response

    def _should_apply_ltr(self, response: Dict[str, Any]) -> bool:
        """判断是否应该应用LTR"""
        if not self.enabled:
            return False

        # 检查响应是否包含文档列表
        return any(key in response for key in ['sources', 'documents', 'retrieved_docs'])

    def _extract_documents(self, response: Dict[str, Any]) -> Optional[List[Dict]]:
        """从响应中提取文档列表"""
        # 尝试不同的字段名
        for key in ['sources', 'documents', 'retrieved_docs', 'results']:
            if key in response and isinstance(response[key], list):
                return response[key]

        return None

    def _update_response_with_ltr(
        self,
        response: Dict[str, Any],
        reranked_docs: List[Dict]
    ) -> Dict[str, Any]:
        """用LTR排序后的文档更新响应"""
        # 更新文档列表
        for key in ['sources', 'documents', 'retrieved_docs', 'results']:
            if key in response:
                response[key] = reranked_docs
                break

        # 添加LTR元数据
        if 'metadata' not in response:
            response['metadata'] = {}

        response['metadata']['ltr_applied'] = True
        response['metadata']['ltr_timestamp'] = datetime.now().isoformat()

        # 如果reranked_docs包含ltr_score，更新置信度
        if reranked_docs and 'ltr_score' in reranked_docs[0]:
            avg_ltr_score = sum(doc.get('ltr_score', 0) for doc in reranked_docs) / len(reranked_docs)

            if 'trust_explanation' in response:
                response['trust_explanation']['ltr_enhanced'] = True
                response['trust_explanation']['ltr_score'] = round(avg_ltr_score, 3)

        return response


# 全局中间件实例
_ltr_middleware = None


def get_ltr_middleware() -> LTRMiddleware:
    """获取LTR中间件实例"""
    global _ltr_middleware
    if _ltr_middleware is None:
        _ltr_middleware = LTRMiddleware()
    return _ltr_middleware


def with_ltr(func):
    """
    装饰器：为RAG函数自动添加LTR能力

    使用方法：
    @with_ltr
    async def my_rag_function(query: str, **kwargs):
        # 原始RAG逻辑
        return response
    """
    @functools.wraps(func)
    async def wrapper(query: str, *args, **kwargs):
        middleware = get_ltr_middleware()
        return await middleware.enhance_rag_response(func, query, *args, **kwargs)

    return wrapper


# 便捷的包装函数
def enable_ltr_for_rag_service(rag_service_class):
    """
    为RAG服务类启用LTR

    使用方法：
    @enable_ltr_for_rag_service
    class MyRAGService:
        async def query(self, question: str):
            # 原始实现
            pass
    """
    original_methods = {}

    # 找到查询方法并包装
    for attr_name in dir(rag_service_class):
        attr = getattr(rag_service_class, attr_name)
        if callable(attr) and any(keyword in attr_name.lower()
                                  for keyword in ['query', 'search', 'retrieve', 'answer']):
            original_methods[attr_name] = attr
            # 替换为包装后的方法
            setattr(rag_service_class, attr_name, with_ltr(attr))

    return rag_service_class
