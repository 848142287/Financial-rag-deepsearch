"""
RAG服务模块
提供检索增强生成的核心功能
"""

from typing import List, Dict, Any, Optional, Tuple
from ..milvus_service import MilvusService
from ..embedding_service import EmbeddingService
from ...core.database import get_db
from ...models.document import Document
from sqlalchemy.orm import Session
import json
import logging

logger = logging.getLogger(__name__)

class RAGService:
    """RAG检索增强生成服务"""

    def __init__(self):
        self.milvus_service = MilvusService()
        self.embedding_service = EmbeddingService()

    async def retrieve_relevant_documents(
        self,
        query: str,
        limit: int = 10,
        document_ids: Optional[List[int]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档片段

        Args:
            query: 查询文本
            limit: 返回结果数量
            document_ids: 限定文档ID范围
            score_threshold: 相似度阈值

        Returns:
            相关文档片段列表
        """
        try:
            # 生成查询向量
            query_embedding = await self.embedding_service.get_embedding(query)

            # 向量搜索
            search_results = await self.milvus_service.search(
                query_embedding=query_embedding,
                limit=limit,
                document_ids=document_ids,
                score_threshold=score_threshold
            )

            # 补充文档元数据
            results = []
            for result in search_results:
                document_id = result.get('document_id')
                chunk_content = result.get('content', '')
                score = result.get('score', 0.0)

                # 获取文档元数据
                doc_info = await self._get_document_info(document_id)

                results.append({
                    'document_id': document_id,
                    'chunk_id': result.get('chunk_id'),
                    'content': chunk_content,
                    'score': score,
                    'metadata': result.get('metadata', {}),
                    'document': doc_info
                })

            return results

        except Exception as e:
            logger.error(f"检索相关文档失败: {str(e)}")
            return []

    async def _get_document_info(self, document_id: int) -> Optional[Dict[str, Any]]:
        """获取文档基本信息"""
        try:
            db = next(get_db())
            document = db.query(Document).filter(Document.id == document_id).first()

            if document:
                return {
                    'id': document.id,
                    'title': document.title,
                    'description': document.description,
                    'file_type': document.file_type,
                    'status': document.status,
                    'created_at': document.created_at.isoformat() if document.created_at else None
                }
            return None

        except Exception as e:
            logger.error(f"获取文档信息失败: {str(e)}")
            return None
        finally:
            if 'db' in locals():
                db.close()

    async def generate_context(
        self,
        query: str,
        max_context_length: int = 4000,
        **kwargs
    ) -> str:
        """
        生成检索上下文

        Args:
            query: 查询文本
            max_context_length: 最大上下文长度
            **kwargs: 其他检索参数

        Returns:
            拼接的上下文文本
        """
        try:
            # 检索相关文档
            results = await self.retrieve_relevant_documents(query, **kwargs)

            if not results:
                return ""

            # 按相关性排序并构建上下文
            context_parts = []
            current_length = 0

            for result in sorted(results, key=lambda x: x['score'], reverse=True):
                content = result['content']
                title = result.get('document', {}).get('title', '未知文档')

                # 格式化片段
                formatted_chunk = f"[{title}]\n{content}\n\n"

                # 检查长度限制
                if current_length + len(formatted_chunk) > max_context_length:
                    # 截断最后一个片段
                    remaining = max_context_length - current_length - 10
                    if remaining > 100:  # 只保留有意义的片段
                        truncated_content = content[:remaining] + "..."
                        formatted_chunk = f"[{title}]\n{truncated_content}\n\n"
                        context_parts.append(formatted_chunk)
                    break

                context_parts.append(formatted_chunk)
                current_length += len(formatted_chunk)

            return "".join(context_parts)

        except Exception as e:
            logger.error(f"生成上下文失败: {str(e)}")
            return ""

    async def search_and_answer(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        检索并生成回答

        Args:
            query: 查询文本
            **kwargs: 其他参数

        Returns:
            包含检索结果和上下文的字典
        """
        try:
            # 检索相关文档
            results = await self.retrieve_relevant_documents(query, **kwargs)

            # 生成上下文
            context = await self.generate_context(query, **kwargs)

            return {
                'query': query,
                'results': results,
                'context': context,
                'total_results': len(results)
            }

        except Exception as e:
            logger.error(f"检索回答失败: {str(e)}")
            return {
                'query': query,
                'results': [],
                'context': "",
                'total_results': 0,
                'error': str(e)
            }

# 全局RAG服务实例
rag_service = RAGService()