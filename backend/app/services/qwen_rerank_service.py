"""
Qwen3 Rerank服务
使用通义千问的重排序API
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)


@dataclass
class QwenRerankConfig:
    """Qwen重排序服务配置"""
    api_key: str = "sk-5233a3a4b1a24426b6846a432794bbe2"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-rerank"
    top_k: int = 10
    max_length: int = 512
    timeout: int = 30


class QwenRerankService:
    """Qwen3重排序服务"""

    def __init__(self, config: Optional[QwenRerankConfig] = None):
        self.config = config or QwenRerankConfig()
        self.client = None

    async def initialize(self):
        """初始化服务"""
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        logger.info(f"Qwen Rerank服务初始化完成 - 模型: {self.config.model}")

    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前K个结果
            return_documents: 是否返回文档内容

        Returns:
            List[(原始索引, 重排序分数)]
        """
        if not self.client:
            await self.initialize()

        try:
            # 调用Qwen rerank API
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.config.model,
                "input": {
                    "query": query,
                    "documents": documents[:self.config.top_k]
                },
                "parameters": {
                    "return_documents": return_documents,
                    "top_n": top_k or self.config.top_k
                }
            }

            logger.info(f"调用Qwen rerank API - 查询: {query[:50]}..., 文档数: {len(documents)}")

            response = await self.client.post(
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                headers=headers,
                json=payload
            )

            response.raise_for_status()
            result = response.json()

            # 解析结果 - 正确的格式是 output.results
            if "output" in result and "results" in result["output"]:
                reranked = []
                for item in result["output"]["results"]:
                    index = item.get("index", 0)
                    score = item.get("relevance_score", 0.0)
                    reranked.append((index, score))

                logger.info(f"重排序完成 - 返回 {len(reranked)} 个结果")
                return reranked
            else:
                logger.warning(f"API返回格式异常: {result}")
                # 返回原始顺序
                return [(i, 1.0 - i * 0.01) for i in range(len(documents))]

        except Exception as e:
            logger.error(f"Qwen rerank调用失败: {e}")
            # 降级：返回原始顺序
            return [(i, 1.0 - i * 0.01) for i in range(len(documents))]

    async def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        重排序包含元数据的文档

        Args:
            query: 查询文本
            documents: 文档列表，每个文档包含content和metadata
            top_k: 返回前K个结果

        Returns:
            重排序后的文档列表
        """
        # 提取文档内容
        contents = [doc.get("content", "") for doc in documents]

        # 调用重排序
        reranked_indices = await self.rerank(query, contents, top_k)

        # 重新组织文档
        result = []
        for original_idx, score in reranked_indices:
            if original_idx < len(documents):
                doc = documents[original_idx].copy()
                doc["rerank_score"] = score
                result.append(doc)

        return result


# 全局单例
_qwen_rerank_service = None


async def get_qwen_rerank_service() -> QwenRerankService:
    """获取Qwen Rerank服务单例"""
    global _qwen_rerank_service
    if _qwen_rerank_service is None:
        _qwen_rerank_service = QwenRerankService()
        await _qwen_rerank_service.initialize()
    return _qwen_rerank_service
