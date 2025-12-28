"""
多模型集成重排序系统
集成多个Reranker模型，提升检索结果排序质量
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RerankerType(Enum):
    """Reranker类型"""
    QWEN_RERANK = "qwen_rerank"
    LLM_RERANK = "llm_rerank"
    CROSS_ENCODER = "cross_encoder"
    BGE_RERANK = "bge_rerank"
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class RerankResult:
    """重排序结果"""
    doc_id: str
    original_score: float
    rerank_score: float
    final_score: float
    original_rank: int
    new_rank: int
    model_contributions: Dict[str, float]  # 各模型贡献


class EnsembleReranker:
    """集成重排序器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化集成重排序器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 初始化各个Reranker
        self.rerankers = {}

        # Qwen Reranker
        if self.config.get('enable_qwen_rerank', True):
            from app.services.reranking.qwen_reranker import QwenReranker
            self.rerankers[RerankerType.QWEN_RERANK] = QwenReranker(
                api_key=self.config.get('qwen_api_key'),
                base_url=self.config.get('qwen_base_url')
            )

        # LLM Reranker
        if self.config.get('enable_llm_rerank', True):
            from app.services.reranking.llm_reranker import LLMReranker
            self.rerankers[RerankerType.LLM_RERANK] = LLMReranker(
                model_name=self.config.get('llm_model', 'gpt-3.5-turbo')
            )

        # Cross Encoder Reranker
        if self.config.get('enable_cross_encoder', False):
            from app.services.reranking.cross_encoder_reranker import CrossEncoderReranker
            self.rerankers[RerankerType.CROSS_ENCODER] = CrossEncoderReranker(
                model_name=self.config.get('cross_encoder_model', 'BAAI/bge-reranker-v2-m3')
            )

        # BGE Reranker
        if self.config.get('enable_bge_rerank', False):
            from app.services.reranking.bge_reranker import BGEReranker
            self.rerankers[RerankerType.BGE_RERANK] = BGEReranker()

        # 语义相似度Reranker
        if self.config.get('enable_semantic_similarity', True):
            from app.services.reranking.semantic_reranker import SemanticReranker
            self.rerankers[RerankerType.SEMANTIC_SIMILARITY] = SemanticReranker()

        # 学习权重管理器
        self.weight_learner = WeightLearner()

        logger.info(f"Initialized {len(self.rerankers)} rerankers: {list(self.rerankers.keys())}")

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        query_type: Optional[str] = None
    ) -> List[RerankResult]:
        """
        集成重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前K个结果
            query_type: 查询类型（用于选择权重）

        Returns:
            重排序结果列表
        """
        if not documents:
            return []

        if len(documents) == 1:
            # 只有一个文档，无需重排序
            return [RerankResult(
                doc_id=str(documents[0].get('id', 0)),
                original_score=documents[0].get('score', 0.0),
                rerank_score=documents[0].get('score', 0.0),
                final_score=documents[0].get('score', 0.0),
                original_rank=0,
                new_rank=0,
                model_contributions={}
            )]

        # 并行执行所有Reranker
        rerank_tasks = [
            self._run_reranker(reranker_type, query, documents)
            for reranker_type in self.rerankers.keys()
        ]

        rerank_results_list = await asyncio.gather(*rerank_tasks, return_exceptions=True)

        # 收集成功的结果
        all_scores = {}  # doc_id -> {reranker_type: score}
        doc_count = len(documents)

        for reranker_type, result in zip(self.rerankers.keys(), rerank_results_list):
            if isinstance(result, Exception):
                logger.warning(f"{reranker_type} reranking failed: {result}")
                continue

            for doc_id, score in result.items():
                if doc_id not in all_scores:
                    all_scores[doc_id] = {}
                all_scores[doc_id][reranker_type] = score

        # 获取权重（基于查询类型或学习权重）
        weights = await self._get_weights(query_type)

        # 计算加权分数
        rerank_results = []

        for i, doc in enumerate(documents):
            doc_id = str(doc.get('id', i))
            original_score = doc.get('score', 0.0)
            original_rank = i

            # 获取该文档的各模型分数
            doc_scores = all_scores.get(doc_id, {})

            # 计算加权重排序分数
            weighted_score = 0.0
            total_weight = 0.0
            model_contributions = {}

            for reranker_type, weight in weights.items():
                if reranker_type in doc_scores:
                    score = doc_scores[reranker_type]
                    contribution = score * weight
                    weighted_score += contribution
                    total_weight += weight
                    model_contributions[reranker_type.value] = contribution

            # 归一化
            if total_weight > 0:
                rerank_score = weighted_score / total_weight
            else:
                rerank_score = original_score

            # 综合分数：原始分数和重排序分数的加权
            final_score = (
                original_score * self.config.get('original_weight', 0.3) +
                rerank_score * self.config.get('rerank_weight', 0.7)
            )

            rerank_results.append({
                'doc_id': doc_id,
                'original_score': original_score,
                'rerank_score': rerank_score,
                'final_score': final_score,
                'original_rank': original_rank,
                'model_contributions': model_contributions
            })

        # 按最终分数排序
        rerank_results.sort(key=lambda x: x['final_score'], reverse=True)

        # 更新新排名
        final_results = []
        for new_rank, result in enumerate(rerank_results):
            final_results.append(RerankResult(
                doc_id=result['doc_id'],
                original_score=result['original_score'],
                rerank_score=result['rerank_score'],
                final_score=result['final_score'],
                original_rank=result['original_rank'],
                new_rank=new_rank,
                model_contributions=result['model_contributions']
            ))

        # 截取top_k
        if top_k is not None:
            final_results = final_results[:top_k]

        return final_results

    async def _run_reranker(
        self,
        reranker_type: RerankerType,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """运行单个Reranker"""
        reranker = self.rerankers[reranker_type]
        return await reranker.rerank(query, documents)

    async def _get_weights(self, query_type: Optional[str] = None) -> Dict[RerankerType, float]:
        """获取权重配置"""
        # 如果有查询类型，使用学习权重
        if query_type:
            learned_weights = await self.weight_learner.get_weights(query_type)
            if learned_weights:
                return learned_weights

        # 否则使用默认权重
        return {
            RerankerType.QWEN_RERANK: self.config.get('qwen_weight', 0.3),
            RerankerType.LLM_RERANK: self.config.get('llm_weight', 0.25),
            RerankerType.CROSS_ENCODER: self.config.get('cross_encoder_weight', 0.2),
            RerankerType.BGE_RERANK: self.config.get('bge_weight', 0.15),
            RerankerType.SEMANTIC_SIMILARITY: self.config.get('semantic_weight', 0.1),
        }

    async def update_weights_with_feedback(
        self,
        query_type: str,
        reranker_performances: Dict[str, float]
    ):
        """基于反馈更新权重"""
        await self.weight_learner.update_weights(query_type, reranker_performances)


class WeightLearner:
    """权重学习器"""

    def __init__(self):
        # 存储各查询类型的权重
        self.type_weights = {}  # query_type -> {reranker_type: weight}

        # 存储性能历史
        self.performance_history = {}  # query_type -> {reranker_type: [scores]}

    async def get_weights(self, query_type: str) -> Optional[Dict[RerankerType, float]]:
        """获取学习到的权重"""
        if query_type in self.type_weights:
            # 将字符串类型转换为枚举
            weights_str = self.type_weights[query_type]
            weights_enum = {}
            for type_str, weight in weights_str.items():
                try:
                    reranker_type = RerankerType(type_str)
                    weights_enum[reranker_type] = weight
                except ValueError:
                    continue
            return weights_enum
        return None

    async def update_weights(
        self,
        query_type: str,
        performances: Dict[str, float]
    ):
        """更新权重（基于性能）"""
        # 记录性能历史
        if query_type not in self.performance_history:
            self.performance_history[query_type] = {}

        for reranker_str, performance in performances.items():
            if reranker_str not in self.performance_history[query_type]:
                self.performance_history[query_type][reranker_str] = []

            self.performance_history[query_type][reranker_str].append(performance)

            # 只保留最近100次
            if len(self.performance_history[query_type][reranker_str]) > 100:
                self.performance_history[query_type][reranker_str].pop(0)

        # 计算平均性能
        avg_performances = {}
        for reranker_str, scores in self.performance_history[query_type].items():
            avg_performances[reranker_str] = np.mean(scores)

        # 计算权重（性能归一化）
        total_performance = sum(avg_performances.values())

        if total_performance > 0:
            weights = {
                reranker_str: perf / total_performance
                for reranker_str, perf in avg_performances.items()
            }
        else:
            # 均匀权重
            weights = {
                reranker_str: 1.0 / len(avg_performances)
                for reranker_str in avg_performances
            }

        # 保存权重
        self.type_weights[query_type] = weights

        logger.info(f"Updated weights for {query_type}: {weights}")


# 单独的Reranker实现
class QwenReranker:
    """Qwen Reranker（使用通义千问API）"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key
        self.base_url = base_url

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """调用Qwen Rerank API"""
        try:
            import httpx

            # 准备文档文本
            doc_texts = [doc.get('content', '')[:1000] for doc in documents]  # 限制长度

            payload = {
                "model": "qwen3-rerank",
                "input": {
                    "query": query,
                    "documents": doc_texts
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(doc_texts)
                }
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/services/rerank/text-rerank/text-rerank",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

            # 解析结果
            scores = {}
            for item in result.get("output", {}).get("results", []):
                index = item.get("index", 0)
                score = item.get("relevance_score", 0.0)
                doc_id = str(documents[index].get('id', index))
                scores[doc_id] = float(score)

            return scores

        except Exception as e:
            logger.error(f"Qwen reranking failed: {e}")
            # 返回原始分数
            return {str(doc.get('id', i)): doc.get('score', 0.0) for i, doc in enumerate(documents)}


class SemanticReranker:
    """语义相似度Reranker"""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """基于语义相似度重排序"""
        try:
            # 编码查询
            query_embedding = self.model.encode([query])

            # 编码文档
            doc_texts = [doc.get('content', '') for doc in documents]
            doc_embeddings = self.model.encode(doc_texts)

            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

            scores = {
                str(documents[i].get('id', i)): float(similarities[i])
                for i in range(len(documents))
            }

            return scores

        except Exception as e:
            logger.error(f"Semantic reranking failed: {e}")
            return {str(doc.get('id', i)): doc.get('score', 0.0) for i, doc in enumerate(documents)}


class LLMReranker:
    """LLM Reranker（使用LLM进行智能排序）"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """使用LLM重新排序"""
        try:
            # 构建prompt
            doc_descriptions = []
            for i, doc in enumerate(documents):
                content = doc.get('content', '')[:200]  # 截断
                doc_descriptions.append(f"{i}. {content}")

            prompt = f"""请根据查询对以下文档进行相关性排序。

查询：{query}

文档列表：
{chr(10).join(doc_descriptions)}

请返回排序后的文档索引（从0开始），用逗号分隔，最相关的排在前面。
只返回数字序列，不要其他内容。"""

            # TODO: 调用LLM API
            # 这里需要根据实际的LLM服务进行实现

            # 暂时返回原始分数
            return {str(doc.get('id', i)): doc.get('score', 0.0) for i, doc in enumerate(documents)}

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return {str(doc.get('id', i)): doc.get('score', 0.0) for i, doc in enumerate(documents)}
