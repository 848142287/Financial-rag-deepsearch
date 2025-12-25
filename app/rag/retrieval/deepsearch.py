"""
DeepSearch - 深度多轮检索
整合多种检索策略，实现迭代优化和结果融合
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .lightrag import LightRAG
from .graphrag import GraphRAG


@dataclass
class QueryExpansion:
    """查询扩展"""
    original_query: str
    expanded_queries: List[str]
    expansion_method: str


@dataclass
class DeepSearchConfig:
    """DeepSearch配置"""
    max_iterations: int = 3  # 最大迭代次数
    top_k_per_strategy: int = 10  # 每个策略返回的结果数
    final_top_k: int = 5  # 最终返回的结果数
    use_query_expansion: bool = True
    use_reranking: bool = True
    similarity_threshold: float = 0.6
    diversity_threshold: float = 0.7


class DeepSearch:
    """深度多轮检索系统"""

    def __init__(
        self,
        lightrag: LightRAG,
        graphrag: GraphRAG,
        config: DeepSearchConfig = None
    ):
        """
        初始化DeepSearch

        Args:
            lightrag: LightRAG实例
            graphrag: GraphRAG实例
            config: DeepSearch配置
        """
        self.lightrag = lightrag
        self.graphrag = graphrag
        self.config = config or DeepSearchConfig()

        # 加载重排序模型（可选）
        self.reranker = None
        if TRANSFORMERS_AVAILABLE and config.use_reranking:
            try:
                self.reranker = SentenceTransformer('distiluse-base-multilingual-cased-v2')
            except:
                print("Warning: Failed to load reranker, will skip reranking")

    def expand_query(self, query: str) -> QueryExpansion:
        """
        查询扩展

        Args:
            query: 原始查询

        Returns:
            扩展后的查询
        """
        expanded_queries = []

        # 策略1: 同义词扩展（简化版）
        # 实际应用中可以使用词典或WordNet
        synonyms_map = {
            "风格": ["风格类型", "投资风格", "股票风格"],
            "轮动": ["轮换", "切换", "交替"],
            "股票": ["个股", "证券", "权益类资产"],
            "债券": ["固定收益", "固收", "债权"],
            "基金": ["开放式基金", "封闭式基金", "ETF"],
        }

        for term, synonyms in synonyms_map.items():
            if term in query:
                for synonym in synonyms:
                    expanded_query = query.replace(term, synonym)
                    expanded_queries.append(expanded_query)

        # 策略2: 添加上下文相关词
        # 根据金融领域添加相关词汇
        finance_context_terms = {
            "投资": ["资产配置", "投资策略", "风险管理"],
            "风险": ["波动率", "回撤", "风险控制"],
            "收益": ["回报率", "收益率", "超额收益"],
        }

        for term, context_terms in finance_context_terms.items():
            if term in query:
                for context_term in context_terms:
                    expanded_queries.append(f"{query} {context_term}")

        # 去重
        expanded_queries = list(set(expanded_queries))

        return QueryExpansion(
            original_query=query,
            expanded_queries=expanded_queries[:5],  # 限制扩展数量
            expansion_method="synonym_and_context"
        )

    def hybrid_retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        混合检索（整合LightRAG和GraphRAG）

        Args:
            query: 查询文本
            query_embedding: 查询向量（可选）

        Returns:
            检索结果
        """
        all_results = []
        result_scores = defaultdict(float)

        # 策略1: LightRAG检索
        try:
            lightrag_result = self.lightrag.retrieve(query, query_embedding)
            for result in lightrag_result.get("results", []):
                doc_id = result["id"]
                # 计算综合分数：相似度 * 1.0（LightRAG权重）
                score = result.get("score", 0) * 1.0
                result_scores[doc_id] += score

                if doc_id not in [r["id"] for r in all_results]:
                    all_results.append({
                        **result,
                        "strategies": ["lightrag"],
                        "scores": {"lightrag": score}
                    })
        except Exception as e:
            print(f"LightRAG retrieval failed: {e}")

        # 策略2: GraphRAG检索
        try:
            graphrag_result = self.graphrag.retrieve(query)
            for result in graphrag_result.get("results", []):
                doc_id = result["id"]
                # 计算综合分数：相似度 * 0.8（GraphRAG权重）
                score = result.get("score", 0) * 0.8
                result_scores[doc_id] += score

                # 更新已存在的结果
                existing = next((r for r in all_results if r["id"] == doc_id), None)
                if existing:
                    existing["strategies"].append("graphrag")
                    existing["scores"]["graphrag"] = score
                    existing["score"] = result_scores[doc_id]
                else:
                    all_results.append({
                        **result,
                        "strategies": ["graphrag"],
                        "scores": {"graphrag": score},
                        "score": score
                    })
        except Exception as e:
            print(f"GraphRAG retrieval failed: {e}")

        # 排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "query": query,
            "results": all_results[:self.config.top_k_per_strategy],
            "total_retrieved": len(all_results),
            "strategies_used": ["lightrag", "graphrag"],
            "method": "hybrid"
        }

    def iterative_retrieve(
        self,
        query: str,
        max_iterations: int = None
    ) -> Dict[str, Any]:
        """
        迭代检索：多轮检索累积结果

        Args:
            query: 查询文本
            max_iterations: 最大迭代次数

        Returns:
            最终检索结果
        """
        max_iterations = max_iterations or self.config.max_iterations
        all_results = []
        seen_doc_ids = set()

        current_query = query

        for iteration in range(max_iterations):
            # 执行检索
            iteration_result = self.hybrid_retrieve(current_query)

            # 累积新结果
            new_results = [
                r for r in iteration_result.get("results", [])
                if r["id"] not in seen_doc_ids
            ]

            for result in new_results:
                seen_doc_ids.add(result["id"])
                all_results.append(result)

            # 检查是否需要继续迭代
            if len(all_results) >= self.config.final_top_k * 2:
                # 已有足够结果，停止迭代
                break

            # 如果有扩展查询，使用扩展查询继续检索
            if iteration == 0 and self.config.use_query_expansion:
                expansion = self.expand_query(query)
                if expansion.expanded_queries:
                    # 使用第一个扩展查询继续
                    current_query = expansion.expanded_queries[0]
                else:
                    break
            else:
                break

        # 重排序
        if self.reranker and len(all_results) > 1:
            all_results = self._rerank_results(query, all_results)

        # 去重并排序
        unique_results = []
        seen = set()
        for result in all_results:
            doc_id = result["id"]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "query": query,
            "results": unique_results[:self.config.final_top_k],
            "total_retrieved": len(unique_results),
            "iterations": iteration + 1,
            "method": "deepsearch"
        }

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        智能检索：根据查询复杂度自动选择检索策略

        Args:
            query: 查询文本
            query_embedding: 查询向量（可选）
            mode: 检索模式（"lightrag", "graphrag", "deepsearch", "auto"）

        Returns:
            检索结果
        """
        start_time = time.time()

        # 自动模式：根据查询特征选择策略
        if mode == "auto":
            mode = self._select_retrieval_strategy(query)

        # 执行检索
        if mode == "lightrag":
            result = self.lightrag.retrieve(query, query_embedding)
        elif mode == "graphrag":
            result = self.graphrag.retrieve(query)
        elif mode == "deepsearch":
            result = self.iterative_retrieve(query)
        else:  # hybrid
            result = self.hybrid_retrieve(query, query_embedding)

        retrieval_time = time.time() - start_time

        # 添加检索时间
        result["retrieval_time"] = retrieval_time

        # 生成结果摘要
        result["summary"] = self._generate_result_summary(result)

        return result

    def _select_retrieval_strategy(self, query: str) -> str:
        """
        根据查询特征选择最佳检索策略

        Args:
            query: 查询文本

        Returns:
            检索策略名称
        """
        # 简单的特征判断
        # 包含关系词的查询 -> GraphRAG
        relation_keywords = ["关系", "关联", "连接", "依赖", "影响"]
        if any(keyword in query for keyword in relation_keywords):
            return "graphrag"

        # 包含实体名称的查询 -> GraphRAG
        # 简化判断：大写字母开头的词可能是实体
        if any(word[0].isupper() for word in query.split() if len(word) > 1):
            return "graphrag"

        # 复杂查询 -> DeepSearch
        complex_keywords = ["分析", "比较", "总结", "评估", "影响"]
        if len(query) > 20 and any(keyword in query for keyword in complex_keywords):
            return "deepsearch"

        # 简单查询 -> LightRAG
        return "lightrag"

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        使用重排序模型重新排序结果

        Args:
            query: 查询文本
            results: 检索结果列表

        Returns:
            重排序后的结果列表
        """
        if not self.reranker:
            return results

        try:
            # 提取文本内容
            result_texts = [r.get("content", "") for r in results]

            # 计算查询与结果的相似度
            query_embedding = self.reranker.encode([query])
            result_embeddings = self.reranker.encode(result_texts)

            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, result_embeddings)[0]

            # 更新分数
            for i, result in enumerate(results):
                result["rerank_score"] = float(similarities[i])
                result["score"] = result.get("score", 0) * 0.7 + float(similarities[i]) * 0.3

            # 重新排序
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            return results

        except Exception as e:
            print(f"Reranking failed: {e}")
            return results

    def _generate_result_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成结果摘要

        Args:
            result: 检索结果

        Returns:
            摘要信息
        """
        results = result.get("results", [])

        if not results:
            return {
                "total_results": 0,
                "has_graph_context": False,
                "avg_score": 0.0
            }

        # 统计信息
        total_results = len(results)
        has_graph_context = any(
            "graphrag" in r.get("strategies", [])
            for r in results
        )
        avg_score = sum(r.get("score", 0) for r in results) / total_results

        # 策略分布
        strategy_counts = defaultdict(int)
        for r in results:
            for strategy in r.get("strategies", ["unknown"]):
                strategy_counts[strategy] += 1

        return {
            "total_results": total_results,
            "has_graph_context": has_graph_context,
            "avg_score": avg_score,
            "strategy_distribution": dict(strategy_counts)
        }

    def batch_retrieve(
        self,
        queries: List[str],
        mode: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        批量检索

        Args:
            queries: 查询列表
            mode: 检索模式

        Returns:
            检索结果列表
        """
        results = []

        for query in queries:
            try:
                result = self.retrieve(query, mode=mode)
                results.append(result)
            except Exception as e:
                print(f"Failed to retrieve for query '{query}': {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "results": [],
                    "total_retrieved": 0
                })

        return results
