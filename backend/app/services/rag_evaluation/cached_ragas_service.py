"""
带Redis缓存的Ragas评估服务
- 缓存检索结果到Redis
- 基于余弦相似度(>0.8)直接返回缓存结果
- 生成检索评估报告

v2.0 - 集成evaluation_common公共模块
- 使用统一的EvaluationCacheManager
- 集成链路追踪
- 统一错误处理
"""

import json
from app.core.structured_logging import get_structured_logger
from datetime import datetime
import numpy as np

from app.services.embeddings.unified_embedding_service import get_embedding_service
# 导入evaluation_common公共模块

logger = get_structured_logger(__name__)

class CachedRagasService:
    """带缓存的Ragas评估服务

    v2.0: 使用统一的EvaluationCacheManager
    """

    # 相似度阈值
    SIMILARITY_THRESHOLD = 0.8

    # 缓存TTL (秒) - 7天
    CACHE_TTL = 604800

    def __init__(self):
        """初始化服务"""
        self.embedding_service = get_embedding_service()
        self.cache_manager = evaluation_cache_manager  # 使用统一的缓存管理器
        self.ragas_evaluator = RAGASEvaluator()

    async def retrieve_with_cache(
        self,
        query: str,
        retrieval_func: callable,
        evaluate: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        带缓存的检索

        v2.0: 使用统一的EvaluationCacheManager

        Args:
            query: 检索问题
            retrieval_func: 检索函数，用于实际检索
            evaluate: 是否进行Ragas评估
            force_refresh: 是否强制刷新缓存

        Returns:
            检索结果和评估报告
        """
        # 创建追踪
        trace_id = await create_evaluation_trace(
            query=query,
            evaluation_type="cached_retrieval",
            metadata={'evaluate': evaluate, 'force_refresh': force_refresh}
        )

        try:
            # 1. 对查询进行向量化
            query_embedding = await self.embedding_service.encode_single(query)

            # 2. 如果不强制刷新，尝试从缓存获取
            if not force_refresh:
                async with trace_evaluation_stage(trace_id, "cache_lookup"):
                    cached_result = await self.cache_manager.get_by_similarity(
                        query=query,
                        query_embedding=query_embedding,
                        pattern="rag_cache:*",
                        min_similarity=self.SIMILARITY_THRESHOLD
                    )

                if cached_result:
                    logger.info(
                        f"从缓存返回相似查询结果 (相似度: {cached_result['similarity']:.3f})",
                        extra={'trace_id': trace_id}
                    )
                    await finish_evaluation_trace(trace_id, status="completed", results={'from_cache': True})

                    return {
                        "query": query,
                        "results": cached_result["data"]["results"],
                        "contexts": cached_result["data"]["contexts"],
                        "answer": cached_result["data"]["answer"],
                        "evaluation": cached_result["data"].get("evaluation"),
                        "from_cache": True,
                        "similarity": cached_result["similarity"]
                    }

            # 3. 执行实际检索
            logger.info("缓存未命中，执行实际检索...", extra={'trace_id': trace_id})
            async with trace_evaluation_stage(trace_id, "actual_retrieval"):
                retrieval_results = await retrieval_func(query)

            # 4. 提取上下文和生成答案
            contexts = [r.get('content', '') for r in retrieval_results[:10]]

            # 5. 如果需要，进行Ragas评估
            evaluation = None
            if evaluate:
                try:
                    async with trace_evaluation_stage(trace_id, "ragas_evaluation"):
                        # 假设有答案，如果没有可以生成一个
                        answer = self._generate_answer_from_contexts(query, contexts)
                        ragas_result = await self.ragas_evaluator.evaluate(
                            question=query,
                            answer=answer,
                            contexts=contexts
                        )
                        evaluation = self._serialize_evaluation(ragas_result)
                except Exception as e:
                    logger.error(f"Ragas评估失败: {e}", extra={'trace_id': trace_id})

            # 6. 保存到缓存 - 使用统一缓存管理器
            await self.cache_manager.set_evaluation(
                query=query,
                query_embedding=query_embedding.tolist(),
                results=retrieval_results,
                evaluation=evaluation,
                ttl=self.CACHE_TTL
            )

            await finish_evaluation_trace(trace_id, status="completed", results={'from_cache': False})

            return {
                "query": query,
                "results": retrieval_results,
                "contexts": contexts,
                "answer": contexts[0][:200] if contexts else "",
                "evaluation": evaluation,
                "from_cache": False
            }

        except CacheOperationError as e:
            logger.error(f"缓存操作失败: {e}", extra={'trace_id': trace_id})
            # 缓存失败不影响主流程，继续执行
            raise
        except Exception as e:
            logger.error(f"检索失败: {e}", extra={'trace_id': trace_id})
            await finish_evaluation_trace(trace_id, status="failed", results={'error': str(e)})
            raise

    async def batch_retrieve_with_cache(
        self,
        queries: List[str],
        retrieval_func: callable,
        evaluate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量检索（带缓存）

        Args:
            queries: 查询列表
            retrieval_func: 检索函数
            evaluate: 是否进行评估

        Returns:
            检索结果列表
        """
        results = []

        for query in queries:
            try:
                result = await self.retrieve_with_cache(
                    query=query,
                    retrieval_func=retrieval_func,
                    evaluate=evaluate
                )
                results.append(result)
            except Exception as e:
                logger.error(f"批量检索失败 ({query}): {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "from_cache": False
                })

        return results

    async def generate_evaluation_report(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成Ragas评估报告

        Args:
            evaluations: 评估结果列表

        Returns:
            评估报告
        """
        if not evaluations:
            return {"error": "没有评估数据"}

        # 统计指标
        total_evaluations = len(evaluations)
        cached_count = sum(1 for e in evaluations if e.get("from_cache", False))

        # 提取分数
        scores = {
            "faithfulness": [],
            "answer_relevance": [],
            "context_relevance": [],
            "context_recall": [],
            "answer_correctness": [],
            "overall": []
        }

        for eval_data in evaluations:
            evaluation = eval_data.get("evaluation", {})
            if evaluation and "results" in evaluation:
                for result in evaluation["results"]:
                    metric_name = result.get("metric", "")
                    score = result.get("score", 0)
                    if metric_name in scores:
                        scores[metric_name].append(score)

                if "overall_score" in evaluation:
                    scores["overall"].append(evaluation["overall_score"])

        # 计算平均分
        avg_scores = {}
        for metric, score_list in scores.items():
            if score_list:
                avg_scores[metric] = {
                    "mean": np.mean(score_list),
                    "std": np.std(score_list),
                    "min": np.min(score_list),
                    "max": np.max(score_list),
                    "count": len(score_list)
                }

        # 计算命中率
        cache_hit_rate = cached_count / total_evaluations if total_evaluations > 0 else 0

        # 评估是否达到85%目标
        target_score = 0.85
        overall_mean = avg_scores.get("overall", {}).get("mean", 0)
        target_met = overall_mean >= target_score

        report = {
            "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_evaluations": total_evaluations,
                "cached_count": cached_count,
                "cache_hit_rate": cache_hit_rate,
                "target_score": target_score,
                "target_met": target_met,
                "overall_score": overall_mean
            },
            "metrics": avg_scores,
            "recommendations": self._generate_recommendations(avg_scores, target_met)
        }

        # 保存报告到Redis
        await self._save_report_to_cache(report)

        return report

    def _generate_recommendations(
        self,
        scores: Dict[str, Dict[str, float]],
        target_met: bool
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if target_met:
            recommendations.append("✅ 整体表现优秀，已达到85%目标！")
            return recommendations

        # 分析各指标
        for metric, stats in scores.items():
            mean_score = stats.get("mean", 0)
            if mean_score < 0.7:
                recommendations.append(
                    f"⚠️ {metric} 分数较低 ({mean_score:.2%})，需要重点改进"
                )
            elif mean_score < 0.85:
                recommendations.append(
                    f"ℹ️ {metric} 分数 ({mean_score:.2%}) 接近目标，继续优化"
                )

        if not recommendations:
            recommendations.append("继续优化各项指标以达到85%目标")

        return recommendations

    async def _save_report_to_cache(self, report: Dict[str, Any]):
        """保存报告到缓存(使用redis_client,因为报告缓存不同于评估缓存)"""
        try:
            from app.core.redis_client import redis_client
            report_key = f"rag_report:{report['report_id']}"
            await redis_client.setex(
                report_key,
                86400,  # 24小时
                json.dumps(report, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"保存报告到缓存失败: {e}")

    def _generate_answer_from_contexts(
        self,
        query: str,
        contexts: List[str]
    ) -> str:
        """从上下文生成答案"""
        # 简单实现：合并前几个上下文
        combined = " ".join(contexts[:3])
        return combined[:500] if combined else "无法生成答案"

    def _serialize_evaluation(self, evaluation: RAGASEvaluation) -> Dict[str, Any]:
        """序列化评估结果"""
        return {
            "evaluation_id": evaluation.evaluation_id,
            "overall_score": evaluation.overall_score,
            "results": [
                {
                    "metric": r.metric.value,
                    "score": r.score,
                    "reasoning": r.reasoning,
                    "confidence": r.confidence
                }
                for r in evaluation.results
            ]
        }

# 全局实例
cached_ragas_service = CachedRagasService()

async def get_cached_ragas_service() -> CachedRagasService:
    """获取服务实例"""
    return cached_ragas_service
