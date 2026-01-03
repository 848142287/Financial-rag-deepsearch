"""
检索性能优化模块
提供动态阈值调整和分页优化策略
从 swxy/backend 移植
"""

from app.core.structured_logging import get_structured_logger
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from app.core.base_optimizer import (
    BaseOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationStatus,
    OptimizationMetrics
)

logger = get_structured_logger(__name__)


class RetrievalOptimizer(BaseOptimizer):
    """
    检索优化器

    功能：
    - 动态阈值调整
    - 分页优化策略
    - 性能监控
    """

    optimizer_type = "retrieval"
    optimizer_version = "2.0.0"

    def __init__(self, config: Optional[OptimizationConfig] = None):
        super().__init__(config)

        # 配置
        self.RERANK_PAGE_LIMIT = 3  # 前3页使用重排序

        # 初始阈值
        self.initial_min_match = 0.3
        self.initial_similarity = 0.1

        # 降级阈值
        self.fallback_min_match = 0.1
        self.fallback_similarity = 0.17

        # 当前最佳参数
        self._best_thresholds = {
            "min_match": self.initial_min_match,
            "similarity": self.initial_similarity
        }

    async def optimize(
        self,
        objective_func: Optional[Callable[[Dict[str, Any]], float]] = None,
        initial_parameters: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        执行检索参数优化

        Args:
            objective_func: 目标函数（可选，用于自定义优化目标）
            initial_parameters: 初始参数
            constraints: 约束条件

        Returns:
            OptimizationResult
        """
        # 使用默认参数
        if initial_parameters is None:
            initial_parameters = {
                "min_match": self.initial_min_match,
                "similarity": self.initial_similarity
            }

        # 如果没有提供目标函数，使用默认的优化逻辑
        if objective_func is None:
            objective_func = self._default_objective

        return await self._execute_optimization(
            objective_func,
            initial_parameters,
            constraints,
            self._retrieval_optimization_logic
        )

    async def _retrieval_optimization_logic(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        initial_parameters: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> OptimizationResult:
        """检索优化的具体逻辑"""
        current_params = initial_parameters.copy()
        best_value = float('-inf')

        for iteration in range(1, self.config.max_iterations + 1):
            # 评估当前参数
            current_value = objective_func(current_params)

            # 记录指标
            improvement = current_value - best_value if best_value != float('-inf') else 0.0
            metrics = OptimizationMetrics(
                timestamp=self._get_now(),
                iteration=iteration,
                objective_value=current_value,
                current_parameters=current_params.copy(),
                improvement=improvement
            )
            self.record_metrics(metrics)

            # 更新最佳参数
            if current_value > best_value:
                best_value = current_value
                self._best_thresholds = current_params.copy()

            # 检查是否应该停止
            if self.should_stop(metrics):
                self.logger.info(f"优化在迭代 {iteration} 后收敛")
                break

            # 检查超时
            if self._check_timeout():
                self.logger.warning(f"优化在迭代 {iteration} 时超时")
                break

            # 调整参数（简单的网格搜索策略）
            current_params = self._adjust_parameters(current_params, iteration)

        return self._create_result(
            success=True,
            status=OptimizationStatus.COMPLETED,
            objective_value=best_value,
            best_parameters=self._best_thresholds
        )

    def _default_objective(self, parameters: Dict[str, Any]) -> float:
        """
        默认目标函数

        Args:
            parameters: 检索参数

        Returns:
            目标值（越高越好）
        """
        # 简单的目标：平衡召回率和精确度
        min_match = parameters.get("min_match", 0.3)
        similarity = parameters.get("similarity", 0.1)

        # 阈值不能太低（保证质量）也不能太高（保证召回）
        # 目标是找到平衡点
        balance_score = 1.0 - abs(min_match - 0.3) - abs(similarity - 0.2)

        return max(0.0, balance_score)

    def _adjust_parameters(self, parameters: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """调整参数（简单的搜索策略）"""
        new_params = parameters.copy()

        # 随着迭代次数增加，逐步减小调整幅度
        step_size = 0.1 / (1 + iteration // 10)

        # 调整 min_match
        if iteration % 2 == 0:
            new_params["min_match"] = max(0.0, min(1.0, parameters["min_match"] + step_size))
        else:
            new_params["min_match"] = max(0.0, min(1.0, parameters["min_match"] - step_size))

        # 调整 similarity
        if iteration % 3 == 0:
            new_params["similarity"] = max(0.0, min(1.0, parameters["similarity"] + step_size))
        else:
            new_params["similarity"] = max(0.0, min(1.0, parameters["similarity"] - step_size))

        return new_params

    def should_stop(self, metrics: OptimizationMetrics) -> bool:
        """判断是否应该停止优化"""
        # 如果改进很小，认为已经收敛
        if abs(metrics.improvement) < self.config.convergence_threshold:
            return True

        return False

    def _get_now(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now()

    def should_use_rerank(self, page: int, total_results: int) -> bool:
        """
        判断是否应该使用重排序

        Args:
            page: 当前页码
            total_results: 总结果数

        Returns:
            True表示使用重排序
        """
        # 前3页且结果不为空时使用重排序
        return page <= self.RERANK_PAGE_LIMIT and total_results > 0

    def adjust_thresholds(
        self,
        query: str,
        initial_results: int,
        max_retries: int = 1
    ) -> Dict[str, float]:
        """
        动态调整检索阈值

        Args:
            query: 查询文本
            initial_results: 初始结果数量
            max_retries: 最大重试次数

        Returns:
            调整后的阈值字典
        """
        thresholds = {
            "min_match": self.initial_min_match,
            "similarity": self.initial_similarity
        }

        # 如果初始结果为0，降低阈值重试
        if initial_results == 0 and max_retries > 0:
            logger.info(f"初始结果为0，降低阈值重试")
            thresholds["min_match"] = self.fallback_min_match
            thresholds["similarity"] = self.fallback_similarity

        return thresholds

    def optimize_pagination(
        self,
        page: int,
        page_size: int,
        total_results: int,
        use_rerank: bool
    ) -> Dict[str, Any]:
        """
        优化分页参数

        Args:
            page: 当前页码
            page_size: 每页大小
            total_results: 总结果数
            use_rerank: 是否使用重排序

        Returns:
            优化后的分页参数
        """
        # 计算偏移量
        offset = (page - 1) * page_size

        # 如果使用重排序，获取更多结果用于重排序
        if use_rerank:
            # 获取前RERANK_PAGE_LIMIT页的结果用于重排序
            fetch_size = page_size * self.RERANK_PAGE_LIMIT
        else:
            fetch_size = page_size

        # 确保不超过总结果数
        if offset + fetch_size > total_results:
            fetch_size = max(0, total_results - offset)

        return {
            "offset": offset,
            "limit": page_size,
            "fetch_size": fetch_size,
            "use_rerank": use_rerank
        }

    def calculate_fusion_weights(
        self,
        query_type: str = "default"
    ) -> Dict[str, float]:
        """
        计算融合权重

        Args:
            query_type: 查询类型

        Returns:
            权重字典
        """
        # 默认权重：文本5%，向量95%
        weights = {
            "text_weight": 0.05,
            "vector_weight": 0.95
        }

        # 根据查询类型调整
        if query_type == "keyword_heavy":
            # 关键词密集的查询，增加文本权重
            weights["text_weight"] = 0.3
            weights["vector_weight"] = 0.7

        elif query_type == "semantic_heavy":
            # 语义查询，增加向量权重
            weights["text_weight"] = 0.02
            weights["vector_weight"] = 0.98

        return weights

    def estimate_quality_score(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> float:
        """
        估算检索质量分数

        Args:
            results: 检索结果
            query: 查询文本

        Returns:
            质量分数 (0-1)
        """
        if not results:
            return 0.0

        # 基于结果数量和相似度计算
        result_count = len(results)
        avg_similarity = np.mean([
            r.get('similarity', r.get('score', 0.5))
            for r in results
        ])

        # 数量分数（最多20个结果为满分）
        count_score = min(result_count / 20.0, 1.0)

        # 相似度分数
        similarity_score = avg_similarity

        # 综合
        quality_score = (count_score * 0.3 + similarity_score * 0.7)

        return quality_score

    def should_expand_query(
        self,
        quality_score: float,
        result_count: int
    ) -> bool:
        """
        判断是否应该扩展查询

        Args:
            quality_score: 质量分数
            result_count: 结果数量

        Returns:
            True表示需要扩展
        """
        # 质量低或结果少时扩展
        return quality_score < 0.5 or result_count < 3

    def optimize_batch_size(
        self,
        total_queries: int,
        available_workers: int = 4
    ) -> int:
        """
        优化批处理大小

        Args:
            total_queries: 总查询数
            available_workers: 可用工作线程数

        Returns:
            最优批次大小
        """
        # 每个worker处理一定数量的查询
        queries_per_worker = max(1, total_queries // available_workers)

        # 确保不超过合理上限
        batch_size = min(queries_per_worker, 10)

        return batch_size


# 创建全局服务实例
retrieval_optimizer = RetrievalOptimizer()


def get_retrieval_optimizer() -> RetrievalOptimizer:
    """获取检索优化器实例"""
    return retrieval_optimizer
