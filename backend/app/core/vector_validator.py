"""
向量质量验证工具
检测和过滤无效向量（零向量、NaN、Inf等）
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    reason: Optional[str] = None
    index: Optional[int] = None
    embedding: Optional[np.ndarray] = None


class VectorQualityValidator:
    """向量质量验证器"""

    # 质量阈值
    MIN_NORM = 0.01  # 最小向量范数（低于此值认为是零向量）
    MAX_NORM = 1000  # 最大向量范数（异常检测）

    def __init__(
        self,
        expected_dim: int = 1024,
        check_zero_vector: bool = True,
        check_nan: bool = True,
        check_inf: bool = True,
        check_dimension: bool = True,
        min_norm: Optional[float] = None,
        max_norm: Optional[float] = None
    ):
        """
        初始化验证器

        Args:
            expected_dim: 期望的向量维度
            check_zero_vector: 是否检查零向量
            check_nan: 是否检查 NaN
            check_inf: 是否检查 Inf
            check_dimension: 是否检查维度
            min_norm: 最小范数阈值
            max_norm: 最大范数阈值
        """
        self.expected_dim = expected_dim
        self.check_zero_vector = check_zero_vector
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.check_dimension = check_dimension
        self.min_norm = min_norm if min_norm is not None else self.MIN_NORM
        self.max_norm = max_norm if max_norm is not None else self.MAX_NORM

    def validate_embedding(
        self,
        embedding: np.ndarray,
        index: Optional[int] = None
    ) -> ValidationResult:
        """
        验证单个向量

        Args:
            embedding: 嵌入向量
            index: 向量索引（用于日志）

        Returns:
            ValidationResult
        """
        # 检查维度
        if self.check_dimension:
            if len(embedding) != self.expected_dim:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Dimension mismatch: expected {self.expected_dim}, got {len(embedding)}",
                    index=index,
                    embedding=embedding
                )

        # 检查 NaN
        if self.check_nan and np.any(np.isnan(embedding)):
            nan_count = np.sum(np.isnan(embedding))
            return ValidationResult(
                is_valid=False,
                reason=f"Contains {nan_count} NaN values",
                index=index,
                embedding=embedding
            )

        # 检查 Inf
        if self.check_inf and np.any(np.isinf(embedding)):
            inf_count = np.sum(np.isinf(embedding))
            return ValidationResult(
                is_valid=False,
                reason=f"Contains {inf_count} Inf values",
                index=index,
                embedding=embedding
            )

        # 检查零向量
        if self.check_zero_vector:
            norm = np.linalg.norm(embedding)
            if norm < self.min_norm:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Zero vector (norm={norm:.6f} < {self.min_norm})",
                    index=index,
                    embedding=embedding
                )

            # 检查异常大的范数
            if norm > self.max_norm:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Abnormally large norm (norm={norm:.2f} > {self.max_norm})",
                    index=index,
                    embedding=embedding
                )

        return ValidationResult(
            is_valid=True,
            reason="OK",
            index=index,
            embedding=embedding
        )

    def validate_embeddings(
        self,
        embeddings: List[np.ndarray]
    ) -> Tuple[List[ValidationResult], List[int]]:
        """
        验证多个向量

        Args:
            embeddings: 嵌入向量列表

        Returns:
            (验证结果列表, 无效索引列表)
        """
        results = []
        invalid_indices = []

        for i, embedding in enumerate(embeddings):
            result = self.validate_embedding(embedding, i)
            results.append(result)

            if not result.is_valid:
                invalid_indices.append(i)
                logger.warning(
                    f"Invalid embedding at index {i}: {result.reason}"
                )

        return results, invalid_indices

    def filter_valid_embeddings(
        self,
        embeddings: List[np.ndarray],
        texts: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[ValidationResult], List[int]]:
        """
        过滤出有效向量

        Args:
            embeddings: 嵌入向量列表
            texts: 对应的文本列表（可选，用于日志）

        Returns:
            (有效向量列表, 所有验证结果, 无效索引列表)
        """
        results, invalid_indices = self.validate_embeddings(embeddings)

        valid_embeddings = []
        for i, (result, emb) in enumerate(zip(results, embeddings)):
            if result.is_valid:
                valid_embeddings.append(emb)
            else:
                text_preview = texts[i][:50] if texts and i < len(texts) else "N/A"
                logger.warning(
                    f"Filtered out invalid embedding at index {i} "
                    f"(text: '{text_preview}...'): {result.reason}"
                )

        return valid_embeddings, results, invalid_indices

    def get_validation_stats(
        self,
        results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        获取验证统计信息

        Args:
            results: 验证结果列表

        Returns:
            统计信息字典
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = total - valid

        # 统计失败原因
        failure_reasons = {}
        for r in results:
            if not r.is_valid and r.reason:
                failure_reasons[r.reason] = failure_reasons.get(r.reason, 0) + 1

        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "valid_rate": valid / total if total > 0 else 0,
            "failure_reasons": failure_reasons
        }


def validate_embedding(
    embedding: np.ndarray,
    expected_dim: int = 1024
) -> bool:
    """
    快速验证单个向量（便捷函数）

    Args:
        embedding: 嵌入向量
        expected_dim: 期望维度

    Returns:
        bool: 是否有效
    """
    validator = VectorQualityValidator(expected_dim=expected_dim)
    result = validator.validate_embedding(embedding)
    return result.is_valid


def validate_embeddings(
    embeddings: List[np.ndarray],
    expected_dim: int = 1024,
    return_valid_only: bool = False
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """
    批量验证向量（便捷函数）

    Args:
        embeddings: 嵌入向量列表
        expected_dim: 期望维度
        return_valid_only: 是否只返回有效向量

    Returns:
        (有效向量列表, 无效索引列表, 统计信息)
    """
    validator = VectorQualityValidator(expected_dim=expected_dim)

    if return_valid_only:
        valid_embeddings, results, invalid_indices = validator.filter_valid_embeddings(embeddings)
        stats = validator.get_validation_stats(results)
        return valid_embeddings, invalid_indices, stats
    else:
        results, invalid_indices = validator.validate_embeddings(embeddings)
        stats = validator.get_validation_stats(results)
        return embeddings, invalid_indices, stats
