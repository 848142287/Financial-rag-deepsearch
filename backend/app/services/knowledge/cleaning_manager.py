"""
文档清洗管理器
整合所有清洗功能，提供一站式清洗服务

功能：
- 去除页眉页脚（document_cleaner）
- 格式规范化（content_cleaner）
- 去除噪音和无意义内容
- 质量评分和统计
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from .document_cleaner import DocumentCleaner, CleaningStats
from .content_cleaner import ContentCleaner, CleaningResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCleaningResult:
    """增强的清洗结果"""
    original_content: str
    cleaned_content: str
    document_stats: Dict[str, Any]
    content_stats: Dict[str, Any]
    quality_score: float
    improvements: List[str]
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class CleaningManager:
    """
    清洗管理器

    整合 document_cleaner 和 content_cleaner，提供完整的清洗流程
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 初始化清洗器
        self.document_cleaner = DocumentCleaner(
            self.config.get('document_cleaner', {})
        )

        self.content_cleaner = ContentCleaner(
            self.config.get('content_cleaner', {})
        )

        # 清洗流程配置
        self.cleaning_pipeline = self.config.get('cleaning_pipeline', [
            'remove_headers_footers',  # 第一步：去除页眉页脚
            'remove_noise',             # 第二步：去除噪音
            'format_normalization',     # 第三步：格式规范化
            'deduplication'             # 第四步：去重
        ])

        logger.info("文档清洗管理器初始化完成")

    async def clean_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnhancedCleaningResult:
        """
        完整的文档清洗流程

        参数:
            content: 原始文档内容
            metadata: 文档元数据（可选）

        返回:
            EnhancedCleaningResult: 清洗结果
        """
        import time
        start_time = time.time()

        original_content = content
        current_content = content
        improvements = []

        try:
            # 执行清洗流程
            for step in self.cleaning_pipeline:
                if step == 'remove_headers_footers':
                    # 去除页眉页脚和页码
                    current_content, doc_stats = self.document_cleaner.clean_document(
                        current_content,
                        metadata
                    )

                    if doc_stats.removed_headers > 0:
                        improvements.append(f"去除 {doc_stats.removed_headers} 个页眉")
                    if doc_stats.removed_footers > 0:
                        improvements.append(f"去除 {doc_stats.removed_footers} 个页脚")
                    if doc_stats.removed_page_numbers > 0:
                        improvements.append(f"去除 {doc_stats.removed_page_numbers} 个页码")
                    if doc_stats.removed_noise > 0:
                        improvements.append(f"去除 {doc_stats.removed_noise} 个噪音片段")

                elif step == 'format_normalization':
                    # 格式规范化
                    content_result = self.content_cleaner.clean_content(
                        current_content,
                        content_type='text'
                    )

                    current_content = content_result.cleaned_content

                    if content_result.fixes_applied:
                        improvements.append(f"应用 {len(content_result.fixes_applied)} 项格式修复")

                elif step == 'deduplication':
                    # 去重（已在 document_cleaner 中处理）
                    pass

            # 最终质量评分
            quality_score = self._calculate_quality_score(
                original_content,
                current_content,
                doc_stats if 'doc_stats' in locals() else None
            )

            # 生成统计信息
            document_stats = {
                'original_length': len(original_content),
                'cleaned_length': len(current_content),
                'reduction_ratio': (len(original_content) - len(current_content)) / len(original_content) if original_content else 0,
                **(doc_stats.__dict__ if 'doc_stats' in locals() else {})
            }

            content_stats = {
                'quality_score': quality_score,
                'improvements_count': len(improvements),
                'improvements': improvements
            }

            processing_time = time.time() - start_time

            result = EnhancedCleaningResult(
                original_content=original_content,
                cleaned_content=current_content,
                document_stats=document_stats,
                content_stats=content_stats,
                quality_score=quality_score,
                improvements=improvements,
                processing_time=processing_time
            )

            logger.info(f"文档清洗完成: 质量分数 {quality_score:.2f}, "
                       f"减少 {document_stats['reduction_ratio']*100:.1f}%, "
                       f"耗时 {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"文档清洗失败: {e}", exc_info=True)

            # 返回原始内容
            return EnhancedCleaningResult(
                original_content=original_content,
                cleaned_content=original_content,
                document_stats={'error': str(e)},
                content_stats={},
                quality_score=0.0,
                improvements=[],
                processing_time=time.time() - start_time
            )

    def _calculate_quality_score(
        self,
        original: str,
        cleaned: str,
        doc_stats: Optional[CleaningStats] = None
    ) -> float:
        """
        计算内容质量分数

        基于：
        - 内容减少比例（去除噪音越多越好）
        - 保留内容的连贯性
        - 段落结构完整性
        """
        score = 0.0

        # 基础分：50分
        score += 50.0

        # 去除噪音加分（最多20分）
        if doc_stats:
            noise_ratio = (
                doc_stats.removed_headers +
                doc_stats.removed_footers +
                doc_stats.removed_page_numbers +
                doc_stats.removed_noise
            ) / max(len(original.split('\n')), 1)

            score += min(noise_ratio * 100, 20.0)

        # 内容连贯性加分（最多20分）
        paragraphs = cleaned.split('\n\n')
        if len(paragraphs) > 0:
            avg_para_length = sum(len(p) for p in paragraphs) / len(paragraphs)

            # 理想段落长度：50-500字符
            if 50 <= avg_para_length <= 500:
                score += 20.0
            elif avg_para_length > 0:
                score += 10.0

        # 结构完整性加分（最多10分）
        lines = cleaned.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]

        if non_empty_lines:
            completeness = min(len(non_empty_lines) / 10, 1.0)
            score += completeness * 10.0

        return min(score, 100.0)


# 便捷函数
async def clean_document_enhanced(
    content: str,
    config: Optional[Dict[str, Any]] = None
) -> EnhancedCleaningResult:
    """
    增强的文档清洗便捷函数

    参数:
        content: 原始文档内容
        config: 配置字典

    返回:
        EnhancedCleaningResult: 清洗结果

    示例:
        result = await clean_document_enhanced(raw_text)
        print(f"质量分数: {result.quality_score}")
        print(f"改进: {result.improvements}")
    """
    manager = CleaningManager(config)
    return await manager.clean_document(content)


def quick_clean(content: str) -> str:
    """
    快速清洗（同步，使用默认配置）

    参数:
        content: 原始文档内容

    返回:
        清洗后的内容

    示例:
        cleaned = quick_clean(raw_text)
    """
    import asyncio
    result = asyncio.run(clean_document_enhanced(content))
    return result.cleaned_content
