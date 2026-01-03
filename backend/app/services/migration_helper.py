"""
渐进式迁移工具
安全地将旧代码迁移到新服务，确保功能不受影响
"""

import os
import sys
import hashlib
import random
from typing import Any
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class MigrationHelper:
    """迁移辅助工具"""

    def __init__(self):
        self.enable_unified = os.getenv("USE_UNIFIED_DOCUMENT_SERVICE", "false").lower() == "true"
        self.percentage = int(os.getenv("UNIFIED_SERVICE_PERCENTAGE", "0"))

    def should_use_unified_service(self, document_id: Any = None) -> bool:
        """
        判断是否使用新服务（基于百分比配置）

        Args:
            document_id: 文档ID，用于一致性哈希

        Returns:
            bool: True表示使用新服务，False表示使用旧服务
        """
        if not self.enable_unified:
            return False

        if self.percentage >= 100:
            return True

        if self.percentage <= 0:
            return False

        # 使用document_id进行一致性哈希，确保同一个文档始终使用相同的服务
        if document_id is not None:
            hash_value = int(hashlib.md5(str(document_id).encode()).hexdigest(), 16)
            return (hash_value % 100) < self.percentage
        else:
            # 随机选择
            return random.randint(0, 99) < self.percentage

    async def process_document_with_fallback(
        self,
        document_id: int,
        file_path: str,
        config: dict = None,
        db = None
    ) -> dict:
        """
        带fallback的文档处理

        自动选择新旧服务，并记录性能数据
        """
        import time
        from app.core.structured_logging import get_structured_logger

        logger = get_structured_logger(__name__)
        config = config or {}

        # 判断使用哪个服务
        use_unified = self.should_use_unified_service(document_id)

        start_time = time.time()

        try:
            if use_unified:
                logger.info(f"[文档 {document_id}] 使用统一服务处理")

                # 使用新服务
                from app.services.unified_document_service import unified_document_service

                result = await unified_document_service.process_document(
                    document_id=document_id,
                    file_path=file_path,
                    config=config,
                    db=db
                )

                # 标记来源
                result['_service_type'] = 'unified'
                result['_migration_info'] = {
                    'service': 'unified',
                    'percentage': self.percentage
                }

                # 记录成功
                logger.info(f"[文档 {document_id}] 统一服务处理成功: {result.get('chunks_count', 0)} chunks")

                return result

            else:
                logger.info(f"[文档 {document_id}] 使用传统服务处理")

                # 使用旧服务（保持不变）
                from app.services.knowledge.pipeline import knowledge_base_pipeline

                result = await knowledge_base_pipeline.process_document(
                    file_path=file_path,
                    document_id=str(document_id),
                    config=config
                )

                # 标记来源
                result['_service_type'] = 'legacy'
                result['_migration_info'] = {
                    'service': 'legacy',
                    'percentage': self.percentage
                }

                logger.info(f"[文档 {document_id}] 传统服务处理成功")

                return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[文档 {document_id}] 处理失败 ({use_unified and 'unified' or 'legacy'}): {e}")

            # 如果是新服务失败，自动fallback到旧服务
            if use_unified:
                logger.warning(f"[文档 {document_id}] 新服务失败，fallback到旧服务")

                try:
                    from app.services.knowledge.pipeline import knowledge_base_pipeline

                    result = await knowledge_base_pipeline.process_document(
                        file_path=file_path,
                        document_id=str(document_id),
                        config=config
                    )

                    result['_service_type'] = 'legacy'
                    result['_fallback'] = True
                    result['_fallback_error'] = str(e)

                    return result

                except Exception as fallback_error:
                    logger.error(f"[文档 {document_id}] Fallback也失败: {fallback_error}")
                    raise
            else:
                raise

    def get_migration_status(self) -> dict:
        """获取迁移状态"""
        return {
            'enabled': self.enable_unified,
            'percentage': self.percentage,
            'phase': self._get_migration_phase()
        }

    def _get_migration_phase(self) -> str:
        """获取当前迁移阶段"""
        if self.percentage == 0:
            return "disabled"
        elif self.percentage <= 10:
            return "pilot"  # 试运行
        elif self.percentage <= 50:
            return "gradual"  # 逐步推广
        elif self.percentage < 100:
            return "majority"  # 主要流量
        else:
            return "complete"  # 完全迁移


# 全局实例
migration_helper = MigrationHelper()
