"""
统一的检索器管理器
解决检索器初始化问题，提供健壮的检索服务
"""

from app.core.structured_logging import get_structured_logger
from dataclasses import dataclass
from enum import Enum

logger = get_structured_logger(__name__)

class RetrieverType(Enum):
    """检索器类型"""
    VECTOR = "vector_search"
    GRAPH = "graph_search"
    KEYWORD = "keyword_search"
    TEMPORAL = "temporal_search"
    HYBRID = "hybrid_search"

@dataclass
class RetrieverConfig:
    """检索器配置"""
    enabled: bool = True
    essential: bool = False  # 是否为必需的检索器
    fallback_enabled: bool = True  # 是否启用降级策略

class RetrieverManager:
    """检索器管理器 - 健壮的初始化和管理"""

    def __init__(self):
        self.retrievers: Dict[RetrieverType, Any] = {}
        self.configs: Dict[RetrieverType, RetrieverConfig] = {}
        self.initialization_status: Dict[RetrieverType, str] = {}

        # 定义检索器配置
        self._setup_retriever_configs()

        # 初始化检索器
        self._initialize_all_retrievers()

    def _setup_retriever_configs(self):
        """设置检索器配置"""
        self.configs = {
            RetrieverType.VECTOR: RetrieverConfig(
                enabled=True,
                essential=True,  # 向量检索是必需的
                fallback_enabled=False
            ),
            RetrieverType.GRAPH: RetrieverConfig(
                enabled=True,
                essential=False,  # 图谱检索不是必需的
                fallback_enabled=True
            ),
            RetrieverType.KEYWORD: RetrieverConfig(
                enabled=True,
                essential=False,
                fallback_enabled=True
            ),
            RetrieverType.TEMPORAL: RetrieverConfig(
                enabled=False,  # 暂时禁用
                essential=False,
                fallback_enabled=True
            ),
            RetrieverType.HYBRID: RetrieverConfig(
                enabled=True,
                essential=False,
                fallback_enabled=True
            )
        }

    def _initialize_all_retrievers(self):
        """初始化所有检索器"""
        logger.info("开始初始化检索器...")

        for retriever_type, config in self.configs.items():
            if not config.enabled:
                logger.info(f"检索器 {retriever_type.value} 已禁用，跳过初始化")
                self.initialization_status[retriever_type] = "disabled"
                continue

            try:
                retriever = self._initialize_retriever(retriever_type)
                if retriever:
                    self.retrievers[retriever_type] = retriever
                    self.initialization_status[retriever_type] = "success"
                    logger.info(f"✓ 检索器 {retriever_type.value} 初始化成功")
                else:
                    self.initialization_status[retriever_type] = "failed"
                    if config.essential:
                        raise RuntimeError(f"必需的检索器 {retriever_type.value} 初始化失败")
                    else:
                        logger.warning(f"⚠ 检索器 {retriever_type.value} 初始化失败，将使用降级策略")

            except Exception as e:
                self.initialization_status[retriever_type] = "error"
                logger.error(f"✗ 检索器 {retriever_type.value} 初始化异常: {e}")
                if self.configs[retriever_type].essential:
                    raise RuntimeError(f"必需的检索器 {retriever_type.value} 初始化失败: {e}")

        self._log_initialization_summary()

    def _initialize_retriever(self, retriever_type: RetrieverType) -> Optional[Any]:
        """初始化单个检索器"""
        try:
            if retriever_type == RetrieverType.VECTOR:
                return self._init_vector_retriever()
            elif retriever_type == RetrieverType.GRAPH:
                return self._init_graph_retriever()
            elif retriever_type == RetrieverType.KEYWORD:
                return self._init_keyword_retriever()
            elif retriever_type == RetrieverType.HYBRID:
                return self._init_hybrid_retriever()
            else:
                logger.warning(f"未知的检索器类型: {retriever_type}")
                return None

        except ImportError as e:
            logger.error(f"导入模块失败: {e}")
            return None
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return None

    def _init_vector_retriever(self):
        """初始化向量检索器"""
        try:
            # 尝试导入 MilvusService
            from app.services.milvus_service import MilvusService
            retriever = MilvusService()

            # 测试连接
            import asyncio
            test_result = asyncio.run(self._test_retriever(retriever, "vector"))

            if test_result:
                logger.info("向量检索器连接测试成功")
                return retriever
            else:
                logger.warning("向量检索器连接测试失败")
                return None

        except Exception as e:
            logger.error(f"向量检索器初始化失败: {e}")
            return None

    def _init_graph_retriever(self):
        """初始化图谱检索器"""
        try:
            # 尝试导入 Neo4jService
            from app.services.neo4j_service import Neo4jService
            retriever = Neo4jService()

            # 测试连接
            import asyncio
            test_result = asyncio.run(self._test_retriever(retriever, "graph"))

            if test_result:
                logger.info("图谱检索器连接测试成功")
                return retriever
            else:
                logger.warning("图谱检索器连接测试失败")
                return None

        except Exception as e:
            logger.error(f"图谱检索器初始化失败: {e}")
            return None

    def _init_keyword_retriever(self):
        """初始化关键词检索器"""
        try:
            # 尝试导入关键词检索服务
            from app.services.knowledge.retrieval_engine import MultiStrategyRetrievalEngine
            # 使用检索引擎中的关键词检索方法
            retriever = MultiStrategyRetrievalEngine()
            logger.info("关键词检索器初始化成功")
            return retriever

        except Exception as e:
            logger.error(f"关键词检索器初始化失败: {e}")
            return None

    def _init_hybrid_retriever(self):
        """初始化混合检索器"""
        try:
            from app.services.knowledge.retrieval_engine import MultiStrategyRetrievalEngine
            retriever = MultiStrategyRetrievalEngine()
            logger.info("混合检索器初始化成功")
            return retriever

        except Exception as e:
            logger.error(f"混合检索器初始化失败: {e}")
            return None

    async def _test_retriever(self, retriever, retriever_type: str) -> bool:
        """测试检索器连接"""
        try:
            if retriever_type == "vector":
                # 测试向量检索器
                test_vector = [0.0] * 1536  # 假设使用1536维向量
                result = await retriever.search_vectors(
                    collection_name="documents",
                    query_vectors=[test_vector],
                    top_k=1
                )
                return True

            elif retriever_type == "graph":
                # 测试图谱检索器
                result = await retriever.search_entity(
                    entity_name="test",
                    entity_type="COMPANY",
                    limit=1
                )
                return True

        except Exception as e:
            logger.warning(f"检索器测试失败: {e}")
            return False

        return False

    def _log_initialization_summary(self):
        """记录初始化摘要"""
        success_count = sum(1 for status in self.initialization_status.values() if status == "success")
        failed_count = sum(1 for status in self.initialization_status.values() if status in ["failed", "error"])
        disabled_count = sum(1 for status in self.initialization_status.values() if status == "disabled")

        logger.info("=" * 60)
        logger.info("检索器初始化摘要:")
        logger.info(f"  成功: {success_count}")
        logger.info(f"  失败: {failed_count}")
        logger.info(f"  禁用: {disabled_count}")
        logger.info("=" * 60)

        for retriever_type, status in self.initialization_status.items():
            status_symbol = {
                "success": "✓",
                "failed": "✗",
                "error": "✗",
                "disabled": "⊘"
            }.get(status, "?")

            logger.info(f"  {status_symbol} {retriever_type.value}: {status}")

    def get_retriever(self, retriever_type: RetrieverType):
        """获取检索器"""
        retriever = self.retrievers.get(retriever_type)

        if not retriever:
            config = self.configs.get(retriever_type)
            if config and config.fallback_enabled:
                logger.warning(f"检索器 {retriever_type.value} 不可用，使用降级策略")
                return self._get_fallback_retriever(retriever_type)
            else:
                raise RuntimeError(f"检索器 {retriever_type.value} 不可用且无降级策略")

        return retriever

    def _get_fallback_retriever(self, retriever_type: RetrieverType):
        """获取降级检索器"""
        # 简单的降级实现：返回模拟检索器
        logger.info(f"使用模拟检索器替代 {retriever_type.value}")

        class MockRetriever:
            async def search(self, query, top_k=10, **kwargs):
                return []

        return MockRetriever()

    def is_available(self, retriever_type: RetrieverType) -> bool:
        """检查检索器是否可用"""
        return retriever_type in self.retrievers

    def get_status(self) -> Dict[str, Any]:
        """获取所有检索器状态"""
        return {
            "retrievers": {
                retriever_type.value: {
                    "status": self.initialization_status.get(retriever_type, "unknown"),
                    "available": self.is_available(retriever_type),
                    "config": {
                        "enabled": config.enabled,
                        "essential": config.essential,
                        "fallback_enabled": config.fallback_enabled
                    }
                }
                for retriever_type, config in self.configs.items()
            },
            "summary": {
                "total": len(self.configs),
                "enabled": sum(1 for c in self.configs.values() if c.enabled),
                "available": len(self.retrievers),
                "essential_available": all(
                    self.is_available(rt) for rt, cfg in self.configs.items()
                    if cfg.essential
                )
            }
        }

# 全局检索器管理器实例
_retriever_manager_instance = None

def get_retriever_manager() -> RetrieverManager:
    """获取检索器管理器单例"""
    global _retriever_manager_instance
    if _retriever_manager_instance is None:
        _retriever_manager_instance = RetrieverManager()
    return _retriever_manager_instance
