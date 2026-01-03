"""
统一批处理配置管理
消除系统中30+处分散的batch_size配置,提供统一的配置中心

优化目标:
1. 配置统一管理
2. 支持环境变量覆盖
3. 支持动态调整
4. 提供配置验证
"""

import os
from typing import Dict, Any, Optional
from enum import Enum

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class BatchScenario(str, Enum):
    """批处理场景枚举"""
    EMBEDDING_GENERATION = "embedding_generation"       # Embedding向量生成
    EMBEDDING_CACHE = "embedding_cache"                 # Embedding缓存
    VECTOR_STORAGE_MYSQL = "vector_storage_mysql"       # 向量存储到MySQL
    VECTOR_STORAGE_MILVUS = "vector_storage_milvus"     # 向量存储到Milvus
    DOCUMENT_PARSING = "document_parsing"               # 文档解析
    MARKDOWN_SUPPLEMENT = "markdown_supplement"         # Markdown补充解析
    KNOWLEDGE_SYNC = "knowledge_sync"                   # 知识图谱同步
    EVALUATION = "evaluation"                           # RAGAS评估
    PDF_PROCESSING = "pdf_processing"                   # PDF批处理
    TABLE_RECOGNITION = "table_recognition"             # 表格识别
    FORMULA_RECOGNITION = "formula_recognition"         # 公式识别

@dataclass
class UnifiedBatchConfig:
    """统一批处理配置"""

    # Embedding相关
    embedding_generation: int = 32           # Embedding向量批量生成
    embedding_cache_size: int = 1000         # Embedding缓存大小
    embedding_cache_ttl: int = 3600          # 缓存TTL(秒)

    # 向量存储相关
    mysql_insert_batch: int = 100            # MySQL批量插入
    milvus_insert_batch: int = 100           # Milvus批量插入

    # 文档处理相关
    document_parse_batch: int = 10           # 文档批量解析
    markdown_supplement_batch: int = 5       # Markdown补充批量处理
    pdf_processing_batch: int = 10           # PDF批处理

    # 知识图谱相关
    knowledge_sync_batch: int = 100          # 知识图谱同步
    entity_extraction_batch: int = 50        # 实体抽取
    relation_extraction_batch: int = 50      # 关系抽取

    # 评估相关
    evaluation_batch: int = 500              # RAGAS评估批处理
    ragas_worker_batch: int = 500            # RAGAS工作线程批处理

    # 模型推理相关
    table_recognition_batch: int = 16        # 表格识别
    formula_recognition_batch: int = 64      # 公式识别
    vlm_document_batch: int = 3              # VLM文档分析

    # 自适应配置
    enable_adaptive: bool = True             # 启用自适应调整
    min_batch_size: int = 1                  # 最小batch size
    max_batch_size: int = 128                # 最大batch size
    memory_usage_threshold: float = 0.3      # 内存使用阈值(30%)

    # 性能配置
    max_concurrent_batches: int = 5          # 最大并发批次数
    batch_timeout: float = 300.0             # 批处理超时(秒)

    def __post_init__(self):
        """配置初始化后验证"""
        self._validate_config()
        self._log_config()

    def _validate_config(self):
        """验证配置合法性"""
        validations = [
            (self.embedding_generation > 0, "embedding_generation must be > 0"),
            (self.mysql_insert_batch > 0, "mysql_insert_batch must be > 0"),
            (self.min_batch_size >= 1, "min_batch_size must be >= 1"),
            (self.max_batch_size >= self.min_batch_size, "max_batch_size must be >= min_batch_size"),
            (0 < self.memory_usage_threshold < 1, "memory_usage_threshold must be in (0, 1)"),
        ]

        for condition, error_msg in validations:
            if not condition:
                raise ValueError(f"Invalid config: {error_msg}")

    def _log_config(self):
        """记录配置信息"""
        logger.info(f"📊 Unified Batch Config initialized:")
        logger.info(f"  - Embedding Generation: {self.embedding_generation}")
        logger.info(f"  - MySQL Insert Batch: {self.mysql_insert_batch}")
        logger.info(f"  - Milvus Insert Batch: {self.milvus_insert_batch}")
        logger.info(f"  - Document Parse Batch: {self.document_parse_batch}")
        logger.info(f"  - Knowledge Sync Batch: {self.knowledge_sync_batch}")
        logger.info(f"  - Adaptive Mode: {self.enable_adaptive}")

    @classmethod
    def from_env(cls) -> 'UnifiedBatchConfig':
        """
        从环境变量加载配置

        环境变量列表:
        - EMBEDDING_BATCH_SIZE: Embedding批量大小
        - MYSQL_BATCH_SIZE: MySQL批量大小
        - MILVUS_BATCH_SIZE: Milvus批量大小
        - DOCUMENT_PARSE_BATCH_SIZE: 文档解析批量大小
        - ENABLE_ADAPTIVE_BATCH: 启用自适应批处理
        - MIN_BATCH_SIZE: 最小批量大小
        - MAX_BATCH_SIZE: 最大批量大小
        - MEMORY_USAGE_THRESHOLD: 内存使用阈值
        """
        return cls(
            # Embedding相关
            embedding_generation=int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
            embedding_cache_size=int(os.getenv('EMBEDDING_CACHE_SIZE', '1000')),
            embedding_cache_ttl=int(os.getenv('EMBEDDING_CACHE_TTL', '3600')),

            # 向量存储相关
            mysql_insert_batch=int(os.getenv('MYSQL_BATCH_SIZE', '100')),
            milvus_insert_batch=int(os.getenv('MILVUS_BATCH_SIZE', '100')),

            # 文档处理相关
            document_parse_batch=int(os.getenv('DOCUMENT_PARSE_BATCH_SIZE', '10')),
            markdown_supplement_batch=int(os.getenv('MARKDOWN_SUPPLEMENT_BATCH_SIZE', '5')),
            pdf_processing_batch=int(os.getenv('PDF_PROCESSING_BATCH_SIZE', '10')),

            # 知识图谱相关
            knowledge_sync_batch=int(os.getenv('KNOWLEDGE_SYNC_BATCH_SIZE', '100')),
            entity_extraction_batch=int(os.getenv('ENTITY_EXTRACTION_BATCH_SIZE', '50')),
            relation_extraction_batch=int(os.getenv('RELATION_EXTRACTION_BATCH_SIZE', '50')),

            # 评估相关
            evaluation_batch=int(os.getenv('EVALUATION_BATCH_SIZE', '500')),
            ragas_worker_batch=int(os.getenv('RAGAS_WORKER_BATCH_SIZE', '500')),

            # 模型推理相关
            table_recognition_batch=int(os.getenv('TABLE_RECOGNITION_BATCH_SIZE', '16')),
            formula_recognition_batch=int(os.getenv('FORMULA_RECOGNITION_BATCH_SIZE', '64')),
            vlm_document_batch=int(os.getenv('VLM_DOCUMENT_BATCH_SIZE', '3')),

            # 自适应配置
            enable_adaptive=os.getenv('ENABLE_ADAPTIVE_BATCH', 'true').lower() == 'true',
            min_batch_size=int(os.getenv('MIN_BATCH_SIZE', '1')),
            max_batch_size=int(os.getenv('MAX_BATCH_SIZE', '128')),
            memory_usage_threshold=float(os.getenv('MEMORY_USAGE_THRESHOLD', '0.3')),

            # 性能配置
            max_concurrent_batches=int(os.getenv('MAX_CONCURRENT_BATCHES', '5')),
            batch_timeout=float(os.getenv('BATCH_TIMEOUT', '300.0')),
        )

    def get_batch_size(self, scenario: BatchScenario) -> int:
        """
        获取指定场景的batch size

        Args:
            scenario: 批处理场景

        Returns:
            batch size值
        """
        scenario_mapping = {
            BatchScenario.EMBEDDING_GENERATION: self.embedding_generation,
            BatchScenario.EMBEDDING_CACHE: self.embedding_cache_size,
            BatchScenario.VECTOR_STORAGE_MYSQL: self.mysql_insert_batch,
            BatchScenario.VECTOR_STORAGE_MILVUS: self.milvus_insert_batch,
            BatchScenario.DOCUMENT_PARSING: self.document_parse_batch,
            BatchScenario.MARKDOWN_SUPPLEMENT: self.markdown_supplement_batch,
            BatchScenario.KNOWLEDGE_SYNC: self.knowledge_sync_batch,
            BatchScenario.EVALUATION: self.evaluation_batch,
            BatchScenario.PDF_PROCESSING: self.pdf_processing_batch,
            BatchScenario.TABLE_RECOGNITION: self.table_recognition_batch,
            BatchScenario.FORMULA_RECOGNITION: self.formula_recognition_batch,
        }

        batch_size = scenario_mapping.get(scenario, 10)

        # 应用限制
        batch_size = max(self.min_batch_size, min(batch_size, self.max_batch_size))

        return batch_size

    def adjust_for_memory(self, available_memory_gb: float) -> 'UnifiedBatchConfig':
        """
        根据可用内存调整配置

        Args:
            available_memory_gb: 可用内存(GB)

        Returns:
            调整后的配置
        """
        # 内存不足时减小batch size
        if available_memory_gb < 8:
            logger.warning(f"⚠️  Low memory detected ({available_memory_gb:.1f}GB), reducing batch sizes")
            factor = 0.5
        elif available_memory_gb > 32:
            logger.info(f"✅ High memory available ({available_memory_gb:.1f}GB), increasing batch sizes")
            factor = 2.0
        else:
            factor = 1.0

        # 调整各项配置
        return UnifiedBatchConfig(
            embedding_generation=max(1, int(self.embedding_generation * factor)),
            embedding_cache_size=self.embedding_cache_size,
            mysql_insert_batch=max(1, int(self.mysql_insert_batch * factor)),
            milvus_insert_batch=max(1, int(self.milvus_insert_batch * factor)),
            document_parse_batch=max(1, int(self.document_parse_batch * factor)),
            markdown_supplement_batch=max(1, int(self.markdown_supplement_batch * factor)),
            knowledge_sync_batch=max(1, int(self.knowledge_sync_batch * factor)),
            evaluation_batch=self.evaluation_batch,  # 评估不受影响
            table_recognition_batch=max(1, int(self.table_recognition_batch * factor)),
            formula_recognition_batch=max(1, int(self.formula_recognition_batch * factor)),
            vlm_document_batch=self.vlm_document_batch,
            enable_adaptive=self.enable_adaptive,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            memory_usage_threshold=self.memory_usage_threshold,
            max_concurrent_batches=self.max_concurrent_batches,
            batch_timeout=self.batch_timeout,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'embedding_generation': self.embedding_generation,
            'embedding_cache_size': self.embedding_cache_size,
            'mysql_insert_batch': self.mysql_insert_batch,
            'milvus_insert_batch': self.milvus_insert_batch,
            'document_parse_batch': self.document_parse_batch,
            'markdown_supplement_batch': self.markdown_supplement_batch,
            'knowledge_sync_batch': self.knowledge_sync_batch,
            'evaluation_batch': self.evaluation_batch,
            'enable_adaptive': self.enable_adaptive,
            'min_batch_size': self.min_batch_size,
            'max_batch_size': self.max_batch_size,
            'memory_usage_threshold': self.memory_usage_threshold,
            'max_concurrent_batches': self.max_concurrent_batches,
            'batch_timeout': self.batch_timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedBatchConfig':
        """从字典创建配置"""
        return cls(**data)

# 全局配置实例
_global_config: Optional[UnifiedBatchConfig] = None

def get_batch_config(use_env: bool = True) -> UnifiedBatchConfig:
    """
    获取全局批处理配置

    Args:
        use_env: 是否从环境变量加载

    Returns:
        统一批处理配置实例
    """
    global _global_config

    if _global_config is None:
        if use_env:
            _global_config = UnifiedBatchConfig.from_env()
            logger.info("📊 Loaded batch config from environment variables")
        else:
            _global_config = UnifiedBatchConfig()
            logger.info("📊 Using default batch config")

    return _global_config

def reset_batch_config():
    """重置全局配置(用于测试)"""
    global _global_config
    _global_config = None
    logger.info("📊 Batch config reset")

# 便利函数
def get_embedding_batch_size() -> int:
    """获取Embedding批量大小"""
    return get_batch_config().get_batch_size(BatchScenario.EMBEDDING_GENERATION)

def get_mysql_batch_size() -> int:
    """获取MySQL批量大小"""
    return get_batch_config().get_batch_size(BatchScenario.VECTOR_STORAGE_MYSQL)

def get_milvus_batch_size() -> int:
    """获取Milvus批量大小"""
    return get_batch_config().get_batch_size(BatchScenario.VECTOR_STORAGE_MILVUS)

def get_document_parse_batch_size() -> int:
    """获取文档解析批量大小"""
    return get_batch_config().get_batch_size(BatchScenario.DOCUMENT_PARSING)

def get_markdown_supplement_batch_size() -> int:
    """获取Markdown补充批量大小"""
    return get_batch_config().get_batch_size(BatchScenario.MARKDOWN_SUPPLEMENT)

# 使用示例
if __name__ == "__main__":
    # 示例1: 使用默认配置
    config1 = UnifiedBatchConfig()
    print(f"Embedding Batch Size: {config1.embedding_generation}")

    # 示例2: 从环境变量加载
    config2 = UnifiedBatchConfig.from_env()
    print(f"MySQL Batch Size: {config2.mysql_insert_batch}")

    # 示例3: 获取全局配置
    config3 = get_batch_config()
    print(f"Document Parse Batch: {config3.get_batch_size(BatchScenario.DOCUMENT_PARSING)}")

    # 示例4: 根据内存调整
    config_low_memory = config3.adjust_for_memory(4.0)  # 4GB内存
    print(f"Low Memory Embedding Batch: {config_low_memory.embedding_generation}")

    config_high_memory = config3.adjust_for_memory(64.0)  # 64GB内存
    print(f"High Memory Embedding Batch: {config_high_memory.embedding_generation}")
