"""
统一配置中心
集中管理所有配置，提供统一的配置访问接口

特性：
- 配置分层管理
- 配置验证
- 环境变量支持
- 热重载支持
- 配置导出/导入
"""

import os
import json
from pathlib import Path
from enum import Enum

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

T = TypeVar('T', bound=BaseModel)

class ConfigLevel(Enum):
    """配置层级"""
    SYSTEM = "system"       # 系统级配置
    SERVICE = "service"     # 服务级配置
    USER = "user"           # 用户级配置

class ConfigSource(Enum):
    """配置来源"""
    FILE = "file"           # 配置文件
    ENV = "env"             # 环境变量
    DATABASE = "database"   # 数据库
    DEFAULT = "default"     # 默认值

# ============================================================================
# 嵌入服务配置
# ============================================================================

@dataclass
class EmbeddingServiceConfig:
    """嵌入服务配置"""
    # 模型配置
    provider: str = "bge_local"  # bge_local, qwen_api
    model_path: str = "backend/models/bge-large-zh-v1.5"
    model_name: str = "bge-large-zh-v1.5"
    device: str = "cpu"

    # 推理配置
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    timeout: int = 60

    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600

    # 向量维度
    embedding_dimension: int = 1024

# ============================================================================
# 向量存储配置
# ============================================================================

@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    # Milvus配置
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "financial_documents"

    # 索引配置
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128
    nprobe: int = 10

    # 连接配置
    timeout: int = 30
    pool_size: int = 10
    max_retries: int = 3
    retry_delay: float = 0.5

    # 健康检查
    health_check_interval: int = 60

# ============================================================================
# Redis配置
# ============================================================================

@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    # 连接配置
    timeout: int = 5
    max_connections: int = 50
    socket_timeout: int = 5

    # 缓存配置
    l1_max_size: int = 1000
    l1_ttl: int = 300
    l2_max_size: int = 10000
    l2_ttl: int = 3600

# ============================================================================
# 文档解析配置
# ============================================================================

@dataclass
class DocumentParserConfig:
    """文档解析配置"""
    # 文件大小限制
    max_file_size: int = 100 * 1024 * 1024  # 100MB

    # 支持的文件类型
    supported_types: List[str] = field(default_factory=lambda: [
        "pdf", "xlsx", "xls", "docx", "pptx", "ppt", "md", "txt"
    ])

    # 解析选项
    extract_images: bool = True
    extract_tables: bool = True
    extract_metadata: bool = True

    # 输出格式
    output_markdown: bool = True
    output_json: bool = True

# ============================================================================
# Chunking配置
# ============================================================================

@dataclass
class ChunkingConfig:
    """文档分块配置"""
    # 基础配置
    chunk_size: int = 512
    chunk_overlap: int = 50

    # 自适应配置
    enable_adaptive: bool = True
    document_type_configs: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 特殊处理
    preserve_tables: bool = True
    preserve_formulas: bool = True
    section_aware: bool = True

# ============================================================================
# 统一配置
# ============================================================================

class UnifiedConfig:
    """
    统一配置中心

    提供统一的配置访问接口
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置中心

        Args:
            config_file: 可选的配置文件路径
        """
        self.config_file = config_file
        self._configs: Dict[str, Any] = {}
        self._load_configs()

    def _load_configs(self):
        """加载配置"""
        # 1. 加载默认配置
        self._load_default_configs()

        # 2. 从环境变量加载
        self._load_from_env()

        # 3. 从配置文件加载
        if self.config_file:
            self._load_from_file()

    def _load_default_configs(self):
        """加载默认配置"""
        self._configs = {
            "embedding": EmbeddingServiceConfig(),
            "vector_store": VectorStoreConfig(),
            "redis": RedisConfig(),
            "document_parser": DocumentParserConfig(),
            "chunking": ChunkingConfig()
        }

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 嵌入服务
        if os.getenv("EMBEDDING_PROVIDER"):
            self._configs["embedding"].provider = os.getenv("EMBEDDING_PROVIDER")
        if os.getenv("EMBEDDING_MODEL_PATH"):
            self._configs["embedding"].model_path = os.getenv("EMBEDDING_MODEL_PATH")
        if os.getenv("EMBEDDING_DEVICE"):
            self._configs["embedding"].device = os.getenv("EMBEDDING_DEVICE")

        # 向量存储
        if os.getenv("MILVUS_HOST"):
            self._configs["vector_store"].host = os.getenv("MILVUS_HOST")
        if os.getenv("MILVUS_PORT"):
            self._configs["vector_store"].port = int(os.getenv("MILVUS_PORT"))
        if os.getenv("MILVUS_COLLECTION"):
            self._configs["vector_store"].collection_name = os.getenv("MILVUS_COLLECTION")

        # Redis
        if os.getenv("REDIS_HOST"):
            self._configs["redis"].host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self._configs["redis"].port = int(os.getenv("REDIS_PORT"))

    def _load_from_file(self):
        """从配置文件加载"""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                logger.warning(f"配置文件不存在: {self.config_file}")
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 更新配置
            for key, value in data.items():
                if key in self._configs:
                    if isinstance(value, dict):
                        # 更新dataclass字段
                        config_obj = self._configs[key]
                        for field_name, field_value in value.items():
                            if hasattr(config_obj, field_name):
                                setattr(config_obj, field_name, field_value)

            logger.info(f"已加载配置文件: {self.config_file}")

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")

    def get(self, key: str) -> Any:
        """
        获取配置

        Args:
            key: 配置键（如 "embedding", "vector_store"）

        Returns:
            配置对象
        """
        return self._configs.get(key)

    def get_embedding_config(self) -> EmbeddingServiceConfig:
        """获取嵌入服务配置"""
        return self._configs["embedding"]

    def get_vector_store_config(self) -> VectorStoreConfig:
        """获取向量存储配置"""
        return self._configs["vector_store"]

    def get_redis_config(self) -> RedisConfig:
        """获取Redis配置"""
        return self._configs["redis"]

    def get_document_parser_config(self) -> DocumentParserConfig:
        """获取文档解析配置"""
        return self._configs["document_parser"]

    def get_chunking_config(self) -> ChunkingConfig:
        """获取分块配置"""
        return self._configs["chunking"]

    def update(self, key: str, config: Any):
        """
        更新配置

        Args:
            key: 配置键
            config: 新的配置对象
        """
        self._configs[key] = config
        logger.info(f"配置已更新: {key}")

    def export(self) -> Dict[str, Any]:
        """
        导出所有配置

        Returns:
            配置字典
        """
        result = {}
        for key, config_obj in self._configs.items():
            if hasattr(config_obj, '__dataclass_fields__'):
                # dataclass对象
                result[key] = {
                    field_name: getattr(config_obj, field_name)
                    for field_name in config_obj.__dataclass_fields__
                }
            else:
                result[key] = config_obj
        return result

    def save_to_file(self, file_path: str):
        """
        保存配置到文件

        Args:
            file_path: 文件路径
        """
        try:
            config_data = self.export()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise

    def validate(self) -> List[str]:
        """
        验证配置

        Returns:
            错误信息列表（空列表表示无错误）
        """
        errors = []

        # 验证嵌入配置
        try:
            embedding_config = self.get_embedding_config()
            if embedding_config.batch_size <= 0:
                errors.append("embedding.batch_size 必须大于0")
            if embedding_config.embedding_dimension <= 0:
                errors.append("embedding.embedding_dimension 必须大于0")
        except Exception as e:
            errors.append(f"嵌入配置验证失败: {e}")

        # 验证向量存储配置
        try:
            vector_config = self.get_vector_store_config()
            if vector_config.port <= 0 or vector_config.port > 65535:
                errors.append(f"vector_store.port 无效: {vector_config.port}")
        except Exception as e:
            errors.append(f"向量存储配置验证失败: {e}")

        # 验证Redis配置
        try:
            redis_config = self.get_redis_config()
            if redis_config.port <= 0 or redis_config.port > 65535:
                errors.append(f"redis.port 无效: {redis_config.port}")
        except Exception as e:
            errors.append(f"Redis配置验证失败: {e}")

        return errors

    def print_config(self):
        """打印所有配置（用于调试）"""
        print("=" * 80)
        print("统一配置中心")
        print("=" * 80)

        for key, config_obj in self._configs.items():
            print(f"\n[{key.upper()}]")
            if hasattr(config_obj, '__dataclass_fields__'):
                for field_name in config_obj.__dataclass_fields__:
                    value = getattr(config_obj, field_name)
                    print(f"  {field_name}: {value}")
            else:
                print(f"  {config_obj}")

        print("=" * 80)

# ============================================================================
# 全局配置实例
# ============================================================================

_global_config: Optional[UnifiedConfig] = None

def get_config(config_file: Optional[str] = None) -> UnifiedConfig:
    """
    获取全局配置实例

    Args:
        config_file: 可选的配置文件路径

    Returns:
        UnifiedConfig实例
    """
    global _global_config

    if _global_config is None:
        # 从环境变量获取配置文件路径
        if config_file is None:
            config_file = os.getenv("CONFIG_FILE")

        _global_config = UnifiedConfig(config_file)
        logger.info("全局配置已创建")

    return _global_config

def reload_config(config_file: Optional[str] = None):
    """
    重新加载配置

    Args:
        config_file: 可选的配置文件路径
    """
    global _global_config

    _global_config = UnifiedConfig(config_file)
    logger.info("配置已重新加载")

    return _global_config

# ============================================================================
# 便捷函数
# ============================================================================

def get_embedding_config() -> EmbeddingServiceConfig:
    """获取嵌入服务配置（便捷函数）"""
    return get_config().get_embedding_config()

def get_vector_store_config() -> VectorStoreConfig:
    """获取向量存储配置（便捷函数）"""
    return get_config().get_vector_store_config()

def get_redis_config() -> RedisConfig:
    """获取Redis配置（便捷函数）"""
    return get_config().get_redis_config()

# 导出
__all__ = [
    'UnifiedConfig',
    'get_config',
    'reload_config',
    'get_embedding_config',
    'get_vector_store_config',
    'get_redis_config',
    'EmbeddingServiceConfig',
    'VectorStoreConfig',
    'RedisConfig',
    'DocumentParserConfig',
    'ChunkingConfig'
]
