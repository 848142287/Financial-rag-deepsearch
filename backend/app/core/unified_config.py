"""
统一配置管理服务 - 解决配置分散问题

整合所有分散的配置，提供：
1. 统一的配置加载
2. 配置验证（使用Pydantic）
3. 环境变量管理
4. 配置热更新
5. 配置版本管理
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum
import json
import yaml
from pydantic import BaseModel, Field, validator
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# ========================================================================
# 配置模型定义（使用Pydantic进行验证）
# ========================================================================

class LLMProvider(str, Enum):
    """LLM提供商"""
    DEEPSEEK = "deepseek"
    GLM = "glm"
    QWEN = "qwen"
    OPENAI = "openai"

class EmbeddingProvider(str, Enum):
    """嵌入模型提供商"""
    BGE_LOCAL = "bge_local"
    QWEN_API = "qwen_api"
    OPENAI = "openai"

class LLMConfig(BaseModel):
    """LLM配置"""
    provider: LLMProvider = Field(default=LLMProvider.DEEPSEEK, description="LLM提供商")
    model: str = Field(default="deepseek-chat", description="模型名称")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    base_url: Optional[str] = Field(default=None, description="API基础URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=4096, gt=0, description="最大token数")
    timeout: int = Field(default=120, gt=0, description="超时时间（秒）")
    max_retries: int = Field(default=3, ge=0, le=10, description="最大重试次数")
    enable_fallback: bool = Field(default=True, description="启用备用模型")

    @validator('api_key')
    def validate_api_key(cls, v, values):
        """验证API密钥"""
        if values.get('provider') in [LLMProvider.DEEPSEEK, LLMProvider.OPENAI]:
            if not v:
                raise ValueError(f"{values.get('provider')} 需要配置API密钥")
        return v

class EmbeddingConfig(BaseModel):
    """向量嵌入配置"""
    provider: EmbeddingProvider = Field(default=EmbeddingProvider.BGE_LOCAL, description="嵌入提供商")
    model_name: str = Field(default="BAAI/bge-large-zh-v1.5", description="模型名称")
    dimension: int = Field(default=1024, gt=0, description="向量维度")
    batch_size: int = Field(default=32, gt=0, description="批量大小")
    device: str = Field(default="cuda", description="设备（cuda/cpu）")
    normalize: bool = Field(default=True, description="是否归一化")
    cache_enabled: bool = Field(default=True, description="启用缓存")

class DatabaseConfig(BaseModel):
    """数据库配置"""
    host: str = Field(default="localhost", description="主机地址")
    port: int = Field(default=3306, gt=0, description="端口")
    username: str = Field(default="rag_user", description="用户名")
    password: str = Field(default="rag_password", description="密码")
    database: str = Field(default="financial_rag", description="数据库名")
    pool_size: int = Field(default=5, gt=0, description="连接池大小")
    max_overflow: int = Field(default=10, ge=0, description="最大溢出连接数")
    pool_timeout: int = Field(default=30, gt=0, description="连接池超时（秒）")

class MilvusConfig(BaseModel):
    """Milvus配置"""
    host: str = Field(default="localhost", description="主机地址")
    port: int = Field(default=19530, gt=0, description="端口")
    collection_name: str = Field(default="document_chunks", description="集合名称")
    dimension: int = Field(default=1024, gt=0, description="向量维度")
    index_type: str = Field(default="HNSW", description="索引类型")
    metric_type: str = Field(default="IP", description="度量类型（IP/L2）")

class Neo4jConfig(BaseModel):
    """Neo4j配置"""
    uri: str = Field(default="bolt://localhost:7687", description="连接URI")
    username: str = Field(default="neo4j", description="用户名")
    password: str = Field(default="password", description="密码")
    database: str = Field(default="neo4j", description="数据库名")

class MinioConfig(BaseModel):
    """MinIO配置"""
    endpoint: str = Field(default="localhost:9000", description="端点地址")
    access_key: str = Field(default="minioadmin", description="访问密钥")
    secret_key: str = Field(default="minioadmin", description="秘密密钥")
    bucket_name: str = Field(default="financial-docs", description="桶名称")
    secure: bool = Field(default=False, description="使用HTTPS")

class ProcessingConfig(BaseModel):
    """文档处理配置"""
    enable_multimodal: bool = Field(default=True, description="启用多模态分析")
    enable_entity_extraction: bool = Field(default=True, description="启用实体提取")
    enable_knowledge_graph: bool = Field(default=True, description="启用知识图谱")
    markdown_supplement_enabled: bool = Field(default=True, description="启用Markdown补充")

    # 并行化配置
    max_parallel_chunks: int = Field(default=10, gt=0, description="最大并行chunk数")
    vector_batch_size: int = Field(default=50, gt=0, description="向量批量大小")
    enable_parallel_vectorization: bool = Field(default=True, description="启用并行向量化")

    # 性能优化配置
    chunk_size: int = Field(default=512, gt=0, description="chunk大小（tokens）")
    chunk_overlap: int = Field(default=50, ge=0, description="chunk重叠（tokens）")
    skip_ocr_if_text_exists: bool = Field(default=True, description="有文本时跳过OCR")

class CeleryConfig(BaseModel):
    """Celery配置"""
    broker_url: str = Field(default="redis://localhost:6379/0", description="Broker URL")
    result_backend: str = Field(default="redis://localhost:6379/1", description="结果后端")
    worker_concurrency: int = Field(default=10, gt=0, description="worker并发数")
    task_soft_time_limit: int = Field(default=1800, gt=0, description="软超时（秒）")
    task_time_limit: int = Field(default=2100, gt=0, description="硬超时（秒）")
    task_max_retries: int = Field(default=2, ge=0, le=10, description="最大重试次数")
    prefetch_multiplier: int = Field(default=4, ge=1, description="预取倍数")

class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="json", description="日志格式（json/text）")
    output_dir: str = Field(default="logs", description="输出目录")
    rotation: str = Field(default="500 MB", description="日志轮转大小")
    retention: str = Field(default="30 days", description="日志保留时间")

class AppConfig(BaseModel):
    """应用总配置"""
    app_name: str = Field(default="Financial RAG System", description="应用名称")
    version: str = Field(default="1.0.0", description="版本号")
    environment: str = Field(default="development", description="环境（development/production）")
    debug: bool = Field(default=False, description="调试模式")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        arbitrary_types_allowed = True

# ========================================================================
# 配置管理器
# ========================================================================

class ConfigManager:
    """
    统一配置管理器

    功能：
    1. 从多个来源加载配置（环境变量、配置文件、默认值）
    2. 配置验证（Pydantic）
    3. 配置热更新
    4. 配置导出
    """

    def __init__(self):
        self._config: Optional[AppConfig] = None
        self._config_sources: List[str] = []
        self._watchers = []

    def load_config(
        self,
        config_file: Optional[str] = None,
        env_prefix: str = "RAG_",
        use_env_vars: bool = True
    ) -> AppConfig:
        """
        加载配置

        优先级：环境变量 > 配置文件 > 默认值

        Args:
            config_file: 配置文件路径（YAML/JSON）
            env_prefix: 环境变量前缀
            use_env_vars: 是否使用环境变量

        Returns:
            AppConfig
        """
        config_dict = {}

        # 1. 加载配置文件
        if config_file and Path(config_file).exists():
            config_dict = self._load_config_file(config_file)
            self._config_sources.append(f"file:{config_file}")
            logger.info(f"✅ 从配置文件加载配置: {config_file}")

        # 2. 加载环境变量
        if use_env_vars:
            env_dict = self._load_env_variables(env_prefix)
            config_dict = {**config_dict, **env_dict}
            if env_dict:
                self._config_sources.append(f"env:{len(env_dict)} vars")
                logger.info(f"✅ 从环境变量加载 {len(env_dict)} 个配置项")

        # 3. 验证并创建配置对象
        self._config = AppConfig(**config_dict)

        # 4. 记录配置来源
        if not self._config_sources:
            self._config_sources.append("defaults")

        logger.info(f"✅ 配置加载完成，来源: {', '.join(self._config_sources)}")

        return self._config

    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """从文件加载配置"""
        path = Path(config_file)

        if not path.exists():
            logger.warning(f"配置文件不存在: {config_file}")
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif path.suffix == '.json':
                    return json.load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {path.suffix}")
                    return {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    def _load_env_variables(self, prefix: str) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_dict = {}

        # LLM配置
        if f"{prefix}LLM_PROVIDER" in os.environ:
            env_dict['llm'] = {}
            env_dict['llm']['provider'] = os.environ[f"{prefix}LLM_PROVIDER"]
        if f"{prefix}LLM_API_KEY" in os.environ:
            if 'llm' not in env_dict:
                env_dict['llm'] = {}
            env_dict['llm']['api_key'] = os.environ[f"{prefix}LLM_API_KEY"]
        if f"{prefix}LLM_BASE_URL" in os.environ:
            if 'llm' not in env_dict:
                env_dict['llm'] = {}
            env_dict['llm']['base_url'] = os.environ[f"{prefix}LLM_BASE_URL"]

        # 数据库配置
        if f"{prefix}DB_HOST" in os.environ:
            env_dict['database'] = {}
            env_dict['database']['host'] = os.environ[f"{prefix}DB_HOST"]
        if f"{prefix}DB_PORT" in os.environ:
            if 'database' not in env_dict:
                env_dict['database'] = {}
            env_dict['database']['port'] = int(os.environ[f"{prefix}DB_PORT"])
        if f"{prefix}DB_USERNAME" in os.environ:
            if 'database' not in env_dict:
                env_dict['database'] = {}
            env_dict['database']['username'] = os.environ[f"{prefix}DB_USERNAME"]
        if f"{prefix}DB_PASSWORD" in os.environ:
            if 'database' not in env_dict:
                env_dict['database'] = {}
            env_dict['database']['password'] = os.environ[f"{prefix}DB_PASSWORD"]
        if f"{prefix}DB_DATABASE" in os.environ:
            if 'database' not in env_dict:
                env_dict['database'] = {}
            env_dict['database']['database'] = os.environ[f"{prefix}DB_DATABASE"]

        # Milvus配置
        if f"{prefix}MILVUS_HOST" in os.environ:
            env_dict['milvus'] = {}
            env_dict['milvus']['host'] = os.environ[f"{prefix}MILVUS_HOST"]
        if f"{prefix}MILVUS_PORT" in os.environ:
            if 'milvus' not in env_dict:
                env_dict['milvus'] = {}
            env_dict['milvus']['port'] = int(os.environ[f"{prefix}MILVUS_PORT"])

        # Neo4j配置
        if f"{prefix}NEO4J_URI" in os.environ:
            env_dict['neo4j'] = {}
            env_dict['neo4j']['uri'] = os.environ[f"{prefix}NEO4J_URI"]
        if f"{prefix}NEO4J_USERNAME" in os.environ:
            if 'neo4j' not in env_dict:
                env_dict['neo4j'] = {}
            env_dict['neo4j']['username'] = os.environ[f"{prefix}NEO4J_USERNAME"]
        if f"{prefix}NEO4J_PASSWORD" in os.environ:
            if 'neo4j' not in env_dict:
                env_dict['neo4j'] = {}
            env_dict['neo4j']['password'] = os.environ[f"{prefix}NEO4J_PASSWORD"]

        # MinIO配置
        if f"{prefix}MINIO_ENDPOINT" in os.environ:
            env_dict['minio'] = {}
            env_dict['minio']['endpoint'] = os.environ[f"{prefix}MINIO_ENDPOINT"]
        if f"{prefix}MINIO_ACCESS_KEY" in os.environ:
            if 'minio' not in env_dict:
                env_dict['minio'] = {}
            env_dict['minio']['access_key'] = os.environ[f"{prefix}MINIO_ACCESS_KEY"]
        if f"{prefix}MINIO_SECRET_KEY" in os.environ:
            if 'minio' not in env_dict:
                env_dict['minio'] = {}
            env_dict['minio']['secret_key'] = os.environ[f"{prefix}MINIO_SECRET_KEY"]

        # 并行化配置
        if f"{prefix}MAX_PARALLEL_CHUNKS" in os.environ:
            env_dict['processing'] = {}
            env_dict['processing']['max_parallel_chunks'] = int(os.environ[f"{prefix}MAX_PARALLEL_CHUNKS"])
        if f"{prefix}VECTOR_BATCH_SIZE" in os.environ:
            if 'processing' not in env_dict:
                env_dict['processing'] = {}
            env_dict['processing']['vector_batch_size'] = int(os.environ[f"{prefix}VECTOR_BATCH_SIZE"])
        if f"{prefix}ENABLE_PARALLEL_VECTORIZATION" in os.environ:
            if 'processing' not in env_dict:
                env_dict['processing'] = {}
            env_dict['processing']['enable_parallel_vectorization'] = os.environ[f"{prefix}ENABLE_PARALLEL_VECTORIZATION"].lower() == 'true'

        return env_dict

    @property
    def config(self) -> AppConfig:
        """获取当前配置"""
        if self._config is None:
            # 如果配置未加载，使用默认配置
            self._config = AppConfig()
            logger.warning("⚠️ 配置未加载，使用默认配置")
        return self._config

    def get_llm_config(self) -> LLMConfig:
        """获取LLM配置"""
        return self.config.llm

    def get_embedding_config(self) -> EmbeddingConfig:
        """获取嵌入配置"""
        return self.config.embedding

    def get_database_config(self) -> DatabaseConfig:
        """获取数据库配置"""
        return self.config.database

    def get_milvus_config(self) -> MilvusConfig:
        """获取Milvus配置"""
        return self.config.milvus

    def get_neo4j_config(self) -> Neo4jConfig:
        """获取Neo4j配置"""
        return self.config.neo4j

    def get_minio_config(self) -> MinioConfig:
        """获取MinIO配置"""
        return self.config.minio

    def get_processing_config(self) -> ProcessingConfig:
        """获取处理配置"""
        return self.config.processing

    def get_celery_config(self) -> CeleryConfig:
        """获取Celery配置"""
        return self.config.celery

    def export_config(
        self,
        output_file: str,
        format: str = 'yaml'
    ):
        """
        导出配置到文件

        Args:
            output_file: 输出文件路径
            format: 导出格式（yaml/json）
        """
        config_dict = self.config.dict()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if format == 'yaml':
                    yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
                elif format == 'json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的导出格式: {format}")

            logger.info(f"✅ 配置已导出到: {output_file}")

        except Exception as e:
            logger.error(f"导出配置失败: {e}")

    def reload_config(
        self,
        config_file: Optional[str] = None,
        env_prefix: str = "RAG_"
    ):
        """重新加载配置"""
        logger.info("🔄 重新加载配置...")
        self.load_config(config_file, env_prefix)
        logger.info("✅ 配置重新加载完成")

# 全局单例
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config(
    config_file: Optional[str] = None,
    env_prefix: str = "RAG_"
) -> AppConfig:
    """加载配置（便捷函数）"""
    manager = get_config_manager()
    return manager.load_config(config_file, env_prefix)

def get_config() -> AppConfig:
    """获取当前配置（便捷函数）"""
    return get_config_manager().config
