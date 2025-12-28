"""
应用配置设置
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # 应用基础配置
    app_name: str = "Financial RAG System"
    debug: bool = True
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # 缓存目录配置
    cache_dir: str = "/tmp/.cache"
    huggingface_cache_dir: str = "/tmp/.cache/huggingface"
    transformers_cache_dir: str = "/tmp/.cache/transformers"

    # 数据库配置 - 从环境变量读取
    database_url: str = Field(default="mysql+pymysql://rag_user:rag_password@mysql:3306/financial_rag")

    # Redis配置 - 从环境变量读取
    redis_url: str = Field(default="redis://redis:6379/0")
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = None

    # MinIO配置
    minio_endpoint: str = Field(default="minio:9000")
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "documents"
    minio_secure: bool = False

    # Milvus配置 - 从环境变量读取
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = "document_embeddings"

    # Neo4j配置 - 从环境变量读取
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4j123456")

    # MongoDB配置 - 从环境变量读取
    mongodb_url: str = os.getenv("MONGODB_URI", "mongodb://admin:password@mongodb:27017")
    mongodb_database: str = "financial_rag"

    # AI模型配置 - DeepSeek LLM (用于文本生成)
    openai_api_key: str = "sk-b5dc5233a41944088a7a38b9f45c4251"  # DeepSeek API Key
    openai_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # 支持从环境变量读取DeepSeek配置
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: Optional[str] = None

    # Qwen模型配置 (替换GLM-4.6V)
    qwen_api_key: str = "sk-5233a3a4b1a24426b6846a432794bbe2"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_multimodal_model: str = "qwen-vl-plus"  # 更新为qwen-vl-plus
    qwen_ocr_model: str = "qwen-vl-ocr-latest"
    qwen_embedding_model: str = "qwen2.5-vl-embedding"
    qwen_text_embedding_model: str = "text-embedding-v4"
    qwen_rerank_model: str = "qwen3-rerank"
    qwen_max_tokens: int = 8000
    qwen_temperature: float = 0.3

    # Qwen嵌入和重排序模型配置 (替换BGE)
    qwen_primary_embedding_model: str = "qwen2.5-vl-embedding"
    qwen_backup_embedding_model: str = "text-embedding-v4"
    qwen_reranker_model: str = "qwen3-rerank"
    qwen_device: str = "cpu"  # 可改为 "cuda" 如果有GPU
    qwen_max_length: int = 512
    qwen_batch_size: int = 32
    qwen_normalize_embeddings: bool = True
    qwen_cache_size: int = 1000
    qwen_enable_cache: bool = True

    # BGE相关配置已移除，系统已完全迁移到Qwen模型

    # 文档处理配置
    max_file_size_mb: int = 200
    upload_dir: str = "./uploads"
    file_storage_path: str = "./storage/document_results"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_document: int = 1000
    supported_file_types: List[str] = ["pdf", "docx", "xlsx", "txt", "md", "ppt", "pptx"]

    # Celery配置
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # CORS配置
    allowed_hosts: List[str] = ["*"]

    # 性能配置
    max_concurrent_requests: int = 100
    request_timeout: int = 300  # 5分钟

    # 缓存配置
    cache_ttl: int = 3600  # 1小时
    enable_cache: bool = True

    # 额外配置（为了兼容.env文件）
    mysql_root_password: Optional[str] = None
    vite_api_base_url: Optional[str] = None
    vite_ws_url: Optional[str] = None

    # 以下配置不重复定义,使用上面的主配置

# 创建配置实例
settings = Settings()

# 确保必要的目录存在
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
# 在文件末尾添加Qwen配置
# Qwen模型配置
QWEN_CONFIG = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-5233a3a4b1a24426b6846a432794bbe2",
    "models": {
        "multimodal": "qwen-vl-plus",
        "ocr": "qwen-vl-ocr-latest",
        "embedding": "qwen2.5-vl-embedding",
        "text_embedding": "text-embedding-v4",
        "reranking": "qwen3-rerank"
    }
}

# 更新嵌入模型配置
EMBEDDING_MODEL_NAME = "qwen2.5-vl-embedding"
EMBEDDING_PROVIDER = "qwen"

# 更新重排序模型配置
RERANKING_MODEL_NAME = "qwen3-rerank"
RERANKING_PROVIDER = "qwen"
