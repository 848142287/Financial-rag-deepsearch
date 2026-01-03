"""
应用配置设置 - 清理版
只保留BGE本地模型和GLM模型配置
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

    # ============================================================================
    # 应用基础配置
    # ============================================================================
    app_name: str = "Financial RAG System"
    debug: bool = True
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # 缓存目录配置
    cache_dir: str = "/tmp/.cache"
    huggingface_cache_dir: str = "/tmp/.cache/huggingface"
    transformers_cache_dir: str = "/tmp/.cache/transformers"

    # ============================================================================
    # 数据库配置
    # ============================================================================
    database_url: str = Field(default="mysql+pymysql://rag_user:rag_password@mysql:3306/financial_rag")

    # Redis配置
    redis_url: str = Field(default="redis://redis:6379/0")
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = None

    # ============================================================================
    # 存储配置
    # ============================================================================
    # MinIO配置
    minio_endpoint: str = Field(default="minio:9000")
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "documents"
    minio_secure: bool = False

    # ============================================================================
    # 向量数据库配置
    # ============================================================================
    # Milvus配置
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = "document_embeddings"

    # Neo4j知识图谱配置
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4j123456")

    # ============================================================================
    # Milvus索引配置（优化）
    # ============================================================================
    # 索引类型：IVF_FLAT（平衡速度和精度）、HNSW（最高速度）、IVF_PQ（最省内存）
    milvus_index_type: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
    # IVF参数：聚类中心数量（建议为数据量的平方根，如100万向量则nlist=1000）
    milvus_nlist: int = int(os.getenv("MILVUS_NLIST", "256"))
    # 搜索参数：探查的聚类中心数量（越大召回率越高，但速度越慢）
    milvus_nprobe: int = int(os.getenv("MILVUS_NPROBE", "16"))
    # HNSW参数（如果使用HNSW索引）
    milvus_M: int = int(os.getenv("MILVUS_M", "16"))  # 每个节点的连接数
    milvus_efConstruction: int = int(os.getenv("MILVUS_EF_CONSTRUCTION", "256"))  # 构建时的搜索深度

    # ============================================================================
    # 多模型配置 - 支持主模型和备份模型
    # ============================================================================

    # ============================================================================
    # 视觉模型配置（用于图片分析、OCR、多模态文档解析）
    # ============================================================================
    # Qwen-VL-Max（主视觉模型 - 阿里云DashScope）
    qwen_vl_api_key: str = Field(
        default="sk-5233a3a4b1a24426b6846a432794bbe2",
        description="Qwen-VL API Key (DashScope)"
    )
    qwen_vl_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Qwen-VL API base URL (DashScope compatible API)"
    )
    # Qwen-VL模型选项
    qwen_vl_max_model: str = "qwen-vl-max-latest"  # 最强大的视觉模型
    qwen_vl_plus_model: str = "qwen-vl-plus"  # 平衡性能和速度
    qwen_vl_ocr_model: str = "qwen-vl-ocr-latest"  # 专门用于OCR
    qwen_vl_model: str = "qwen-vl-plus"  # 默认使用qwen-vl-plus
    qwen_vl_max_tokens: int = 8000
    qwen_vl_temperature: float = 0.1  # 低温度保证准确性

    # GLM-4.6V（备份视觉模型 - 智谱AI）
    glm_vl_api_key: str = Field(
        default="3be7fffd8963497684e2746d7f8e8d36.nVBIIb7EAOfxDYMx",
        description="GLM-4.6V API Key (Zhipu AI)"
    )
    glm_vl_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        description="GLM-4.6V API base URL"
    )
    glm_vl_model: str = "glm-4.6v"  # 备份视觉模型
    glm_vl_max_tokens: int = 8000
    glm_vl_temperature: float = 0.1

    # 主视觉模型选择
    primary_vision_model: str = "qwen-vl"  # qwen-vl 或 glm-vl
    fallback_vision_model: str = "glm-vl"  # 备份模型

    # ============================================================================
    # 文本模型配置（用于文本生成、摘要、实体抽取、检索结果生成）
    # ============================================================================
    # Deepseek（主文本模型）
    deepseek_api_key: str = Field(
        default="sk-b5dc5233a41944088a7a38b9f45c4251",
        description="Deepseek API Key"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="Deepseek API base URL"
    )
    deepseek_model: str = "deepseek-chat"  # 主文本模型
    deepseek_temperature: float = 0.7
    deepseek_max_tokens: int = 4000

    # GLM-4.7（备份文本模型 - 智谱AI）
    glm_api_key: str = Field(
        default="3be7fffd8963497684e2746d7f8e8d36.nVBIIb7EAOfxDYMx",
        description="GLM-4.7 API Key (Zhipu AI)"
    )
    glm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        description="GLM-4.7 API base URL"
    )
    glm_model: str = "glm-4.7"  # 备份文本模型
    glm_temperature: float = 0.7
    glm_max_tokens: int = 4000

    # 主文本模型选择
    primary_llm_model: str = "deepseek"  # deepseek 或 glm
    fallback_llm_model: str = "glm"  # 备份模型

    # ============================================================================
    # BGE本地模型配置（主模型）
    # ============================================================================
    # BGE嵌入模型配置 - bge-large-zh-v1.5
    bge_embedding_model_name: str = "bge-large-zh-v1.5"
    bge_embedding_model_path: str = Field(
        default="/app/models/bge-large-zh-v1.5",  # 本地模型路径
        description="BGE嵌入模型路径"
    )
    bge_embedding_device: str = "cpu"  # cpu, cuda, mps
    bge_embedding_max_length: int = 512
    bge_embedding_batch_size: int = 32
    bge_embedding_dimension: int = 1024

    # BGE排序模型配置 - bge-reranker-v2-m3
    bge_reranker_model_name: str = "bge-reranker-v2-m3"
    bge_reranker_model_path: str = Field(
        default="/app/models/bge-reranker-v2-m3",  # 本地模型路径
        description="BGE排序模型路径"
    )
    bge_reranker_device: str = "cpu"
    bge_reranker_max_length: int = 512
    bge_reranker_batch_size: int = 16
    bge_reranker_dimension: int = 1024

    # ============================================================================
    # 模型降级策略配置
    # ============================================================================
    enable_local_embedding: bool = True  # 启用本地BGE嵌入模型
    enable_local_reranker: bool = True  # 启用本地BGE排序模型
    enable_api_fallback: bool = False  # 不启用API降级（只使用本地模型）

    # ============================================================================
    # 文档处理配置
    # ============================================================================
    max_file_size_mb: int = 200
    upload_dir: str = "./data/uploads"
    file_storage_path: str = "./data/processed"
    chunk_size: int = 512
    chunk_overlap: int = 25  # 优化：从50减少到25，降低重复率（约10%重叠）
    max_chunks_per_document: int = 1000
    supported_file_types: List[str] = ["pdf", "doc", "docx", "xlsx", "txt", "md", "ppt", "pptx"]

    # Markdown补充信息配置
    markdown_supplement_enabled: bool = os.getenv("MARKDOWN_SUPPLEMENT_ENABLED", "true").lower() == "true"
    markdown_base_dir: str = os.getenv("MARKDOWN_BASE_DIR", "./data/documents/broker_reports/markdown")
    markdown_quality_threshold: float = float(os.getenv("MARKDOWN_QUALITY_THRESHOLD", "0.2"))
    markdown_max_file_size_mb: int = int(os.getenv("MARKDOWN_MAX_FILE_SIZE_MB", "10"))
    markdown_timeout_seconds: int = int(os.getenv("MARKDOWN_TIMEOUT_SECONDS", "5"))

    # 功能开关
    use_unified_document_service: bool = os.getenv("USE_UNIFIED_DOCUMENT_SERVICE", "true").lower() == "true"
    unified_service_percentage: int = int(os.getenv("UNIFIED_SERVICE_PERCENTAGE", "100"))

    # MinerU配置
    mineru_api_url: str = os.getenv("MINERU_API_URL", "http://mineru-api:8000/file_parse")
    mineru_backend: str = os.getenv("MINERU_BACKEND", "pipeline")
    mineru_timeout: int = int(os.getenv("MINERU_TIMEOUT", "600"))

    # 解析器策略配置
    parser_strategy: str = os.getenv("PARSER_STRATEGY", "auto")  # auto, fast, precise, vlm
    parser_enable_fallback: bool = os.getenv("PARSER_ENABLE_FALLBACK", "true").lower() == "true"
    parser_performance_monitoring: bool = os.getenv("PARSER_PERFORMANCE_MONITORING", "true").lower() == "true"

    # ============================================================================
    # 任务队列配置
    # ============================================================================
    # Celery配置
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

    # ============================================================================
    # 日志和监控配置
    # ============================================================================
    log_level: str = "INFO"
    log_file: str = "./storage/logs/app.log"

    # ============================================================================
    # 检索配置（优化后）
    # ============================================================================
    # 相似度阈值（基于数据分析优化）
    search_score_threshold: float = 0.5208  # 平衡召回率和精确度
    search_conservative_threshold: float = 0.5526  # 保守策略（高召回）
    search_aggressive_threshold: float = 0.4999  # 激进策略（高精度）

    # CORS配置
    allowed_hosts: List[str] = ["*"]

    # 性能配置
    max_concurrent_requests: int = 100
    request_timeout: int = 300  # 5分钟

    # 缓存配置
    cache_ttl: int = 3600  # 1小时
    enable_cache: bool = True

    # ============================================================================
    # 额外配置（为了兼容.env文件）
    # ============================================================================
    mysql_root_password: Optional[str] = None
    vite_api_base_url: Optional[str] = None
    vite_ws_url: Optional[str] = None


# 创建配置实例
settings = Settings()

# 确保必要的目录存在
try:
    os.makedirs(settings.upload_dir, exist_ok=True)
except PermissionError:
    pass
try:
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
except (PermissionError, FileNotFoundError):
    pass
