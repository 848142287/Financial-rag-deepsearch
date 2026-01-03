"""
文档解析流水线优化配置

控制各种优化功能的开关和参数
"""

from pydantic import BaseModel, Field

class EntityExtractionConfig(BaseModel):
    """实体抽取优化配置"""
    enabled: bool = Field(default=True, description="启用优化的实体抽取")
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="最低置信度阈值")
    enable_llm_fallback: bool = Field(default=True, description="启用LLM fallback")
    batch_processing: bool = Field(default=True, description="启用批量处理")
    max_llm_calls: int = Field(default=10, ge=0, description="最大LLM调用次数")

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "min_confidence": 0.7,
                "enable_llm_fallback": True,
                "batch_processing": True,
                "max_llm_calls": 10
            }
        }

class RelationExtractionConfig(BaseModel):
    """关系抽取优化配置"""
    enabled: bool = Field(default=True, description="启用优化的关系抽取")
    use_enhanced_rules: bool = Field(default=True, description="使用增强规则")
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="最低置信度")
    enable_llm_fallback: bool = Field(default=True, description="启用LLM补充")
    min_relations_for_llm: int = Field(default=5, ge=0, description="触发LLM的最小关系数")

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "use_enhanced_rules": True,
                "min_confidence": 0.7,
                "enable_llm_fallback": True,
                "min_relations_for_llm": 5
            }
        }

class EmbeddingOptimizationConfig(BaseModel):
    """Embedding优化配置"""
    enabled: bool = Field(default=True, description="启用优化的embedding策略")
    use_adaptive_chunking: bool = Field(default=True, description="使用自适应chunking")
    add_metadata_prefix: bool = Field(default=True, description="添加元数据前缀")
    include_document_type: bool = Field(default=True, description="包含文档类型")
    include_broker_name: bool = Field(default=True, description="包含券商名称")
    include_date: bool = Field(default=True, description="包含日期")
    enhance_financial_terms: bool = Field(default=True, description="增强金融术语")
    enable_cache: bool = Field(default=True, description="启用embedding缓存")
    cache_size: int = Field(default=1000, ge=0, description="缓存大小")
    cache_ttl: int = Field(default=3600, ge=0, description="缓存TTL(秒)")

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "use_adaptive_chunking": True,
                "add_metadata_prefix": True,
                "include_document_type": True,
                "include_broker_name": True,
                "include_date": True,
                "enhance_financial_terms": True,
                "enable_cache": True,
                "cache_size": 1000,
                "cache_ttl": 3600
            }
        }

class PerformanceOptimizationConfig(BaseModel):
    """性能优化配置"""
    use_batch_operations: bool = Field(default=True, description="使用批量数据库操作")
    batch_size: int = Field(default=16, ge=1, le=100, description="批处理大小")
    enable_concurrent_processing: bool = Field(default=False, description="启用并发处理")
    max_concurrent_tasks: int = Field(default=4, ge=1, le=16, description="最大并发任务数")
    enable_fallback: bool = Field(default=True, description="启用降级策略")

    class Config:
        json_schema_extra = {
            "example": {
                "use_batch_operations": True,
                "batch_size": 16,
                "enable_concurrent_processing": False,
                "max_concurrent_tasks": 4,
                "enable_fallback": True
            }
        }

class OptimizationConfig(BaseModel):
    """总体优化配置"""
    # 子配置
    entity_extraction: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)
    relation_extraction: RelationExtractionConfig = Field(default_factory=RelationExtractionConfig)
    embedding: EmbeddingOptimizationConfig = Field(default_factory=EmbeddingOptimizationConfig)
    performance: PerformanceOptimizationConfig = Field(default_factory=PerformanceOptimizationConfig)

    # 全局开关
    all_optimizations_enabled: bool = Field(default=True, description="全局启用所有优化")

    # 监控配置
    enable_monitoring: bool = Field(default=True, description="启用性能监控")
    log_performance_metrics: bool = Field(default=True, description="记录性能指标")

    class Config:
        json_schema_extra = {
            "example": {
                "all_optimizations_enabled": True,
                "enable_monitoring": True,
                "log_performance_metrics": True,
                "entity_extraction": {
                    "enabled": True,
                    "min_confidence": 0.7
                },
                "relation_extraction": {
                    "enabled": True,
                    "use_enhanced_rules": True
                },
                "embedding": {
                    "enabled": True,
                    "use_adaptive_chunking": True,
                    "add_metadata_prefix": True
                },
                "performance": {
                    "use_batch_operations": True,
                    "batch_size": 16
                }
            }
        }

# 默认配置实例
default_config = OptimizationConfig()

# 生产环境配置 (更保守)
production_config = OptimizationConfig(
    entity_extraction=EntityExtractionConfig(
        enabled=True,
        min_confidence=0.75,  # 更高的置信度
        max_llm_calls=5  # 限制LLM调用
    ),
    relation_extraction=RelationExtractionConfig(
        enabled=True,
        min_confidence=0.75,
        min_relations_for_llm=3  # 更早使用LLM
    ),
    embedding=EmbeddingOptimizationConfig(
        enabled=True,
        cache_size=2000,  # 更大的缓存
        cache_ttl=7200  # 更长的TTL
    ),
    performance=PerformanceOptimizationConfig(
        use_batch_operations=True,
        batch_size=32,  # 更大的批次
        enable_concurrent_processing=False  # 生产环境关闭并发
    )
)

# 开发环境配置 (更多优化)
development_config = OptimizationConfig(
    entity_extraction=EntityExtractionConfig(
        enabled=True,
        min_confidence=0.6,  # 更低的置信度,更多实体
        enable_llm_fallback=True,
        max_llm_calls=20
    ),
    relation_extraction=RelationExtractionConfig(
        enabled=True,
        min_confidence=0.6,
        enable_llm_fallback=True,
        min_relations_for_llm=10
    ),
    embedding=EmbeddingOptimizationConfig(
        enabled=True,
        enhance_financial_terms=True,
        cache_size=500,
        cache_ttl=1800
    ),
    performance=PerformanceOptimizationConfig(
        use_batch_operations=True,
        batch_size=8,
        enable_concurrent_processing=True,  # 开发环境启用并发
        max_concurrent_tasks=8
    )
)

def get_config(env: str = "production") -> OptimizationConfig:
    """
    获取指定环境的配置

    Args:
        env: 环境名称 (production/development/testing)

    Returns:
        OptimizationConfig实例
    """
    configs = {
        "production": production_config,
        "development": development_config,
        "testing": default_config
    }
    return configs.get(env, default_config)

def is_optimization_enabled(optimization_name: str, config: OptimizationConfig = None) -> bool:
    """
    检查特定优化是否启用

    Args:
        optimization_name: 优化名称 (entity_extraction/relation_extraction/embedding/performance)
        config: 配置实例 (默认使用default_config)

    Returns:
        是否启用
    """
    if config is None:
        config = default_config

    if not config.all_optimizations_enabled:
        return False

    optimization_map = {
        "entity_extraction": config.entity_extraction.enabled,
        "relation_extraction": config.relation_extraction.enabled,
        "embedding": config.embedding.enabled,
        "performance": config.performance.use_batch_operations
    }

    return optimization_map.get(optimization_name, False)

# 使用示例
if __name__ == "__main__":
    # 获取生产配置
    prod_config = get_config("production")
    print(f"生产环境配置: {prod_config.model_dump_json(indent=2)}")

    # 检查优化是否启用
    print(f"实体抽取启用: {is_optimization_enabled('entity_extraction', prod_config)}")
    print(f"Embedding优化启用: {is_optimization_enabled('embedding', prod_config)}")
