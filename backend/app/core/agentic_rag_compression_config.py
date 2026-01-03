"""
Agentic RAG 上下文压缩配置

定义不同检索级别的压缩策略和参数
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class CompressionConfig:
    """压缩配置"""
    enabled: bool = True
    method: str = "hierarchical"  # hierarchical, embeddings, llm

    # L1: Embeddings过滤
    l1_top_k: int = 10
    l1_similarity_threshold: float = 0.6
    l1_model: str = "BAAI/bge-large-zh-v1.5"

    # L3: LLM提取
    use_llm: bool = True
    llm_compression_rate: float = 0.5
    llm_max_length: int = 2000

    # 触发条件
    min_doc_count_for_compression: int = 3
    min_tokens_for_compression: int = 2000


@dataclass
class LevelCompressionConfig:
    """级别特定的压缩配置"""

    # Fast级别配置
    fast: CompressionConfig = field(default_factory=lambda: CompressionConfig(
        enabled=True,
        method="hierarchical",
        l1_top_k=5,
        l1_similarity_threshold=0.65,
        use_llm=False,  # Fast模式不使用LLM压缩
        llm_compression_rate=1.0,
        llm_max_length=2000,
        min_doc_count_for_compression=6,
        min_tokens_for_compression=3000
    ))

    # Enhanced级别配置
    enhanced: CompressionConfig = field(default_factory=lambda: CompressionConfig(
        enabled=True,
        method="hierarchical",
        l1_top_k=10,
        l1_similarity_threshold=0.6,
        use_llm=True,
        llm_compression_rate=0.6,
        llm_max_length=1500,
        min_doc_count_for_compression=3,
        min_tokens_for_compression=2000
    ))

    # DeepSearch级别配置
    deep_search: CompressionConfig = field(default_factory=lambda: CompressionConfig(
        enabled=True,
        method="hierarchical",
        l1_top_k=15,
        l1_similarity_threshold=0.55,
        use_llm=True,
        llm_compression_rate=0.4,
        llm_max_length=1200,
        min_doc_count_for_compression=2,
        min_tokens_for_compression=1500
    ))


# 全局配置
_compression_config = LevelCompressionConfig()


def get_compression_config(level: str = "enhanced") -> CompressionConfig:
    """
    获取指定级别的压缩配置

    Args:
        level: 检索级别 (fast/enhanced/deep_search)

    Returns:
        CompressionConfig: 压缩配置
    """
    return getattr(_compression_config, level, _compression_config.enhanced)


def update_compression_config(level: str, **kwargs):
    """
    更新指定级别的压缩配置

    Args:
        level: 检索级别
        **kwargs: 要更新的配置项
    """
    config = get_compression_config(level)

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"未知的配置项: {key}")


def get_all_compression_configs() -> Dict[str, Dict[str, Any]]:
    """
    获取所有级别的压缩配置

    Returns:
        字典格式的所有配置
    """
    return {
        "fast": get_compression_config("fast").__dict__,
        "enhanced": get_compression_config("enhanced").__dict__,
        "deep_search": get_compression_config("deep_search").__dict__
    }


# 预设配置模板
COMPRESSION_PRESETS = {
    "conservative": {
        "fast": {"l1_top_k": 8, "use_llm": False},
        "enhanced": {"l1_top_k": 12, "use_llm": True, "llm_compression_rate": 0.7},
        "deep_search": {"l1_top_k": 20, "use_llm": True, "llm_compression_rate": 0.6}
    },
    "balanced": {
        "fast": {"l1_top_k": 5, "use_llm": False},
        "enhanced": {"l1_top_k": 10, "use_llm": True, "llm_compression_rate": 0.5},
        "deep_search": {"l1_top_k": 15, "use_llm": True, "llm_compression_rate": 0.4}
    },
    "aggressive": {
        "fast": {"l1_top_k": 3, "use_llm": False},
        "enhanced": {"l1_top_k": 7, "use_llm": True, "llm_compression_rate": 0.4},
        "deep_search": {"l1_top_k": 10, "use_llm": True, "llm_compression_rate": 0.3}
    }
}


def apply_compression_preset(preset_name: str):
    """
    应用预设压缩配置

    Args:
        preset_name: 预设名称 (conservative/balanced/aggressive)
    """
    if preset_name not in COMPRESSION_PRESETS:
        raise ValueError(f"未知的预设: {preset_name}")

    preset = COMPRESSION_PRESETS[preset_name]

    for level, config in preset.items():
        update_compression_config(level, **config)


# 导出配置函数
__all__ = [
    "CompressionConfig",
    "LevelCompressionConfig",
    "get_compression_config",
    "update_compression_config",
    "get_all_compression_configs",
    "apply_compression_preset",
    "COMPRESSION_PRESETS"
]
