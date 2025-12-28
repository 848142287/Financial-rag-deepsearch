"""
向量嵌入统一配置
确保所有服务使用一致的向量维度和模型配置
"""
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class EmbeddingModel(str, Enum):
    """支持的嵌入模型"""
    QWEN25_VL = "qwen2.5-vl-embedding"           # 1024维
    QWEN_TEXT_V4 = "text-embedding-v4"            # 1536维
    QWEN_TEXT_V3 = "text-embedding-v3"            # 1024维
    BGE_M3 = "bge-m3"                             # 1024维 (预留)
    DEFAULT = QWEN25_VL


@dataclass
class EmbeddingModelConfig:
    """嵌入模型配置"""
    model_name: str
    dimension: int
    max_sequence_length: int = 8192
    normalize: bool = True
    support_multimodal: bool = False


# 模型配置映射
EMBEDDING_MODEL_CONFIGS: Dict[str, EmbeddingModelConfig] = {
    EmbeddingModel.QWEN25_VL.value: EmbeddingModelConfig(
        model_name=EmbeddingModel.QWEN25_VL.value,
        dimension=1024,
        max_sequence_length=8192,
        normalize=True,
        support_multimodal=True
    ),
    EmbeddingModel.QWEN_TEXT_V4.value: EmbeddingModelConfig(
        model_name=EmbeddingModel.QWEN_TEXT_V4.value,
        dimension=1536,
        max_sequence_length=8192,
        normalize=True,
        support_multimodal=False
    ),
    EmbeddingModel.QWEN_TEXT_V3.value: EmbeddingModelConfig(
        model_name=EmbeddingModel.QWEN_TEXT_V3.value,
        dimension=1024,
        max_sequence_length=8192,
        normalize=True,
        support_multimodal=False
    ),
    EmbeddingModel.BGE_M3.value: EmbeddingModelConfig(
        model_name=EmbeddingModel.BGE_M3.value,
        dimension=1024,
        max_sequence_length=8192,
        normalize=True,
        support_multimodal=False
    ),
}


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    # 默认使用的模型
    primary_model: str = EmbeddingModel.DEFAULT.value

    # 备用模型（降级使用）
    backup_model: str = EmbeddingModel.QWEN_TEXT_V4.value

    # Milvus 连接配置
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "financial_documents"

    # 索引配置
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"  # IP, COSINE, L2
    nlist: int = 128

    # 搜索配置
    default_top_k: int = 10
    nprobe: int = 10

    @property
    def dimension(self) -> int:
        """获取当前主模型的向量维度"""
        return EMBEDDING_MODEL_CONFIGS[self.primary_model].dimension

    @property
    def backup_dimension(self) -> int:
        """获取备用模型的向量维度"""
        return EMBEDDING_MODEL_CONFIGS[self.backup_model].dimension


# 全局配置实例
vector_config = VectorStoreConfig()


def get_model_config(model_name: Optional[str] = None) -> EmbeddingModelConfig:
    """
    获取模型配置

    Args:
        model_name: 模型名称，None则使用主模型

    Returns:
        EmbeddingModelConfig
    """
    model_name = model_name or vector_config.primary_model
    if model_name not in EMBEDDING_MODEL_CONFIGS:
        raise ValueError(f"Unknown embedding model: {model_name}. "
                        f"Available: {list(EMBEDDING_MODEL_CONFIGS.keys())}")
    return EMBEDDING_MODEL_CONFIGS[model_name]


def get_dimension(model_name: Optional[str] = None) -> int:
    """
    获取向量维度

    Args:
        model_name: 模型名称，None则使用主模型

    Returns:
        int: 向量维度
    """
    return get_model_config(model_name).dimension


def validate_embedding_dimension(
    embedding: list,
    model_name: Optional[str] = None
) -> bool:
    """
    验证向量维度是否正确

    Args:
        embedding: 嵌入向量
        model_name: 模型名称

    Returns:
        bool: 是否有效
    """
    expected_dim = get_dimension(model_name)
    actual_dim = len(embedding)
    return expected_dim == actual_dim
