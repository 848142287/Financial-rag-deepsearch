"""
本地模型配置
定义本地模型的优先级和降级策略
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ModelProvider(Enum):
    """模型提供者"""
    LOCAL = "local"           # 本地模型
    QWEN_API = "qwen_api"     # Qwen API
    DEEPSEEK_API = "deepseek_api"  # DeepSeek API
    OPENAI_API = "openai_api"  # OpenAI API (兼容)


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: ModelProvider
    fallback_model: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    device: str = "cpu"  # cpu, cuda, mps
    max_length: int = 512
    batch_size: int = 32


class ModelStrategy:
    """
    模型策略配置
    定义本地模型优先级和降级策略
    """

    # ==================== 嵌入模型配置 ====================
    EMBEDDING = ModelConfig(
        name="bge-large-zh-v1.5",
        provider=ModelProvider.LOCAL,
        fallback_model="text-embedding-v4",  # Qwen API 作为备份
        device="cpu",
        max_length=512,
        batch_size=32
    )

    # ==================== 排序模型配置 ====================
    RERANKER = ModelConfig(
        name="bge-reranker-v2-m3",
        provider=ModelProvider.LOCAL,
        fallback_model="qwen3-rerank",  # Qwen API 作为备份
        device="cpu",
        max_length=512,
        batch_size=16
    )

    # ==================== OCR模型配置 ====================
    # GLM-4.6V云端OCR（主要模型）
    OCR_PRIMARY = ModelConfig(
        name="glm-4.6v",
        provider=ModelProvider.OPENAI_API,  # GLM使用OpenAI兼容API
        api_key_env="GLM_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        max_length=4000,  # OCR需要更多tokens
        fallback_model=None
    )

    # ==================== 多模态LLM配置 ====================
    MULTIMODAL_LLM = ModelConfig(
        name="qwen-vl-plus",
        provider=ModelProvider.QWEN_API,
        api_key_env="QWEN_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        device="cpu",
        max_length=2048
    )

    # ==================== 检索和对话LLM配置 ====================
    CHAT_LLM = ModelConfig(
        name="deepseek-chat",
        provider=ModelProvider.DEEPSEEK_API,
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        max_length=32000,  # DeepSeek 支持 32K 上下文
        temperature=0.3
    )

    # ==================== 嵌入模型选项（按优先级）====================
    EMBEDDING_MODELS = [
        {
            "name": "bge-large-zh-v1.5",
            "provider": "local",
            "path": "models/BAAI/bge-large-zh-v1.5",
            "dimension": 1024,
            "max_length": 512,
            "priority": 1  # 最高优先级
        },
        {
            "name": "text-embedding-v4",
            "provider": "qwen_api",
            "dimension": 1536,
            "max_length": 8192,
            "priority": 2  # 备份
        }
    ]

    # ==================== 排序模型选项（按优先级）====================
    RERANKER_MODELS = [
        {
            "name": "bge-reranker-v2-m3",
            "provider": "local",
            "path": "models/BAAI/bge-reranker-v2-m3",
            "dimension": 1024,
            "max_length": 512,
            "priority": 1  # 最高优先级
        },
        {
            "name": "qwen3-rerank",
            "provider": "qwen_api",
            "dimension": 1024,
            "max_length": 512,
            "priority": 2  # 备份
        }
    ]

    # ==================== OCR模型选项（按优先级）====================
    OCR_MODELS = [
        {
            "name": "glm-4.6v",
            "provider": "glm_api",
            "type": "api",
            "multimodal": True,
            "priority": 1  # 主OCR模型
        }
    ]


# ==================== 模型路径配置 ====================
LOCAL_MODEL_PATHS = {
    # BGE 嵌入模型
    "bge-large-zh-v1.5": "models/BAAI/bge-large-zh-v1.5",

    # BGE 排序模型
    "bge-reranker-v2-m3": "models/BAAI/bge-reranker-v2-m3",

    # GLM-4.6V云端OCR（已替换本地DeepSeek-OCR）
    # 本地OCR模型已移除，统一使用GLM-4.6V云端API

    # 其他本地模型路径
    "bge-small-zh-v1.5": "models/BAAI/bge-small-zh-v1.5",
    "bge-base-zh-v1.5": "models/BAAI/bge-base-zh-v1.5",
}


def get_embedding_model_config() -> ModelConfig:
    """获取嵌入模型配置"""
    return ModelStrategy.EMBEDDING


def get_reranker_model_config() -> ModelConfig:
    """获取排序模型配置"""
    return ModelStrategy.RERANKER


def get_ocr_models() -> List[ModelConfig]:
    """
    获取 OCR 模型列表
    返回: [GLM-4.6V]
    """
    return [
        ModelStrategy.OCR_PRIMARY
    ]


def get_multimodal_llm_config() -> ModelConfig:
    """获取多模态 LLM 配置"""
    return ModelStrategy.MULTIMODAL_LLM


def get_chat_llm_config() -> ModelConfig:
    """获取对话 LLM 配置"""
    return ModelStrategy.CHAT_LLM


def get_local_model_path(model_name: str) -> Optional[str]:
    """
    获取本地模型路径

    Args:
        model_name: 模型名称

    Returns:
        模型路径，如果不存在返回 None
    """
    return LOCAL_MODEL_PATHS.get(model_name)


def is_local_model_available(model_name: str) -> bool:
    """
    检查本地模型是否可用

    Args:
        model_name: 模型名称

    Returns:
        是否可用
    """
    import os
    from pathlib import Path

    path = get_local_model_path(model_name)
    if path is None:
        return False

    model_path = Path(path)
    return model_path.exists()


# ==================== 环境变量配置 ====================
REQUIRED_ENV_VARS = {
    # DeepSeek API
    "DEEPSEEK_API_KEY": {
        "description": "DeepSeek API Key",
        "required": True,
        "fallback": None
    },

    # Qwen API（用于 OCR、多模态、备份嵌入/排序）
    "QWEN_API_KEY": {
        "description": "Qwen API Key",
        "required": True,
        "fallback": None
    },

    # 模型路径
    "BGE_MODEL_PATH": {
        "description": "BGE 模型存储路径",
        "required": False,
        "fallback": "models/BAAI"
    },

    # GLM-4.6V云端OCR配置（已替换本地DeepSeek-OCR）
    "GLM_API_KEY": {
        "description": "GLM API Key (用于GLM-4.6V OCR和多模态)",
        "required": True,
        "fallback": None
    },
}


def validate_env_vars() -> dict:
    """
    验证环境变量配置

    Returns:
        验证结果 {env_var: (is_set, is_valid)}
    """
    import os
    result = {}

    for env_var, config in REQUIRED_ENV_VARS.items():
        is_set = os.getenv(env_var) is not None
        is_valid = is_set  # 简化验证

        result[env_var] = {
            "is_set": is_set,
            "is_valid": is_valid,
            "required": config["required"]
        }

    return result
