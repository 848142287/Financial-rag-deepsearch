#!/usr/bin/env python3
"""
模型配置验证工具
验证本地模型路径和环境变量配置
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

def validate_local_model_paths():
    """验证本地模型路径"""
    print("=" * 70)
    print("🔍 本地模型路径验证")
    print("=" * 70)
    print()

    models_to_check = [
        {
            "name": "bge-large-zh-v1.5",
            "path": settings.bge_embedding_model_path,
            "type": "嵌入模型",
            "required": settings.enable_local_embedding
        },
        {
            "name": "bge-reranker-v2-m3",
            "path": settings.bge_reranker_model_path,
            "type": "排序模型",
            "required": settings.enable_local_reranker
        }
        # OCR模型已改为使用GLM-4.6V云端API，无需本地验证
    ]

    available_count = 0
    total_count = len(models_to_check)

    for model in models_to_check:
        model_path = Path(model["path"])
        exists = model_path.exists()

        status = "✅ 可用" if exists else "❌ 不可用"
        requirement = "必须" if model["required"] else "可选"

        print(f"{model['name']} ({model['type']})")
        print(f"  路径: {model['path']}")
        print(f"  状态: {status}")
        print(f"  要求: {requirement}")

        if exists:
            print(f"  大小: {sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / 1024 / 1024:.2f} MB")
            available_count += 1
        else:
            if model["required"]:
                print(f"  ⚠️  警告: 本地模型不可用，将降级到API服务")
            else:
                print(f"  ℹ️  信息: 将使用API服务")

        print()

    print("-" * 70)
    print(f"本地模型可用率: {available_count}/{total_count} ({available_count/total_count*100:.1f}%)")
    print()

    return available_count

def validate_env_vars():
    """验证环境变量"""
    print("=" * 70)
    print("🔑 环境变量验证")
    print("=" * 70)
    print()

    validation_result = validate_env_vars()

    required_vars = {k: v for k, v in validation_result.items() if v["required"]}
    optional_vars = {k: v for k, v in validation_result.items() if not v["required"]}

    print("必需环境变量:")
    for var_name, var_info in required_vars.items():
        status = "✅ 已设置" if var_info["is_set"] else "❌ 未设置"
        print(f"  {var_name}: {status}")

    print()
    print("可选环境变量:")
    for var_name, var_info in optional_vars.items():
        status = "✅ 已设置" if var_info["is_set"] else "⚠️  未设置 (使用默认值)"
        print(f"  {var_name}: {status}")
        if not var_info["is_set"] and var_info.get("fallback"):
            print(f"    默认值: {var_info['fallback']}")

    print()
    required_set = sum(1 for v in required_vars.values() if v["is_set"])
    required_total = len(required_vars)

    if required_set == required_total:
        print("✅ 所有必需环境变量已设置")
    else:
        print(f"⚠️  {required_total - required_set} 个必需环境变量未设置")

    print()

    return required_set == required_total

def validate_model_strategy():
    """验证模型策略配置"""
    print("=" * 70)
    print("🎯 模型策略配置")
    print("=" * 70)
    print()

    # 嵌入模型策略
    print("1️⃣  嵌入模型策略:")
    print(f"   主模型: {ModelStrategy.EMBEDDING.name} ({ModelStrategy.EMBEDDING.provider.value})")
    print(f"   降级模型: {ModelStrategy.EMBEDDING.fallback_model}")
    print(f"   本地启用: {settings.enable_local_embedding}")
    print(f"   API降级: {settings.enable_api_fallback}")
    print()

    # 排序模型策略
    print("2️⃣  排序模型策略:")
    print(f"   主模型: {ModelStrategy.RERANKER.name} ({ModelStrategy.RERANKER.provider.value})")
    print(f"   降级模型: {ModelStrategy.RERANKER.fallback_model}")
    print(f"   本地启用: {settings.enable_local_reranker}")
    print(f"   API降级: {settings.enable_api_fallback}")
    print()

    # OCR模型策略
    print("3️⃣  OCR模型策略:")
    print(f"   主模型: {ModelStrategy.OCR_PRIMARY.name} ({ModelStrategy.OCR_PRIMARY.provider.value})")
    print(f"   降级模型: {ModelStrategy.OCR_BACKUP.name} ({ModelStrategy.OCR_BACKUP.provider.value})")
    print(f"   本地启用: {settings.enable_local_ocr}")
    print(f"   API降级: {settings.enable_api_fallback}")
    print()

    # 多模态LLM
    print("4️⃣  多模态LLM:")
    print(f"   模型: {ModelStrategy.MULTIMODAL_LLM.name} ({ModelStrategy.MULTIMODAL_LLM.provider.value})")
    print(f"   API: {ModelStrategy.MULTIMODAL_LLM.base_url}")
    print()

    # 检索LLM
    print("5️⃣  检索LLM:")
    print(f"   模型: {ModelStrategy.CHAT_LLM.name} ({ModelStrategy.CHAT_LLM.provider.value})")
    print(f"   API: {ModelStrategy.CHAT_LLM.base_url}")
    print(f"   最大长度: {ModelStrategy.CHAT_LLM.max_length} tokens")
    print()

def print_summary():
    """打印配置摘要"""
    print("=" * 70)
    print("📊 配置摘要")
    print("=" * 70)
    print()

    print("模型优先级策略:")
    print("  1. 嵌入模型: 本地 BGE → Qwen API")
    print("  2. 排序模型: 本地 BGE → Qwen API")
    print("  3. OCR模型: 本地 DeepSeek-OCR → Qwen-VL-OCR API")
    print("  4. 多模态LLM: Qwen-VL-Plus API")
    print("  5. 检索LLM: DeepSeek-Chat API")
    print()

    print("降级策略:")
    if settings.enable_api_fallback:
        print("  ✅ 已启用 API 降级（本地模型不可用时自动切换到API）")
    else:
        print("  ⚠️  API 降级已禁用（仅使用本地模型）")

    print()

    print("下一步操作:")
    if settings.enable_local_embedding:
        bge_path = Path(settings.bge_embedding_model_path)
        if not bge_path.exists():
            print("  1. 下载 BGE 嵌入模型:")
            print(f"     mkdir -p {bge_path.parent}")
            print(f"     # 从 https://huggingface.co/BAAI/bge-large-zh-v1.5 下载")

    if settings.enable_local_reranker:
        reranker_path = Path(settings.bge_reranker_model_path)
        if not reranker_path.exists():
            print("  2. 下载 BGE 排序模型:")
            print(f"     mkdir -p {reranker_path.parent}")
            print(f"     # 从 https://huggingface.co/BAAI/bge-reranker-v2-m3 下载")

    # OCR已使用GLM-4.6V云端API，无需下载本地模型

    print("  3. 配置环境变量:")
    print("     GLM_API_KEY=your_glm_api_key  # 用于GLM-4.7和GLM-4.6V")
    print("     QWEN_API_KEY=your_qwen_api_key  # 用于Qwen API（可选，作为备份）")
    print()

    print("  5. 测试模型加载:")
    print("     python -m app.services.models.model_loader")
    print()

def main():
    """主函数"""
    print()
    print("🚀 模型配置验证工具")
    print()

    # 1. 验证本地模型路径
    available_count = validate_local_model_paths()

    # 2. 验证环境变量
    env_valid = validate_env_vars()

    # 3. 验证模型策略
    validate_model_strategy()

    # 4. 打印摘要
    print_summary()

    # 返回状态
    if available_count > 0 or env_valid:
        print("=" * 70)
        print("✅ 验证完成")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("⚠️  警告: 没有可用的模型配置")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
