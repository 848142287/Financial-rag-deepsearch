"""
服务去重映射配置

Service Deduplication Map - 定义新旧服务之间的映射关系
指导服务整合和弃用的策略
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum

class ServiceStatus(str, Enum):
    """服务状态"""
    ACTIVE = "active"              # 活跃，继续使用
    DEPRECATED = "deprecated"      # 已弃用，保留向后兼容
    REMOVED = "removed"            # 已移除，不再维护
    CONSOLIDATED = "consolidated"  # 已整合到新服务

class ServiceMapping:
    """服务映射信息"""
    def __init__(self,
                 old_service: str,
                 new_service: Optional[str] = None,
                 status: ServiceStatus = ServiceStatus.ACTIVE,
                 migration_path: Optional[str] = None,
                 deprecation_version: Optional[str] = None,
                 removal_version: Optional[str] = None):
        self.old_service = old_service
        self.new_service = new_service
        self.status = status
        self.migration_path = migration_path
        self.deprecation_version = deprecation_version
        self.removal_version = removal_version

# RAG服务去重映射
RAG_SERVICE_MAPPINGS: Dict[str, ServiceMapping] = {
    # 保留的整合服务
    "ConsolidatedRAGService": ServiceMapping(
        old_service="ConsolidatedRAGService",
        status=ServiceStatus.ACTIVE
    ),

    # 需要弃用的旧服务
    "AgenticRAGService": ServiceMapping(
        old_service="AgenticRAGService",
        new_service="ConsolidatedRAGService",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用ConsolidatedRAGService的agentic模式",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),

    "UnifiedRAGService": ServiceMapping(
        old_service="UnifiedRAGService",
        new_service="ConsolidatedRAGService",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用ConsolidatedRAGService的enhanced模式",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),

    "RAGService": ServiceMapping(
        old_service="RAGService",
        new_service="ConsolidatedRAGService",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用ConsolidatedRAGService的simple模式",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),
}

# 文档服务去重映射
DOCUMENT_SERVICE_MAPPINGS: Dict[str, ServiceMapping] = {
    # 保留的整合服务
    "ConsolidatedDocumentService": ServiceMapping(
        old_service="ConsolidatedDocumentService",
        status=ServiceStatus.ACTIVE
    ),

    # 需要弃用的旧服务
    "DocumentProcessor": ServiceMapping(
        old_service="DocumentProcessor",
        new_service="ConsolidatedDocumentService",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用ConsolidatedDocumentService",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),

    "DocumentParser": ServiceMapping(
        old_service="DocumentParser",
        new_service="ConsolidatedDocumentService",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用ConsolidatedDocumentService的解析功能",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),
}

# API端点去重映射
API_ENDPOINT_MAPPINGS: Dict[str, ServiceMapping] = {
    # RAG相关端点
    "/api/v1/rag/query": ServiceMapping(
        old_service="/api/v1/rag/query",
        status=ServiceStatus.ACTIVE
    ),

    "/query": ServiceMapping(
        old_service="/query",
        new_service="/api/v1/rag/query?mode=simple",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用/api/v1/rag/query并指定mode参数",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),

    "/enhanced-query": ServiceMapping(
        old_service="/enhanced-query",
        new_service="/api/v1/rag/query?mode=enhanced",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用/api/v1/rag/query?mode=enhanced",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),

    "/stream-query": ServiceMapping(
        old_service="/stream-query",
        new_service="/api/v1/rag/stream-query",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用/api/v1/rag/stream-query",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),

    # 文档处理端点
    "/api/v1/documents/upload": ServiceMapping(
        old_service="/api/v1/documents/upload",
        status=ServiceStatus.ACTIVE
    ),

    "/upload": ServiceMapping(
        old_service="/upload",
        new_service="/api/v1/documents/upload",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用/api/v1/documents/upload",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),

    "/documents/batch-upload": ServiceMapping(
        old_service="/documents/batch-upload",
        new_service="/api/v1/documents/batch-upload",
        status=ServiceStatus.DEPRECATED,
        migration_path="使用/api/v1/documents/batch-upload",
        deprecation_version="1.1.0",
        removal_version="2.0.0"
    ),
}

# 服务重构优先级
class RefactoringPriority(Enum):
    """重构优先级"""
    HIGH = "high"          # 高优先级，立即处理
    MEDIUM = "medium"      # 中等优先级，近期处理
    LOW = "low"           # 低优先级，长期规划

REFACTORING_PLAN: List[Tuple[RefactoringPriority, str, str]] = [
    # (优先级, 服务名称, 操作)
    (RefactoringPriority.HIGH, "AgenticRAGService", "移除重复实现，迁移到ConsolidatedRAGService"),
    (RefactoringPriority.HIGH, "UnifiedRAGService", "移除重复实现，迁移到ConsolidatedRAGService"),
    (RefactoringPriority.HIGH, "DocumentProcessor", "整合到ConsolidatedDocumentService"),
    (RefactoringPriority.MEDIUM, "EvaluationService", "整合多个评估服务"),
    (RefactoringPriority.MEDIUM, "EmbeddingService", "统一向量嵌入服务"),
    (RefactoringPriority.LOW, "CacheService", "整合缓存实现"),
]

# 需要移除的文件列表
FILES_TO_REMOVE: List[str] = [
    # RAG相关重复文件
    "backend/app/services/rag_service.py",
    "backend/app/services/agentic_rag_service.py",
    "backend/app/services/rag/unified_rag_service.py",

    # 文档处理重复文件
    "backend/app/services/document_processor.py",
    "backend/app/services/document_parser.py",

    # API端点重复文件
    "backend/app/api/endpoints/rag.py",
    "backend/app/api/endpoints/enhanced_rag.py",
    "backend/app/api/endpoints/unified_rag.py",
    "backend/app/api/endpoints/documents.py",
    "backend/app/api/endpoints/documents_batch.py",
]

# 需要更新的导入映射
IMPORT_REMAPPINGS: Dict[str, str] = {
    # 服务导入映射
    "from app.services.consolidated_rag_service import ConsolidatedRAGService as AgenticRAGService":
        "from app.services.consolidated_rag_service import ConsolidatedRAGService as AgenticRAGService",

    "from app.services.consolidated_rag_service import ConsolidatedRAGService as UnifiedRAGService":
        "from app.services.consolidated_rag_service import ConsolidatedRAGService as UnifiedRAGService",

    "from app.services.consolidated_document_service import ConsolidatedDocumentService as DocumentProcessor":
        "from app.services.consolidated_document_service import ConsolidatedDocumentService as DocumentProcessor",

    # API导入映射
    "from app.api.endpoints.consolidated_rag import router as rag_router":
        "from app.api.endpoints.consolidated_rag import router as rag_router",

    "from app.api.endpoints.consolidated_documents import router as documents_router":
        "from app.api.endpoints.consolidated_documents import router as documents_router",
}

def get_service_mapping(service_name: str) -> Optional[ServiceMapping]:
    """获取服务映射信息"""
    # 在所有映射中搜索
    all_mappings = {**RAG_SERVICE_MAPPINGS, **DOCUMENT_SERVICE_MAPPINGS, **API_ENDPOINT_MAPPINGS}
    return all_mappings.get(service_name)

def get_deprecated_services() -> List[ServiceMapping]:
    """获取所有已弃用的服务"""
    all_mappings = {**RAG_SERVICE_MAPPINGS, **DOCUMENT_SERVICE_MAPPINGS, **API_ENDPOINT_MAPPINGS}
    return [mapping for mapping in all_mappings.values() if mapping.status == ServiceStatus.DEPRECATED]

def get_refactoring_plan(priority: Optional[RefactoringPriority] = None) -> List[Tuple[RefactoringPriority, str, str]]:
    """获取重构计划"""
    if priority:
        return [(p, service, action) for p, service, action in REFACTORING_PLAN if p == priority]
    return REFACTORING_PLAN

def generate_migration_script() -> str:
    """生成迁移脚本"""
    script_lines = [
        "#!/usr/bin/env python3",
        "\"\"\"",
        "服务迁移脚本 - 自动更新代码中的服务引用",
        "根据service_deduplication_map.py中的映射关系自动更新导入语句",
        "\"\"\"",
        "",
        "import os",
        "import re",
        "from pathlib import Path",
        "",
        "# 导入映射配置",
        "from app.core.service_deduplication_map import IMPORT_REMAPPINGS, FILES_TO_REMOVE",
        "",
        "def update_imports(file_path: str) -> int:",
        "    \"\"\"更新文件中的导入语句\"\"\"",
        "    if not os.path.exists(file_path):",
        "        return 0",
        "    ",
        "    with open(file_path, 'r', encoding='utf-8') as f:",
        "        content = f.read()",
        "    ",
        "    original_content = content",
        "    changes = 0",
        "    ",
        "    for old_import, new_import in IMPORT_REMAPPINGS.items():",
        "        if old_import in content:",
        "            content = content.replace(old_import, new_import)",
        "            changes += 1",
        "    ",
        "    if changes > 0:",
        "        with open(file_path, 'w', encoding='utf-8') as f:",
        "            f.write(content)",
        "    ",
        "    return changes",
        "",
        "def main():",
        "    \"\"\"主函数\"\"\"",
        "    # 更新Python文件中的导入",
        "    python_files = list(Path('.').rglob('*.py'))",
        "    total_changes = 0",
        "    ",
        "    for file_path in python_files:",
        "        changes = update_imports(str(file_path))",
        "        if changes > 0:",
        "            print(f\"Updated {file_path}: {changes} changes\")",
        "            total_changes += changes",
        "    ",
        "    print(f\"\\nTotal changes: {total_changes}\")",
        "    ",
        "    # 提示需要手动删除的文件",
        "    if FILES_TO_REMOVE:",
        "        print(\"\\nFiles to manually remove:\")",
        "        for file_path in FILES_TO_REMOVE:",
        "            print(f\"  - {file_path}\")",
        "",
        "if __name__ == \"__main__\":",
        "    main()",
    ]

    return "\n".join(script_lines)

# 配置验证函数
def validate_mappings() -> List[str]:
    """验证映射配置的有效性"""
    errors = []

    # 检查循环引用
    for service_name, mapping in RAG_SERVICE_MAPPINGS.items():
        if mapping.new_service:
            new_mapping = RAG_SERVICE_MAPPINGS.get(mapping.new_service)
            if new_mapping and new_mapping.new_service == service_name:
                errors.append(f"Circular reference detected: {service_name} -> {mapping.new_service}")

    # 检查重复的新服务
    new_services = [mapping.new_service for mapping in RAG_SERVICE_MAPPINGS.values() if mapping.new_service]
    if len(new_services) != len(set(new_services)):
        errors.append("Duplicate new services detected in RAG mappings")

    return errors

# 生成迁移脚本
MIGRATION_SCRIPT = generate_migration_script()

if __name__ == "__main__":
    # 验证配置
    validation_errors = validate_mappings()
    if validation_errors:
        print("Validation errors found:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("Service mapping configuration is valid")

    # 保存迁移脚本
    with open("migrate_services.py", "w", encoding="utf-8") as f:
        f.write(MIGRATION_SCRIPT)
    print("Migration script generated: migrate_services.py")