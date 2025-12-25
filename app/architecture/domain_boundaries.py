"""
领域边界定义

Domain Boundaries - 定义清晰的领域边界和服务接口
实施领域驱动设计(DDD)架构
"""

from enum import Enum
from typing import Dict, List, Type, Protocol, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class DomainType(str, Enum):
    """领域类型"""
    CORE = "core"                    # 核心领域 - RAG检索
    SUPPORTING = "supporting"        # 支撑领域 - 文档处理
    GENERIC = "generic"              # 通用领域 - 缓存、存储等
    INFRASTRUCTURE = "infrastructure"  # 基础设施 - 数据库、消息队列等

class BoundedContext:
    """限界上下文"""

    def __init__(self, name: str, domain_type: DomainType, description: str):
        self.name = name
        self.domain_type = domain_type
        self.description = description
        self.services: List[str] = []
        self.interfaces: Dict[str, Type] = {}
        self.dependencies: List[str] = []

    def add_service(self, service_name: str):
        """添加服务"""
        if service_name not in self.services:
            self.services.append(service_name)

    def add_interface(self, interface_name: str, interface_type: Type):
        """添加接口"""
        self.interfaces[interface_name] = interface_type

    def add_dependency(self, dependency: str):
        """添加依赖"""
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)

# 定义领域边界
DOMAIN_BOUNDARIES = {
    # 核心领域 - RAG检索
    "retrieval_domain": BoundedContext(
        name="检索域",
        domain_type=DomainType.CORE,
        description="负责智能检索、语义匹配和答案生成的核心业务逻辑"
    ),

    # 支撑领域 - 文档处理
    "document_domain": BoundedContext(
        name="文档域",
        domain_type=DomainType.SUPPORTING,
        description="负责文档解析、内容提取和结构化处理"
    ),

    # 支撑领域 - 知识图谱
    "knowledge_domain": BoundedContext(
        name="知识域",
        domain_type=DomainType.SUPPORTING,
        description="负责知识图谱构建、实体识别和关系推理"
    ),

    # 通用领域 - 存储服务
    "storage_domain": BoundedContext(
        name="存储域",
        domain_type=DomainType.GENERIC,
        description="负责数据存储、缓存管理和文件系统操作"
    ),

    # 通用领域 - 外部服务
    "external_domain": BoundedContext(
        name="外部域",
        domain_type=DomainType.GENERIC,
        description="负责与外部系统、API和第三方服务的集成"
    ),

    # 基础设施 - 系统服务
    "infrastructure_domain": BoundedContext(
        name="基础设施域",
        domain_type=DomainType.INFRASTRUCTURE,
        description="负责系统配置、监控、日志和部署等基础设施功能"
    )
}

# 定义领域间的依赖关系
DOMAIN_DEPENDENCIES = {
    "retrieval_domain": ["document_domain", "knowledge_domain", "storage_domain", "external_domain"],
    "document_domain": ["storage_domain", "external_domain"],
    "knowledge_domain": ["storage_domain", "external_domain"],
    "storage_domain": ["infrastructure_domain"],
    "external_domain": ["infrastructure_domain"],
    "infrastructure_domain": []
}

# 领域接口定义
class DomainInterface(Protocol):
    """领域接口协议"""

    def get_context_name(self) -> str:
        """获取上下文名称"""
        ...

    def get_services(self) -> List[str]:
        """获取服务列表"""
        ...

    def get_dependencies(self) -> List[str]:
        """获取依赖列表"""
        ...

# 核心领域接口
class RetrievalDomainInterface(DomainInterface):
    """检索域接口"""

    @abstractmethod
    async def query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """执行检索查询"""
        ...

    @abstractmethod
    async def stream_query(self, request: Dict[str, Any]) -> Any:
        """执行流式检索查询"""
        ...

    @abstractmethod
    def get_available_strategies(self) -> List[Dict[str, str]]:
        """获取可用策略"""
        ...

class DocumentDomainInterface(DomainInterface):
    """文档域接口"""

    @abstractmethod
    async def process_document(self, document: Any, content: bytes, mode: str) -> Dict[str, Any]:
        """处理文档"""
        ...

    @abstractmethod
    async def batch_process_documents(self, documents: List[Any], contents: List[bytes], mode: str) -> List[Dict[str, Any]]:
        """批量处理文档"""
        ...

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        ...

class KnowledgeDomainInterface(DomainInterface):
    """知识域接口"""

    @abstractmethod
    async def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """提取实体"""
        ...

    @abstractmethod
    async def build_relationships(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建关系"""
        ...

    @abstractmethod
    async def query_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        """查询知识图谱"""
        ...

class StorageDomainInterface(DomainInterface):
    """存储域接口"""

    @abstractmethod
    async def store(self, key: str, value: Any) -> bool:
        """存储数据"""
        ...

    @abstractmethod
    async def retrieve(self, key: str) -> Any:
        """检索数据"""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除数据"""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查数据是否存在"""
        ...

class ExternalDomainInterface(DomainInterface):
    """外部域接口"""

    @abstractmethod
    async def call_external_api(self, api_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用外部API"""
        ...

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        ...

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        ...

# 领域服务映射
DOMAIN_SERVICE_MAPPING = {
    "retrieval_domain": {
        "services": [
            "ConsolidatedRAGService",
        ],
        "interface": RetrievalDomainInterface,
        "module": "app.services.consolidated_rag_service"
    },

    "document_domain": {
        "services": [
            "ConsolidatedDocumentService",
        ],
        "interface": DocumentDomainInterface,
        "module": "app.services.consolidated_document_service"
    },

    "knowledge_domain": {
        "services": [
            "Neo4jService",
            "EntityExtractionService",
        ],
        "interface": KnowledgeDomainInterface,
        "module": "app.services.neo4j_service"
    },

    "storage_domain": {
        "services": [
            "MilvusService",
            "MinIOService",
            "CacheManager",
        ],
        "interface": StorageDomainInterface,
        "module": "app.services.milvus_service"
    },

    "external_domain": {
        "services": [
            "EmbeddingService",
            "LLMService",
        ],
        "interface": ExternalDomainInterface,
        "module": "app.services.embedding_service"
    }
}

class DomainBoundaryManager:
    """领域边界管理器"""

    def __init__(self):
        self.boundaries = DOMAIN_BOUNDARIES
        self.dependencies = DOMAIN_DEPENDENCIES
        self.service_mapping = DOMAIN_SERVICE_MAPPING
        self.interface_instances: Dict[str, DomainInterface] = {}

    def get_domain_boundary(self, domain_name: str) -> Optional[BoundedContext]:
        """获取领域边界"""
        return self.boundaries.get(domain_name)

    def get_domain_dependencies(self, domain_name: str) -> List[str]:
        """获取领域依赖"""
        return self.dependencies.get(domain_name, [])

    def get_domain_services(self, domain_name: str) -> List[str]:
        """获取领域服务"""
        mapping = self.service_mapping.get(domain_name, {})
        return mapping.get("services", [])

    def get_domain_interface(self, domain_name: str) -> Optional[Type]:
        """获取领域接口"""
        mapping = self.service_mapping.get(domain_name, {})
        return mapping.get("interface")

    def validate_domain_boundaries(self) -> List[str]:
        """验证领域边界"""
        errors = []

        # 检查循环依赖
        for domain_name, deps in self.dependencies.items():
            visited = set()

            def check_circular_dependency(current: str, path: List[str]) -> bool:
                if current in visited:
                    return True
                visited.add(current)

                for dep in self.dependencies.get(current, []):
                    if dep == domain_name and len(path) > 1:
                        return True
                    if dep in path and check_circular_dependency(dep, path + [current]):
                        return True
                return False

            if check_circular_dependency(domain_name, [domain_name]):
                errors.append(f"Circular dependency detected for domain: {domain_name}")

        # 检查未定义的依赖
        for domain_name, deps in self.dependencies.items():
            for dep in deps:
                if dep not in self.boundaries:
                    errors.append(f"Undefined dependency: {dep} for domain {domain_name}")

        return errors

    def get_dependency_graph(self) -> Dict[str, Any]:
        """获取依赖图"""
        graph = {
            "nodes": [],
            "edges": []
        }

        # 添加节点
        for domain_name, boundary in self.boundaries.items():
            graph["nodes"].append({
                "id": domain_name,
                "name": boundary.name,
                "type": boundary.domain_type.value,
                "description": boundary.description,
                "services": self.get_domain_services(domain_name)
            })

        # 添加边
        for domain_name, deps in self.dependencies.items():
            for dep in deps:
                graph["edges"].append({
                    "from": domain_name,
                    "to": dep
                })

        return graph

    def register_interface_instance(self, domain_name: str, instance: DomainInterface):
        """注册接口实例"""
        self.interface_instances[domain_name] = instance

    def get_interface_instance(self, domain_name: str) -> Optional[DomainInterface]:
        """获取接口实例"""
        return self.interface_instances.get(domain_name)

# 全局领域边界管理器
domain_boundary_manager = DomainBoundaryManager()

# 验证领域边界配置
validation_errors = domain_boundary_manager.validate_domain_boundaries()
if validation_errors:
    logger.warning("Domain boundary validation errors:")
    for error in validation_errors:
        logger.warning(f"  - {error}")
else:
    logger.info("Domain boundaries validated successfully")

# 导出主要类和实例
__all__ = [
    "DomainType",
    "BoundedContext",
    "DOMAIN_BOUNDARIES",
    "DOMAIN_DEPENDENCIES",
    "RetrievalDomainInterface",
    "DocumentDomainInterface",
    "KnowledgeDomainInterface",
    "StorageDomainInterface",
    "ExternalDomainInterface",
    "DomainBoundaryManager",
    "domain_boundary_manager"
]