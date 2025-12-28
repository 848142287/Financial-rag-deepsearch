"""
知识图谱Tool - 使用LangChain 1.0+ Tool接口封装
"""

import asyncio
from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from app.services.knowledge.neo4j_service import Neo4jService
from app.core.config import get_settings
from app.core.logging import logger

class KnowledgeGraphInput(BaseModel):
    """知识图谱操作输入参数"""
    operation: str = Field(
        description="操作类型: query, add_entity, add_relation, find_path, get_neighbors"
    )
    query: Optional[str] = Field(default=None, description="图查询语句")
    entity_name: Optional[str] = Field(default=None, description="实体名称")
    entity_type: Optional[str] = Field(default=None, description="实体类型")
    relation: Optional[str] = Field(default=None, description="关系类型")
    source_entity: Optional[str] = Field(default=None, description="源实体")
    target_entity: Optional[str] = Field(default=None, description="目标实体")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="属性信息")

class KnowledgeGraphTool(BaseTool):
    """
    知识图谱Tool

    基于LangChain Tool接口封装的知识图谱操作功能
    """
    name: str = "knowledge_graph"
    description: str = "操作知识图谱，支持查询实体关系、添加实体关系、查找路径等操作"
    args_schema: Type[BaseModel] = KnowledgeGraphInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        self.neo4j_service = None
        self._init_services()

    def _init_services(self):
        """初始化服务组件"""
        try:
            self.neo4j_service = Neo4jService()
        except Exception as e:
            logger.error(f"Failed to initialize Neo4jService: {e}")

    def _run(
        self,
        operation: str,
        query: Optional[str] = None,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation: Optional[str] = None,
        source_entity: Optional[str] = None,
        target_entity: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        同步执行知识图谱操作

        Args:
            operation: 操作类型
            query: 图查询语句
            entity_name: 实体名称
            entity_type: 实体类型
            relation: 关系类型
            source_entity: 源实体
            target_entity: 目标实体
            properties: 属性信息

        Returns:
            操作结果字典
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._arun(
                    operation=operation,
                    query=query,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    relation=relation,
                    source_entity=source_entity,
                    target_entity=target_entity,
                    properties=properties
                )
            )
        finally:
            loop.close()

    async def _arun(
        self,
        operation: str,
        query: Optional[str] = None,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation: Optional[str] = None,
        source_entity: Optional[str] = None,
        target_entity: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        异步执行知识图谱操作

        Args:
            operation: 操作类型
            query: 图查询语句
            entity_name: 实体名称
            entity_type: 实体类型
            relation: 关系类型
            source_entity: 源实体
            target_entity: 目标实体
            properties: 属性信息

        Returns:
            操作结果字典
        """
        if not self.neo4j_service:
            return {
                "success": False,
                "error": "Neo4j service not initialized",
                "result": {}
            }

        try:
            # 根据操作类型调用相应功能
            if operation == "query":
                result = await self._query_knowledge_graph(query)
            elif operation == "add_entity":
                result = await self._add_entity(entity_name, entity_type, properties)
            elif operation == "add_relation":
                result = await self._add_relation(source_entity, target_entity, relation, properties)
            elif operation == "find_path":
                result = await self._find_path(source_entity, target_entity)
            elif operation == "get_neighbors":
                result = await self._get_neighbors(entity_name, relation)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }

            result["operation"] = operation
            return result

        except Exception as e:
            logger.error(f"Knowledge graph operation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": operation,
                "result": {}
            }

    async def _query_knowledge_graph(self, query: Optional[str]) -> Dict[str, Any]:
        """查询知识图谱"""
        if not query:
            return {"success": False, "error": "Query cannot be empty"}

        result = await self.neo4j_service.execute_query(query)
        return {
            "success": True,
            "result": result,
            "query": query,
            "result_count": len(result) if isinstance(result, list) else 1
        }

    async def _add_entity(
        self,
        entity_name: Optional[str],
        entity_type: Optional[str],
        properties: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """添加实体"""
        if not entity_name:
            return {"success": False, "error": "Entity name cannot be empty"}

        result = await self.neo4j_service.create_entity(
            name=entity_name,
            entity_type=entity_type or "DEFAULT",
            properties=properties or {}
        )
        return {
            "success": True,
            "result": {"entity_id": result, "name": entity_name, "type": entity_type}
        }

    async def _add_relation(
        self,
        source_entity: Optional[str],
        target_entity: Optional[str],
        relation: Optional[str],
        properties: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """添加关系"""
        if not source_entity or not target_entity:
            return {"success": False, "error": "Source and target entity cannot be empty"}

        result = await self.neo4j_service.create_relation(
            source_name=source_entity,
            target_name=target_entity,
            relation_type=relation or "RELATED_TO",
            properties=properties or {}
        )
        return {
            "success": True,
            "result": {"relation_id": result, "source": source_entity, "target": target_entity, "relation": relation}
        }

    async def _find_path(
        self,
        source_entity: Optional[str],
        target_entity: Optional[str]
    ) -> Dict[str, Any]:
        """查找实体间路径"""
        if not source_entity or not target_entity:
            return {"success": False, "error": "Source and target entity cannot be empty"}

        result = await self.neo4j_service.find_shortest_path(
            source_name=source_entity,
            target_name=target_entity
        )
        return {
            "success": True,
            "result": {"paths": result, "path_count": len(result)}
        }

    async def _get_neighbors(
        self,
        entity_name: Optional[str],
        relation: Optional[str]
    ) -> Dict[str, Any]:
        """获取实体邻居"""
        if not entity_name:
            return {"success": False, "error": "Entity name cannot be empty"}

        result = await self.neo4j_service.get_neighbors(
            entity_name=entity_name,
            relation_type=relation
        )
        return {
            "success": True,
            "result": {"neighbors": result, "neighbor_count": len(result)}
        }

    def get_tool_description(self) -> str:
        """获取工具详细描述"""
        return """
        知识图谱操作工具，支持多种图操作：

        1. query - 执行Cypher查询语句
        2. add_entity - 添加新实体节点
        3. add_relation - 添加实体间关系
        4. find_path - 查找实体间最短路径
        5. get_neighbors - 获取实体的邻居节点

        可以用于构建和查询知识图谱，支持实体关系挖掘、路径发现等高级功能。
        适用于金融领域的实体关系分析和知识推理。
        """