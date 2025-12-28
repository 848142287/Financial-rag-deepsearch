"""
完整的图数据库客户端实现
替换 graph_sync_service.py 中的空实现
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
import json

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, ClientError

from app.core.graph_config import (
    GraphEntityType,
    GraphRelationType,
    graph_storage_config,
    generate_entity_id,
    generate_relation_id
)

logger = logging.getLogger(__name__)


class Neo4jClientError(Exception):
    """Neo4j客户端错误"""
    pass


class GraphDBClient:
    """
    完整的图数据库客户端实现
    替换 graph_sync_service.py 中的空实现
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化图数据库客户端

        Args:
            config: 配置字典（可选，默认使用全局配置）
        """
        self.config = config or {
            "uri": graph_storage_config.uri,
            "user": graph_storage_config.user,
            "password": graph_storage_config.password,
            "database": graph_storage_config.database
        }

        self.driver: Optional[Driver] = None
        self.is_connected = False

        logger.info(f"初始化 GraphDBClient: {self.config['uri']}")

    async def connect(self):
        """连接到图数据库"""
        try:
            logger.info(f"尝试连接 Neo4j: {self.config['uri']}")

            self.driver = GraphDatabase.driver(
                self.config["uri"],
                auth=(self.config["user"], self.config["password"]),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                connection_timeout=30
            )

            # 测试连接
            with self.driver.session(database=self.config["database"]) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record["test"] == 1:
                    self.is_connected = True
                    logger.info(f"成功连接到 Neo4j: {self.config['uri']}")
                else:
                    raise Neo4jClientError("连接测试失败")

        except ServiceUnavailable as e:
            logger.error(f"Neo4j 服务不可用: {e}")
            raise Neo4jClientError(f"无法连接到 Neo4j: {e}")
        except Exception as e:
            logger.error(f"连接 Neo4j 失败: {e}")
            raise Neo4jClientError(f"连接失败: {e}")

    async def disconnect(self):
        """断开图数据库连接"""
        try:
            if self.driver:
                self.driver.close()
                self.is_connected = False
                logger.info("已断开 Neo4j 连接")
        except Exception as e:
            logger.error(f"断开 Neo4j 连接失败: {e}")

    async def create_constraints(self):
        """创建约束和索引"""
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                # 创建唯一性约束
                constraints = [
                    # Document 节点
                    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
                    "FOR (d:Document) REQUIRE d.id IS UNIQUE",

                    # Chunk 节点
                    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                    "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",

                    # Entity 节点（基于标准化的实体ID）
                    "CREATE CONSTRAINT entity_canonical_id_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE",

                    # KnowledgeNode 节点
                    "CREATE CONSTRAINT knowledge_node_id_unique IF NOT EXISTS "
                    "FOR (kn:KnowledgeNode) REQUIRE kn.id IS UNIQUE",
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logger.info(f"约束创建成功: {constraint[:50]}...")
                    except ClientError as e:
                        if "EquivalentSchemaRuleAlreadyExists" in str(e):
                            logger.info(f"约束已存在: {constraint[:50]}...")
                        else:
                            logger.warning(f"约束创建失败: {e}")

                # 创建索引
                indexes = [
                    # Entity 名称索引
                    "CREATE INDEX entity_name_index IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.name)",

                    # Entity 类型索引
                    "CREATE INDEX entity_type_index IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.type)",

                    # Document 标题索引
                    "CREATE INDEX document_title_index IF NOT EXISTS "
                    "FOR (d:Document) ON (d.title)",

                    # Chunk 内容全文索引
                    "CREATE FULLTEXT INDEX chunk_content_fulltext IF NOT EXISTS "
                    "FOR (c:Chunk) ON EACH [c.content]",

                    # Entity 别名全文索引
                    "CREATE FULLTEXT INDEX entity_aliases_fulltext IF NOT EXISTS "
                    "FOR (e:Entity) ON EACH [e.aliases]",

                    # 关系类型索引
                    "CREATE INDEX relation_type_index IF NOT EXISTS "
                    "FOR ()-[r:MENTIONED_IN]-() ON (type(r))",
                ]

                for index in indexes:
                    try:
                        session.run(index)
                        logger.info(f"索引创建成功: {index[:50]}...")
                    except ClientError as e:
                        if "EquivalentSchemaRuleAlreadyExists" in str(e):
                            logger.info(f"索引已存在: {index[:50]}...")
                        else:
                            logger.warning(f"索引创建失败: {e}")

                logger.info("Neo4j 约束和索引初始化完成")

        except Exception as e:
            logger.error(f"创建约束和索引失败: {e}")
            raise Neo4jClientError(f"初始化约束失败: {e}")

    async def create_node(
        self,
        node_type: str,
        properties: Dict
    ) -> str:
        """
        创建节点

        Args:
            node_type: 节点类型
            properties: 节点属性

        Returns:
            节点ID
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                # 根据节点类型选择ID字段
                if node_type == "Entity":
                    id_field = "canonical_id"
                    if id_field not in properties:
                        # 使用标准化ID
                        entity_name = properties.get("name", "")
                        entity_type = properties.get("type", "UNKNOWN")
                        properties[id_field] = generate_entity_id(
                            entity_name,
                            GraphEntityType(entity_type)
                        )
                else:
                    id_field = "id"
                    if id_field not in properties:
                        properties[id_field] = str(uuid.uuid4())

                # 添加时间戳
                properties["created_at"] = datetime.utcnow().isoformat()
                properties["updated_at"] = datetime.utcnow().isoformat()

                # 创建节点
                query = f"""
                MERGE (n:{node_type} {{{id_field}: $id}})
                ON CREATE SET n = $properties
                ON MATCH SET n += $properties
                RETURN n.{id_field} as id
                """

                result = session.run(query, {
                    "id": properties[id_field],
                    "properties": properties
                })

                record = result.single()
                if record:
                    node_id = record["id"]
                    logger.debug(f"创建节点: {node_type}#{node_id}")
                    return node_id
                else:
                    raise Neo4jClientError("创建节点失败：未返回ID")

        except Exception as e:
            logger.error(f"创建节点失败: {e}")
            raise Neo4jClientError(f"创建节点失败: {e}")

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Dict
    ) -> str:
        """
        创建关系

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            rel_type: 关系类型
            properties: 关系属性

        Returns:
            关系ID
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                # 生成关系ID
                if "id" not in properties:
                    properties["id"] = generate_relation_id(
                        source_id,
                        target_id,
                        GraphRelationType(rel_type)
                    )

                # 添加时间戳
                properties["created_at"] = datetime.utcnow().isoformat()
                properties["updated_at"] = datetime.utcnow().isoformat()

                # 创建关系
                query = f"""
                MATCH (source {{canonical_id: $source_id}})
                MATCH (target {{canonical_id: $target_id}})
                MERGE (source)-[r:{rel_type} {{id: $rel_id}}]->(target)
                ON CREATE SET r = $properties
                ON MATCH SET r += $properties
                RETURN r.id as id
                """

                result = session.run(query, {
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_id": properties["id"],
                    "properties": properties
                })

                record = result.single()
                if record:
                    rel_id = record["id"]
                    logger.debug(f"创建关系: {rel_type}#{rel_id}")
                    return rel_id
                else:
                    raise Neo4jClientError("创建关系失败：未返回ID")

        except Exception as e:
            logger.error(f"创建关系失败: {e}")
            raise Neo4jClientError(f"创建关系失败: {e}")

    async def update_node(self, node_id: str, properties: Dict):
        """
        更新节点属性

        Args:
            node_id: 节点ID
            properties: 要更新的属性
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                properties["updated_at"] = datetime.utcnow().isoformat()

                query = """
                MATCH (n {canonical_id: $node_id})
                SET n += $properties
                RETURN count(n) as updated_count
                """

                result = session.run(query, {
                    "node_id": node_id,
                    "properties": properties
                })

                record = result.single()
                if record["updated_count"] == 0:
                    logger.warning(f"节点未找到: {node_id}")
                else:
                    logger.debug(f"更新节点: {node_id}")

        except Exception as e:
            logger.error(f"更新节点失败: {e}")
            raise Neo4jClientError(f"更新节点失败: {e}")

    async def update_relationship(self, rel_id: str, properties: Dict):
        """
        更新关系属性

        Args:
            rel_id: 关系ID
            properties: 要更新的属性
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                properties["updated_at"] = datetime.utcnow().isoformat()

                query = """
                MATCH ()-[r {id: $rel_id}]->()
                SET r += $properties
                RETURN count(r) as updated_count
                """

                result = session.run(query, {
                    "rel_id": rel_id,
                    "properties": properties
                })

                record = result.single()
                if record["updated_count"] == 0:
                    logger.warning(f"关系未找到: {rel_id}")
                else:
                    logger.debug(f"更新关系: {rel_id}")

        except Exception as e:
            logger.error(f"更新关系失败: {e}")
            raise Neo4jClientError(f"更新关系失败: {e}")

    async def delete_node(self, node_id: str):
        """
        删除节点及其所有关系

        Args:
            node_id: 节点ID
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                query = """
                MATCH (n {canonical_id: $node_id})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """

                result = session.run(query, {"node_id": node_id})
                record = result.single()

                if record["deleted_count"] > 0:
                    logger.info(f"删除节点: {node_id}")
                else:
                    logger.warning(f"节点未找到: {node_id}")

        except Exception as e:
            logger.error(f"删除节点失败: {e}")
            raise Neo4jClientError(f"删除节点失败: {e}")

    async def delete_relationship(self, rel_id: str):
        """
        删除关系

        Args:
            rel_id: 关系ID
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                query = """
                MATCH ()-[r {id: $rel_id}]->()
                DELETE r
                RETURN count(r) as deleted_count
                """

                result = session.run(query, {"rel_id": rel_id})
                record = result.single()

                if record["deleted_count"] > 0:
                    logger.info(f"删除关系: {rel_id}")
                else:
                    logger.warning(f"关系未找到: {rel_id}")

        except Exception as e:
            logger.error(f"删除关系失败: {e}")
            raise Neo4jClientError(f"删除关系失败: {e}")

    async def query_nodes(
        self,
        node_type: str,
        properties: Dict
    ) -> List[Dict]:
        """
        查询节点

        Args:
            node_type: 节点类型
            properties: 属性过滤条件

        Returns:
            节点列表
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                # 构建WHERE条件
                conditions = []
                for key, value in properties.items():
                    if isinstance(value, str):
                        conditions.append(f"n.{key} CONTAINS '${value}'")
                    else:
                        conditions.append(f"n.{key} = ${key}")

                where_clause = " AND ".join(conditions) if conditions else "true"

                query = f"""
                MATCH (n:{node_type})
                WHERE {where_clause}
                RETURN n
                LIMIT 100
                """

                result = session.run(query, properties)
                nodes = []
                for record in result:
                    node = record["n"]
                    nodes.append(dict(node))
                    if len(nodes) >= 100:
                        break

                return nodes

        except Exception as e:
            logger.error(f"查询节点失败: {e}")
            raise Neo4jClientError(f"查询节点失败: {e}")

    async def query_relationships(
        self,
        rel_type: Optional[str] = None,
        properties: Dict = None
    ) -> List[Dict]:
        """
        查询关系

        Args:
            rel_type: 关系类型（可选）
            properties: 属性过滤条件（可选）

        Returns:
            关系列表
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                # 构建查询
                if rel_type:
                    pattern = f"[r:{rel_type}]"
                else:
                    pattern = "[r]"

                # 构建WHERE条件
                conditions = []
                if properties:
                    for key, value in properties.items():
                        if isinstance(value, str):
                            conditions.append(f"r.{key} CONTAINS '${value}'")
                        else:
                            conditions.append(f"r.{key} = ${key}")

                where_clause = " AND ".join(conditions) if conditions else "true"

                query = f"""
                MATCH (source)-{pattern}->(target)
                WHERE {where_clause}
                RETURN source, r, target
                LIMIT 100
                """

                result = session.run(query, properties or {})
                relations = []
                for record in result:
                    relations.append({
                        "source": dict(record["source"]),
                        "relation": dict(record["r"]),
                        "target": dict(record["target"])
                    })
                    if len(relations) >= 100:
                        break

                return relations

        except Exception as e:
            logger.error(f"查询关系失败: {e}")
            raise Neo4jClientError(f"查询关系失败: {e}")

    async def find_entity_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        根据名称查找实体

        Args:
            name: 实体名称
            entity_type: 实体类型（可选）

        Returns:
            实体字典，如果未找到则返回None
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                if entity_type:
                    query = """
                    MATCH (e:Entity {type: $entity_type})
                    WHERE e.name = $name OR $name IN e.aliases
                    RETURN e
                    LIMIT 1
                    """
                    result = session.run(query, {
                        "name": name,
                        "entity_type": entity_type
                    })
                else:
                    query = """
                    MATCH (e:Entity)
                    WHERE e.name = $name OR $name IN e.aliases
                    RETURN e
                    LIMIT 1
                    """
                    result = session.run(query, {"name": name})

                record = result.single()
                if record:
                    entity = dict(record["e"])
                    return entity
                else:
                    return None

        except Exception as e:
            logger.error(f"查找实体失败: {e}")
            raise Neo4jClientError(f"查找实体失败: {e}")

    async def batch_create_nodes(
        self,
        nodes: List[Dict[str, Any]]
    ) -> List[str]:
        """
        批量创建节点

        Args:
            nodes: 节点列表，每个节点包含 node_type 和 properties

        Returns:
            节点ID列表
        """
        if not nodes:
            return []

        if not self.is_connected:
            await self.connect()

        node_ids = []

        try:
            with self.driver.session(database=self.config["database"]) as session:
                for node_data in nodes:
                    node_type = node_data["node_type"]
                    properties = node_data["properties"]

                    # 生成ID
                    if node_type == "Entity":
                        id_field = "canonical_id"
                        if id_field not in properties:
                            entity_name = properties.get("name", "")
                            entity_type = properties.get("type", "UNKNOWN")
                            properties[id_field] = generate_entity_id(
                                entity_name,
                                GraphEntityType(entity_type)
                            )
                    else:
                        id_field = "id"
                        if id_field not in properties:
                            properties[id_field] = str(uuid.uuid4())

                    # 添加时间戳
                    properties["created_at"] = datetime.utcnow().isoformat()
                    properties["updated_at"] = datetime.utcnow().isoformat()

                    query = f"""
                    MERGE (n:{node_type} {{{id_field}: $id}})
                    ON CREATE SET n = $properties
                    ON MATCH SET n += $properties
                    RETURN n.{id_field} as id
                    """

                    result = session.run(query, {
                        "id": properties[id_field],
                        "properties": properties
                    })

                    record = result.single()
                    if record:
                        node_ids.append(record["id"])

                logger.info(f"批量创建 {len(node_ids)} 个节点")
                return node_ids

        except Exception as e:
            logger.error(f"批量创建节点失败: {e}")
            raise Neo4jClientError(f"批量创建节点失败: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息

        Returns:
            统计信息字典
        """
        if not self.is_connected:
            await self.connect()

        try:
            with self.driver.session(database=self.config["database"]) as session:
                # 节点统计
                node_stats = {}
                for label in ["Document", "Chunk", "Entity", "KnowledgeNode"]:
                    result = session.run(
                        f"MATCH (n:{label}) RETURN count(n) as count"
                    )
                    count = result.single()["count"]
                    node_stats[label] = count

                # 关系统计
                rel_stats = {}
                result = session.run(
                    "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count"
                )
                for record in result:
                    rel_stats[record["rel_type"]] = record["count"]

                # 总计
                result = session.run(
                    "MATCH (n) RETURN count(n) as total_nodes"
                )
                total_nodes = result.single()["total_nodes"]

                result = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as total_rels"
                )
                total_rels = result.single()["total_rels"]

                return {
                    "nodes": node_stats,
                    "relationships": rel_stats,
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels
                }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
