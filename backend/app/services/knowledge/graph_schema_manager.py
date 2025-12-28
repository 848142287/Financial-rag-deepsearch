"""
Neo4j 图谱Schema管理器
管理索引、约束和schema版本
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from app.core.graph_config import GraphEntityType, GraphRelationType

logger = logging.getLogger(__name__)


@dataclass
class IndexDefinition:
    """索引定义"""
    name: str
    label: str
    properties: List[str]
    index_type: str  # BTREE, FULLTEXT, LOOKUP
    options: Dict = None


@dataclass
class ConstraintDefinition:
    """约束定义"""
    name: str
    label: str
    property: str
    constraint_type: str  # UNIQUE, EXISTS, NODE_KEY


class GraphSchemaManager:
    """图谱Schema管理器"""

    def __init__(self):
        self.schema_version = "1.0.0"
        self.constraints = self._get_constraints()
        self.indexes = self._get_indexes()

    def _get_constraints(self) -> List[ConstraintDefinition]:
        """获取约束定义"""
        return [
            # Document 节点约束
            ConstraintDefinition(
                name="document_id_unique",
                label="Document",
                property="id",
                constraint_type="UNIQUE"
            ),

            # Chunk 节点约束
            ConstraintDefinition(
                name="chunk_id_unique",
                label="Chunk",
                property="id",
                constraint_type="UNIQUE"
            ),

            # Entity 节点约束（使用canonical_id）
            ConstraintDefinition(
                name="entity_canonical_id_unique",
                label="Entity",
                property="canonical_id",
                constraint_type="UNIQUE"
            ),

            # KnowledgeNode 节点约束
            ConstraintDefinition(
                name="knowledge_node_id_unique",
                label="KnowledgeNode",
                property="id",
                constraint_type="UNIQUE"
            ),

            # 必填字段约束
            ConstraintDefinition(
                name="entity_name_exists",
                label="Entity",
                property="name",
                constraint_type="EXISTS"
            ),

            ConstraintDefinition(
                name="entity_type_exists",
                label="Entity",
                property="type",
                constraint_type="EXISTS"
            ),
        ]

    def _get_indexes(self) -> List[IndexDefinition]:
        """获取索引定义"""
        return [
            # === 节点属性索引 ===

            # Entity 名称索引
            IndexDefinition(
                name="entity_name_index",
                label="Entity",
                properties=["name"],
                index_type="BTREE"
            ),

            # Entity 类型索引
            IndexDefinition(
                name="entity_type_index",
                label="Entity",
                properties=["type"],
                index_type="BTREE"
            ),

            # Entity 置信度索引
            IndexDefinition(
                name="entity_confidence_index",
                label="Entity",
                properties=["confidence"],
                index_type="BTREE"
            ),

            # Document 标题索引
            IndexDefinition(
                name="document_title_index",
                label="Document",
                properties=["title"],
                index_type="BTREE"
            ),

            # Document 创建时间索引
            IndexDefinition(
                name="document_created_at_index",
                label="Document",
                properties=["created_at"],
                index_type="BTREE"
            ),

            # === 复合索引 ===

            # Entity 名称+类型复合索引
            IndexDefinition(
                name="entity_name_type_composite",
                label="Entity",
                properties=["name", "type"],
                index_type="BTREE"
            ),

            # === 全文索引 ===

            # Chunk 内容全文索引
            IndexDefinition(
                name="chunk_content_fulltext",
                label="Chunk",
                properties=["content"],
                index_type="FULLTEXT",
                options={"analyzer": "standard"}
            ),

            # Entity 名称和别名全文索引
            IndexDefinition(
                name="entity_aliases_fulltext",
                label="Entity",
                properties=["name", "aliases"],
                index_type="FULLTEXT",
                options={"analyzer": "standard"}
            ),

            # Document 标题和元数据全文索引
            IndexDefinition(
                name="document_fulltext",
                label="Document",
                properties=["title", "metadata"],
                index_type="FULLTEXT",
                options={"analyzer": "standard"}
            ),

            # === 关系索引 ===

            # MENTIONED_IN 关系类型索引
            IndexDefinition(
                name="mentioned_in_type_index",
                label="",
                properties=[],
                index_type="LOOKUP"
            ),

            # 所有关系类型的索引
            IndexDefinition(
                name="relation_type_index",
                label="",
                properties=[],
                index_type="LOOKUP"
            ),
        ]

    def get_schema_cypher(self) -> List[str]:
        """
        ��取创建schema的Cypher语句列表

        Returns:
            Cypher语句列表
        """
        statements = []

        # 创建约束
        for constraint in self.constraints:
            if constraint.constraint_type == "UNIQUE":
                stmt = (
                    f"CREATE CONSTRAINT {constraint.name} IF NOT EXISTS "
                    f"FOR (n:{constraint.label}) REQUIRE n.{constraint.property} IS UNIQUE"
                )
            elif constraint.constraint_type == "EXISTS":
                stmt = (
                    f"CREATE CONSTRAINT {constraint.name} IF NOT EXISTS "
                    f"FOR (n:{constraint.label}) REQUIRE n.{constraint.property} IS NOT NULL"
                )
            else:
                continue

            statements.append(stmt)

        # 创建索引
        for index in self.indexes:
            if index.index_type == "BTREE":
                props = ", ".join(f"n.{p}" for p in index.properties)
                stmt = (
                    f"CREATE INDEX {index.name} IF NOT EXISTS "
                    f"FOR (n:{index.label}) ON ({props})"
                )
            elif index.index_type == "FULLTEXT":
                props = ", ".join(f"n.{p}" for p in index.properties)
                options_str = ", ".join(
                    f"{k}: '{v}'" for k, v in (index.options or {}).items()
                )
                stmt = (
                    f"CREATE FULLTEXT INDEX {index.name} IF NOT EXISTS "
                    f"FOR (n:{index.label}) ON EACH [{props}]"
                )
                if options_str:
                    stmt += f" OPTIONS {{{options_str}}}"
            elif index.index_type == "LOOKUP":
                # 关系类型索引（特殊处理）
                stmt = (
                    f"CREATE INDEX {index.name} IF NOT EXISTS "
                    f"FOR ()-[r]-() ON (type(r))"
                )
            else:
                continue

            statements.append(stmt)

        return statements

    def get_drop_cypher(self) -> List[str]:
        """
        获取删除schema的Cypher语句列表

        Returns:
            Cypher语句列表
        """
        statements = []

        # 删除索引
        for index in self.indexes:
            stmt = f"DROP INDEX {index.name} IF EXISTS"
            statements.append(stmt)

        # 删除约束
        for constraint in self.constraints:
            stmt = f"DROP CONSTRAINT {constraint.name} IF EXISTS"
            statements.append(stmt)

        return statements

    def get_schema_info(self) -> Dict:
        """
        获取schema信息

        Returns:
            Schema信息字典
        """
        return {
            "version": self.schema_version,
            "constraints_count": len(self.constraints),
            "indexes_count": len(self.indexes),
            "constraints": [
                {
                    "name": c.name,
                    "label": c.label,
                    "property": c.property,
                    "type": c.constraint_type
                }
                for c in self.constraints
            ],
            "indexes": [
                {
                    "name": i.name,
                    "label": i.label,
                    "properties": i.properties,
                    "type": i.index_type,
                    "options": i.options
                }
                for i in self.indexes
            ]
        }


class GraphSchemaInitializer:
    """图谱Schema初始化器"""

    def __init__(self, driver):
        """
        初始化Schema初始化器

        Args:
            driver: Neo4j驱动实例
        """
        self.driver = driver
        self.schema_manager = GraphSchemaManager()

    async def initialize_schema(
        self,
        database: str = "neo4j",
        force_recreate: bool = False
    ) -> Dict[str, Any]:
        """
        初始化schema

        Args:
            database: 数据库名称
            force_recreate: 是否强制重新创建（删除后重建）

        Returns:
            初始化结果
        """
        results = {
            "success": False,
            "created_constraints": [],
            "failed_constraints": [],
            "created_indexes": [],
            "failed_indexes": [],
            "errors": []
        }

        try:
            with self.driver.session(database=database) as session:
                # 如果需要，先删除现有schema
                if force_recreate:
                    logger.warning("强制重新创建schema，删除所有约束和索引")
                    drop_statements = self.schema_manager.get_drop_cypher()
                    for stmt in drop_statements:
                        try:
                            session.run(stmt)
                        except Exception as e:
                            logger.warning(f"删除失败: {e}")

                # 创建约束
                logger.info("开始创建约束...")
                for constraint in self.schema_manager.constraints:
                    try:
                        if constraint.constraint_type == "UNIQUE":
                            stmt = (
                                f"CREATE CONSTRAINT {constraint.name} IF NOT EXISTS "
                                f"FOR (n:{constraint.label}) REQUIRE n.{constraint.property} IS UNIQUE"
                            )
                        elif constraint.constraint_type == "EXISTS":
                            stmt = (
                                f"CREATE CONSTRAINT {constraint.name} IF NOT EXISTS "
                                f"FOR (n:{constraint.label}) REQUIRE n.{constraint.property} IS NOT NULL"
                            )
                        else:
                            continue

                        session.run(stmt)
                        results["created_constraints"].append(constraint.name)
                        logger.info(f"✓ 约束创建成功: {constraint.name}")

                    except Exception as e:
                        error_msg = f"约束创建失败 {constraint.name}: {e}"
                        results["failed_constraints"].append(constraint.name)
                        results["errors"].append(error_msg)
                        logger.error(f"✗ {error_msg}")

                # 创建索引
                logger.info("开始创建索引...")
                for index in self.schema_manager.indexes:
                    try:
                        if index.index_type == "BTREE":
                            props = ", ".join(f"n.{p}" for p in index.properties)
                            stmt = (
                                f"CREATE INDEX {index.name} IF NOT EXISTS "
                                f"FOR (n:{index.label}) ON ({props})"
                            )
                        elif index.index_type == "FULLTEXT":
                            props = ", ".join(f"n.{p}" for p in index.properties)
                            options_str = ", ".join(
                                f"{k}: '{v}'" for k, v in (index.options or {}).items()
                            )
                            stmt = (
                                f"CREATE FULLTEXT INDEX {index.name} IF NOT EXISTS "
                                f"FOR (n:{index.label}) ON EACH [{props}]"
                            )
                            if options_str:
                                stmt += f" OPTIONS {{{options_str}}}"
                        elif index.index_type == "LOOKUP":
                            # 关系类型索引
                            if index.name == "relation_type_index":
                                stmt = (
                                    "CREATE INDEX relation_type_lookup IF NOT EXISTS "
                                    "FOR ()-[r:MENTIONED_IN]-() ON (r)"
                                )
                                session.run(stmt)
                            continue
                        else:
                            continue

                        session.run(stmt)
                        results["created_indexes"].append(index.name)
                        logger.info(f"✓ 索引创建成功: {index.name}")

                    except Exception as e:
                        error_msg = f"索引创建失败 {index.name}: {e}"
                        results["failed_indexes"].append(index.name)
                        results["errors"].append(error_msg)
                        logger.error(f"✗ {error_msg}")

                # 检查结果
                if not results["errors"]:
                    results["success"] = True
                    logger.info("Schema初始化完成")
                else:
                    logger.warning(f"Schema初始化完成，但有 {len(results['errors'])} 个错误")

                return results

        except Exception as e:
            error_msg = f"Schema初始化失败: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
            return results

    async def verify_schema(self, database: str = "neo4j") -> Dict[str, Any]:
        """
        验证schema

        Args:
            database: 数据库名称

        Returns:
            验证结果
        """
        try:
            with self.driver.session(database=database) as session:
                # 检查约束
                actual_constraints = session.run("SHOW CONSTRAINTS").data()
                constraint_names = [c["name"] for c in actual_constraints]

                # 检查索引
                actual_indexes = session.run("SHOW INDEXES").data()
                index_names = [i["name"] for i in actual_indexes]

                # 对比期望的schema
                expected_constraints = [c.name for c in self.schema_manager.constraints]
                expected_indexes = [i.name for i in self.schema_manager.indexes]

                missing_constraints = set(expected_constraints) - set(constraint_names)
                missing_indexes = set(expected_indexes) - set(index_names)

                return {
                    "valid": len(missing_constraints) == 0 and len(missing_indexes) == 0,
                    "actual_constraints": len(constraint_names),
                    "actual_indexes": len(index_names),
                    "missing_constraints": list(missing_constraints),
                    "missing_indexes": list(missing_indexes),
                    "details": {
                        "constraints": actual_constraints,
                        "indexes": actual_indexes
                    }
                }

        except Exception as e:
            logger.error(f"Schema验证失败: {e}")
            return {
                "valid": False,
                "error": str(e)
            }


# 全局Schema信息
graph_schema_manager = GraphSchemaManager()
