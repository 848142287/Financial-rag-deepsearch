"""
GraphRAG - 知识图谱增强检索
使用Neo4j知识图谱实现关系推理和实体增强检索
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
import re

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j not installed. Install with: pip install neo4j")

from sqlalchemy import create_engine, text


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]


@dataclass
class Relation:
    """关系"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]


@dataclass
class GraphRAGConfig:
    """GraphRAG配置"""
    max_depth: int = 2  # 最大遍历深度
    max_entities: int = 10  # 最大实体数
    max_relations: int = 20  # 最大关系数
    entity_score_threshold: float = 0.5
    use_entity_linking: bool = True
    use_relation_traversal: bool = True


class GraphRAG:
    """知识图谱增强检索"""

    def __init__(
        self,
        neo4j_config: Dict[str, Any],
        mysql_config: Dict[str, Any],
        config: GraphRAGConfig = None
    ):
        """
        初始化GraphRAG

        Args:
            neo4j_config: Neo4j配置
            mysql_config: MySQL配置
            config: GraphRAG配置
        """
        self.neo4j_config = neo4j_config
        self.mysql_config = mysql_config
        self.config = config or GraphRAGConfig()

        # Neo4j连接
        if NEO4J_AVAILABLE:
            self.driver = GraphDatabase.driver(
                neo4j_config.get("uri", "bolt://neo4j:7687"),
                auth=(
                    neo4j_config.get("user", "neo4j"),
                    neo4j_config.get("password", "neo4j123")
                )
            )
        else:
            self.driver = None

        # MySQL连接
        self.mysql_engine = create_engine(
            f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@"
            f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
        )

    def extract_entities(self, query: str) -> List[Entity]:
        """
        从查询中提取实体

        Args:
            query: 查询文本

        Returns:
            实体列表
        """
        entities = []

        # 简单的实体提取策略
        # 1. 提取大写字母开头的词（可能是专有名词）
        words = re.findall(r'\b[A-Z][a-z]+\b', query)

        # 2. 提取数字+单位的组合
        numbers = re.findall(r'\b\d+[%年月日元]\b', query)

        potential_entities = words + numbers

        # 在知识图谱中查找这些实体
        for entity_name in potential_entities:
            matched_entities = self._find_entities_in_graph(entity_name)
            entities.extend(matched_entities)

        # 去重
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.id not in seen:
                seen.add(entity.id)
                unique_entities.append(entity)

        return unique_entities[:self.config.max_entities]

    def _find_entities_in_graph(self, entity_name: str) -> List[Entity]:
        """
        在知识图谱中查找实体

        Args:
            entity_name: 实体名称

        Returns:
            匹配的实体列表
        """
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                # 模糊匹配实体名称
                result = session.run("""
                    MATCH (n)
                    WHERE n.name CONTAINS $entity_name
                    RETURN n
                    LIMIT 5
                """, entity_name=entity_name)

                entities = []
                for record in result:
                    node = record["n"]
                    entities.append(Entity(
                        id=str(node.element_id),
                        name=node.get("name", ""),
                        type=list(node.labels)[0] if node.labels else "Unknown",
                        properties=dict(node)
                    ))

                return entities

        except Exception as e:
            print(f"Failed to find entities: {e}")
            return []

    def traverse_graph(
        self,
        entities: List[Entity],
        max_depth: int = None
    ) -> List[Relation]:
        """
        从实体出发遍历知识图谱

        Args:
            entities: 起始实体列表
            max_depth: 最大遍历深度

        Returns:
            关系列表
        """
        if not self.driver or not entities:
            return []

        max_depth = max_depth or self.config.max_depth
        relations = []
        seen_relations = set()

        try:
            with self.driver.session() as session:
                for entity in entities:
                    # BFS遍历
                    result = session.run("""
                        MATCH path = (start:Entity {name: $entity_name})-[*1..{depth}]-(related)
                        RETURN relationships(path) as rels
                        LIMIT 50
                    """, entity_name=entity.name, depth=max_depth)

                    for record in result:
                        for rel in record["rels"]:
                            rel_key = f"{rel.start_node.element_id}_{rel.type}_{rel.end_node.element_id}"

                            if rel_key not in seen_relations:
                                seen_relations.add(rel_key)

                                relations.append(Relation(
                                    source=str(rel.start_node.element_id),
                                    target=str(rel.end_node.element_id),
                                    relation_type=rel.type,
                                    properties=dict(rel)
                                ))

                                if len(relations) >= self.config.max_relations:
                                    return relations

            return relations

        except Exception as e:
            print(f"Failed to traverse graph: {e}")
            return []

    def find_related_documents(
        self,
        entities: List[Entity],
        relations: List[Relation]
    ) -> List[str]:
        """
        基于实体和关系查找相关文档

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            相关文档ID列表
        """
        if not self.driver:
            return []

        related_doc_ids = set()

        try:
            with self.driver.session() as session:
                # 查找与实体相关的文档
                for entity in entities:
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE e.name = $entity_name
                        MATCH (e)-[r:APPEARS_IN]->(d:Document)
                        RETURN d.id as doc_id
                    """, entity_name=entity.name)

                    for record in result:
                        if record["doc_id"]:
                            related_doc_ids.add(str(record["doc_id"]))

            return list(related_doc_ids)

        except Exception as e:
            print(f"Failed to find related documents: {e}")
            return []

    def retrieve(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        执行基于知识图谱的检索

        Args:
            query: 查询文本

        Returns:
            检索结果
        """
        start_time = time.time()

        # 1. 实体识别与链接
        entities = []
        if self.config.use_entity_linking:
            entities = self.extract_entities(query)

        # 2. 关系遍历
        relations = []
        if self.config.use_relation_traversal and entities:
            relations = self.traverse_graph(entities)

        # 3. 查找相关文档
        related_doc_ids = []
        if entities or relations:
            related_doc_ids = self.find_related_documents(entities, relations)

        # 4. 获取文档详情
        doc_details = {}
        if related_doc_ids:
            doc_details = self._get_document_details(related_doc_ids)

        # 5. 组装结果
        results = []
        for doc_id in related_doc_ids[:self.config.max_entities]:
            doc_info = doc_details.get(doc_id, {})
            results.append({
                "id": doc_id,
                "title": doc_info.get("title", ""),
                "filename": doc_info.get("filename", ""),
                "score": 1.0,  # 图谱检索默认最高分
                "retrieval_method": "knowledge_graph",
                "entities": [{"name": e.name, "type": e.type} for e in entities],
                "relations": [{"type": r.relation_type} for r in relations[:5]]
            })

        retrieval_time = time.time() - start_time

        return {
            "query": query,
            "results": results,
            "total_retrieved": len(results),
            "entities_found": len(entities),
            "relations_found": len(relations),
            "retrieval_time": retrieval_time,
            "method": "graphrag"
        }

    def _get_document_details(
        self,
        document_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """获取文档详情"""
        if not document_ids:
            return {}

        try:
            with self.mysql_engine.connect() as conn:
                placeholders = ",".join([":id" + str(i) for i in range(len(document_ids))])
                sql_query = text(f"""
                    SELECT
                        id,
                        title,
                        filename,
                        metadata
                    FROM documents
                    WHERE id IN ({placeholders})
                """)

                params = {f"id{i}": doc_id for i, doc_id in enumerate(document_ids)}
                result = conn.execute(sql_query, params)

                doc_details = {}
                for row in result:
                    doc_details[str(row.id)] = {
                        "title": row.title,
                        "filename": row.filename,
                        "metadata": row.metadata if row.metadata else {}
                    }

                return doc_details

        except Exception as e:
            print(f"Failed to get document details: {e}")
            return {}

    def get_entity_context(
        self,
        entity_name: str
    ) -> Dict[str, Any]:
        """
        获取实体的上下文信息

        Args:
            entity_name: 实体名称

        Returns:
            实体上下文
        """
        if not self.driver:
            return {}

        try:
            with self.driver.session() as session:
                # 获取实体及其关系
                result = session.run("""
                    MATCH (e:Entity {name: $entity_name})
                    OPTIONAL MATCH (e)-[r]-(related)
                    RETURN e,
                           collect(DISTINCT {type: type(r), target: related.name}) as relations
                """, entity_name=entity_name)

                records = list(result)
                if not records:
                    return {}

                record = records[0]
                entity = record["e"]

                return {
                    "name": entity.get("name", ""),
                    "type": list(entity.labels)[0] if entity.labels else "Unknown",
                    "properties": dict(entity),
                    "relations": record["relations"]
                }

        except Exception as e:
            print(f"Failed to get entity context: {e}")
            return {}

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
