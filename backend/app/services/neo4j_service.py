"""
Neo4j图数据库服务

⚠️  **DEPRECATED** - 此服务已废弃

请使用新的统一知识图谱服务：
- `app.services.unified_knowledge_graph.UnifiedKnowledgeGraphService`
- `app.services.fusion_service.fusion_document_service`

迁移原因：
- 旧服务存在实体重复、ID冲突问题
- 缺少跨文档实体消歧功能
- 固定置信度，无法动态调整

迁移指南：请参阅 `docs/代码迁移指南.md`

此文件保留用于向后兼容，将在未来版本中移除。
"""

import warnings
warnings.warn(
    "Neo4jService 已废弃，请使用 UnifiedKnowledgeGraphService。"
    "详见 docs/代码迁移指南.md",
    DeprecationWarning,
    stacklevel=2
)

from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver
import logging
import json
import uuid
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class Neo4jService:
    """Neo4j图数据库服务"""

    def __init__(self):
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.driver: Optional[Driver] = None
        self.is_connected = False

        # 调试日志
        logger.info(f"初始化Neo4j服务: URI={self.uri}, User={self.user}")

    async def connect(self):
        """连接到Neo4j"""
        try:
            logger.info(f"尝试连接Neo4j: {self.uri}")
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )

            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record["test"] == 1:
                    logger.info(f"成功连接到Neo4j: {self.uri}")
                else:
                    raise Exception("连接测试失败")

        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise

    async def disconnect(self):
        """断开Neo4j连接"""
        try:
            if self.driver:
                self.driver.close()
                logger.info("已断开Neo4j连接")
        except Exception as e:
            logger.error(f"断开Neo4j连接失败: {e}")

    async def init_constraints(self):
        """初始化约束和索引"""
        try:
            await self.connect()

            with self.driver.session() as session:
                # 创建唯一性约束
                constraints = [
                    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE"
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logger.info(f"约束创建/验证成功: {constraint}")
                    except Exception as e:
                        logger.warning(f"约束创建失败: {e}")

                # 创建索引
                indexes = [
                    "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)",
                    "CREATE INDEX chunk_content_index IF NOT EXISTS FOR (c:Chunk) ON (c.content)"
                ]

                for index in indexes:
                    try:
                        session.run(index)
                        logger.info(f"索引创建/验证成功: {index}")
                    except Exception as e:
                        logger.warning(f"索引创建失败: {e}")

            logger.info("Neo4j约束和索引初始化完成")

        except Exception as e:
            logger.error(f"初始化Neo4j约束失败: {e}")
            raise

    async def create_document_node(self, document_id: int, title: str, metadata: Dict[str, Any]) -> bool:
        """创建文档节点"""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (d:Document {id: $document_id})
                SET d.title = $title,
                    d.metadata = $metadata,
                    d.created_at = datetime(),
                    d.updated_at = datetime()
                RETURN d
                """

                result = session.run(query, {
                    "document_id": document_id,
                    "title": title,
                    "metadata": json.dumps(metadata)
                })

                return result.single() is not None

        except Exception as e:
            logger.error(f"创建文档节点失败: {e}")
            return False

    async def create_chunk_nodes(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]]
    ) -> List[int]:
        """创建文档块节点"""
        try:
            chunk_ids = []

            with self.driver.session() as session:
                for i, chunk in enumerate(chunks):
                    query = """
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        document_id: $document_id,
                        chunk_index: $chunk_index,
                        content: $content,
                        start_char: $start_char,
                        end_char: $end_char,
                        created_at: datetime()
                    })
                    WITH c
                    MATCH (d:Document {id: $document_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    RETURN c.id as id
                    """

                    result = session.run(query, {
                        "chunk_id": f"{document_id}_{i}",
                        "document_id": document_id,
                        "chunk_index": chunk.get("chunk_index", i),
                        "content": chunk["content"][:10000],  # 限制长度
                        "start_char": chunk.get("metadata", {}).get("start_char", 0),
                        "end_char": chunk.get("metadata", {}).get("end_char", 0)
                    })

                    record = result.single()
                    if record:
                        chunk_ids.append(record["id"])

            logger.info(f"创建 {len(chunk_ids)} 个文档块节点，文档ID: {document_id}")
            return chunk_ids

        except Exception as e:
            logger.error(f"创建文档块节点失败: {e}")
            return []

    async def extract_and_create_entities(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """提取实体并创建实体节点"""
        try:
            entity_stats = {"persons": 0, "organizations": 0, "locations": 0, "concepts": 0}

            # 简单的实体提取规则（实际应用中应该使用NER模型）
            import re

            with self.driver.session() as session:
                for chunk in chunks:
                    content = chunk["content"]
                    chunk_id = f"{document_id}_{chunk.get('chunk_index', 0)}"

                    # 提取人名（简单规则）
                    persons = re.findall(r'[\u4e00-\u9fff]{2,4}(?:先生|女士|博士|教授|总裁|CEO|董事长|行长)', content)
                    for person in persons:
                        await self._create_entity_if_not_exists(session, person, "Person", document_id, chunk_id)
                        entity_stats["persons"] += 1

                    # 提取机构（简单规则）
                    orgs = re.findall(r'[\u4e00-\u9fff]*(?:银行|证券|保险|基金|公司|集团|企业|研究院)', content)
                    for org in orgs:
                        await self._create_entity_if_not_exists(session, org, "Organization", document_id, chunk_id)
                        entity_stats["organizations"] += 1

                    # 提取地点（简单规则）
                    locations = re.findall(r'[\u4e00-\u9fff]*(?:市|省|县|区|国|地区)', content)
                    for location in locations:
                        await self._create_entity_if_not_exists(session, location, "Location", document_id, chunk_id)
                        entity_stats["locations"] += 1

                    # 提取概念（简单规则）
                    concepts = re.findall(r'[\u4e00-\u9fff]*(?:率|比、指标、指数、政策、法规、标准)', content)
                    for concept in concepts:
                        await self._create_entity_if_not_exists(session, concept, "Concept", document_id, chunk_id)
                        entity_stats["concepts"] += 1

            logger.info(f"实体提取完成: {entity_stats}")
            return entity_stats

        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return {"persons": 0, "organizations": 0, "locations": 0, "concepts": 0}

    async def _create_entity_if_not_exists(
        self,
        session,
        name: str,
        entity_type: str,
        document_id: int,
        chunk_id: str
    ):
        """创建实体节点（如果不存在）"""
        try:
            # 创建或获取实体节点
            query = """
            MERGE (e:Entity {name: $name, type: $entity_type})
            ON CREATE SET e.id = random() * 1000000, e.created_at = datetime()
            ON MATCH SET e.updated_at = datetime()
            WITH e
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """

            session.run(query, {
                "name": name,
                "entity_type": entity_type,
                "chunk_id": chunk_id
            })

        except Exception as e:
            logger.error(f"创建实体节点失败: {e}")

    async def search_related_entities(
        self,
        keywords: List[str],
        document_ids: Optional[List[int]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索相关实体和关系"""
        try:
            with self.driver.session() as session:
                # 构建查询条件
                keyword_conditions = " OR ".join([f"e.name CONTAINS '{kw}'" for kw in keywords])

                doc_filter = ""
                if document_ids:
                    doc_ids_str = ", ".join(map(str, document_ids))
                    doc_filter = f"AND d.document_id IN [{doc_ids_str}]"

                query = f"""
                MATCH (e:Entity)
                WHERE ({keyword_conditions})
                MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                MATCH (d:Document)-[:HAS_CHUNK]->(c)
                WHERE d.document_id IS NOT NULL {doc_filter}
                OPTIONAL MATCH (e)-[r]-(related:Entity)
                RETURN DISTINCT
                    e.name as entity_name,
                    e.type as entity_type,
                    c.content as content,
                    d.id as document_id,
                    d.title as document_title,
                    collect(DISTINCT related.name) as related_entities,
                    count(DISTINCT related) as relation_count
                ORDER BY relation_count DESC, length(content) DESC
                LIMIT {limit}
                """

                result = session.run(query)
                records = result.data()

                search_results = []
                for record in records:
                    search_results.append({
                        "content": record["content"],
                        "score": min(1.0, record["relation_count"] / 5.0),  # 简单评分
                        "source_type": "graph",
                        "document_id": record["document_id"],
                        "entities": [record["entity_name"]],
                        "relations": record["related_entities"],
                        "document_title": record["document_title"]
                    })

                logger.info(f"图数据库搜索完成，返回 {len(search_results)} 个结果")
                return search_results

        except Exception as e:
            logger.error(f"图数据库搜索失败: {e}")
            return []

    async def get_entity_documents(
        self,
        entity_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取与实体相关联的所有文档

        Args:
            entity_name: 实体名称
            limit: 返回结果数量

        Returns:
            文档列表
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Entity {name: $entity_name})
                MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                MATCH (d:Document)-[:HAS_CHUNK]->(c)
                WITH d, count(c) as mention_count
                RETURN DISTINCT
                    d.id as document_id,
                    d.title as document_title,
                    d.metadata as metadata,
                    mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
                """

                result = session.run(query, {
                    "entity_name": entity_name,
                    "limit": limit
                })

                documents = []
                for record in result:
                    doc_metadata = {}
                    if record.get("metadata"):
                        try:
                            doc_metadata = json.loads(record["metadata"])
                        except:
                            doc_metadata = {}

                    documents.append({
                        "document_id": record["document_id"],
                        "document_title": record["document_title"],
                        "metadata": doc_metadata,
                        "mention_count": record["mention_count"],
                        "entity_name": entity_name
                    })

                logger.info(f"找到 {len(documents)} 个与实体 '{entity_name}' 相关的文档")
                return documents

        except Exception as e:
            logger.error(f"获取实体文档失败: {e}")
            return []

    async def get_entity_relationships(self, entity_name: str, limit: int = 20) -> Dict[str, Any]:
        """获取实体关系网络"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Entity {name: $entity_name})
                OPTIONAL MATCH (e)-[r1]-(related1:Entity)
                OPTIONAL MATCH (related1)-[r2]-(related2:Entity)
                WHERE related2 <> e
                RETURN DISTINCT
                    e.name as center_entity,
                    collect(DISTINCT {
                        name: related1.name,
                        type: related1.type,
                        relationship: type(r1)
                    }) as direct_relations,
                    collect(DISTINCT related2.name) as indirect_entities
                LIMIT 1
                """

                result = session.run(query, {"entity_name": entity_name})
                record = result.single()

                if record:
                    return {
                        "center_entity": record["center_entity"],
                        "direct_relations": record["direct_relations"],
                        "indirect_entities": record["indirect_entities"]
                    }
                else:
                    return {"center_entity": entity_name, "direct_relations": [], "indirect_entities": []}

        except Exception as e:
            logger.error(f"获取实体关系失败: {e}")
            return {"center_entity": entity_name, "direct_relations": [], "indirect_entities": []}

    async def delete_document_graph(self, document_id: int) -> bool:
        """删除文档相关的图数据"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(e:Entity)
                DETACH DELETE d, c
                """

                session.run(query, {"document_id": document_id})

            logger.info(f"删除文档 {document_id} 的图数据成功")
            return True

        except Exception as e:
            logger.error(f"删除文档图数据失败: {e}")
            return False

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图数据库统计信息"""
        try:
            with self.driver.session() as session:
                # 获取节点统计
                node_stats = {}
                node_types = ["Document", "Chunk", "Entity", "User"]
                for node_type in node_types:
                    result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                    count = result.single()["count"]
                    node_stats[node_type.lower()] = count

                # 获取关系统计
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                relation_count = result.single()["count"]

                # 获取实体类型统计
                result = session.run("MATCH (e:Entity) RETURN e.type as type, count(e) as count")
                entity_type_stats = {}
                for record in result.data():
                    entity_type_stats[record["type"]] = record["count"]

                return {
                    "nodes": node_stats,
                    "relations": relation_count,
                    "entity_types": entity_type_stats
                }

        except Exception as e:
            logger.error(f"获取图统计信息失败: {e}")
            return {"nodes": {}, "relations": 0, "entity_types": {}}


# 新增的增强方法
    async def create_fulltext_indexes(self):
        """创建全文索引"""
        try:
            with self.driver.session() as session:
                # 创建块内容全文索引
                session.run("""
                CREATE FULLTEXT INDEX chunkContentFulltext IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.content]
                """)

                # 创建实体名称全文索引
                session.run("""
                CREATE FULLTEXT INDEX entityNameFulltext IF NOT EXISTS
                FOR (e:Entity) ON EACH [e.name]
                """)

                # 创建文档标题全文索引
                session.run("""
                CREATE FULLTEXT INDEX documentTitleFulltext IF NOT EXISTS
                FOR (d:Document) ON EACH [d.title]
                """)

                logger.info("全文索引创建成功")
                return True

        except Exception as e:
            logger.error(f"创建全文索引失败: {str(e)}")
            return False

    async def enhanced_entity_extraction(
        self,
        chunk_id: str,
        content: str,
        entity_types: Optional[List[str]] = None
    ) -> List[str]:
        """增强的实体提取和链接"""
        try:
            # 使用更丰富的实体类型
            if entity_types is None:
                entity_types = [
                    "PERSON", "ORGANIZATION", "LOCATION", "DATE",
                    "MONEY", "PERCENT", "PRODUCT", "EVENT",
                    "LAW", "FINANCIAL_TERM", "COMPANY", "STOCK"
                ]

            # 这里可以集成更高级的NER模型
            entities = await self._extract_entities_with_ner(content, entity_types)

            # 创建实体并建立关系
            entity_ids = []
            with self.driver.session() as session:
                for entity in entities:
                    # 创建或更新实体
                    entity_query = """
                    MERGE (e:Entity {
                        name: $name,
                        type: $type
                    })
                    ON CREATE SET
                        e.id = $entity_id,
                        e.count = 1,
                        e.first_seen = datetime(),
                        e.aliases = [$alias],
                        e.confidence = $confidence
                    ON MATCH SET
                        e.count = e.count + 1,
                        e.last_seen = datetime(),
                        e.aliases = CASE
                            WHEN $alias IN e.aliases THEN e.aliases
                            ELSE e.aliases + $alias
                        END,
                        e.confidence = (e.confidence + $confidence) / 2
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS {
                        frequency: $frequency,
                        confidence: $confidence,
                        extracted_at: datetime()
                    }]->(e)
                    RETURN e.id as entity_id
                    """

                    result = session.run(entity_query, {
                        "name": entity["name"],
                        "type": entity["type"],
                        "entity_id": str(uuid.uuid4()),
                        "alias": entity.get("alias", entity["name"]),
                        "confidence": entity.get("confidence", 0.8),
                        "frequency": entity.get("frequency", 1),
                        "chunk_id": chunk_id
                    })

                    entity_record = result.single()
                    if entity_record:
                        entity_ids.append(entity_record["entity_id"])

            logger.info(f"为块 {chunk_id} 提取了 {len(entities)} 个实体")
            return entity_ids

        except Exception as e:
            logger.error(f"增强实体提取失败: {str(e)}")
            return []

    async def semantic_search(
        self,
        query: str,
        search_type: str = "fulltext",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """增强的语义搜索"""
        try:
            with self.driver.session() as session:
                if search_type == "fulltext":
                    # 全文搜索
                    query_str = """
                    CALL db.index.fulltext.queryNodes("chunkContentFulltext", $query)
                    YIELD node, score
                    WITH node, score
                    MATCH (d:Document)-[:HAS_CHUNK]->(node)
                    WHERE $filters IS NULL OR
                          ($filters.document_type IS NULL OR d.metadata.type = $filters.document_type)
                    RETURN
                        node.id as chunk_id,
                        node.content as content,
                        d.id as document_id,
                        d.title as document_title,
                        score,
                        node.chunk_index as chunk_index
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                else:
                    # 属性搜索
                    query_str = """
                    MATCH (c:Chunk)
                    WHERE c.content CONTAINS $query
                    MATCH (d:Document)-[:HAS_CHUNK]->(c)
                    WHERE $filters IS NULL OR
                          ($filters.document_type IS NULL OR d.metadata.type = $filters.document_type)
                    RETURN
                        c.id as chunk_id,
                        c.content as content,
                        d.id as document_id,
                        d.title as document_title,
                        1.0 as score,
                        c.chunk_index as chunk_index
                    ORDER BY c.chunk_index
                    LIMIT $limit
                    """

                results = []
                for record in session.run(query_str, {
                    "query": query,
                    "filters": filters or {},
                    "limit": limit
                }):
                    results.append({
                        "chunk_id": record["chunk_id"],
                        "content": record["content"][:500],  # 截取前500字符
                        "document_id": record["document_id"],
                        "document_title": record["document_title"],
                        "score": record["score"],
                        "chunk_index": record.get("chunk_index", 0)
                    })

                return results

        except Exception as e:
            logger.error(f"语义搜索失败: {str(e)}")
            return []

    async def create_entity_relationships(self, chunk_id: str):
        """创建实体间的关系"""
        try:
            with self.driver.session() as session:
                # 提取实体间的关系（简化实现）
                query = """
                MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e1:Entity)
                MATCH (c)-[:MENTIONS]->(e2:Entity)
                WHERE e1.id < e2.id  // 避免重复
                AND e1.type IN ['PERSON', 'ORGANIZATION', 'COMPANY']
                AND e2.type IN ['PERSON', 'ORGANIZATION', 'COMPANY']

                // 检查是否已存在关系
                MERGE (e1)-[r:RELATED_TO]-(e2)
                ON CREATE SET
                    r.context = c.content,
                    r.chunk_id = c.id,
                    r.strength = 1,
                    r.created_at = datetime()
                ON MATCH SET
                    r.strength = r.strength + 1

                RETURN count(r) as relationships_created
                """

                result = session.run(query, {"chunk_id": chunk_id})
                count = result.single()["relationships_created"]

                logger.info(f"为块 {chunk_id} 创建了 {count} 个实体关系")
                return count

        except Exception as e:
            logger.error(f"创建实体关系失败: {str(e)}")
            return 0

    async def get_enhanced_graph_stats(self) -> Dict[str, Any]:
        """获取增强的图谱统计信息"""
        try:
            with self.driver.session() as session:
                # 基础统计
                stats = await self.get_graph_statistics()

                # 获取关系类型统计
                rel_type_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
                """
                rel_type_stats = {}
                for record in session.run(rel_type_query):
                    rel_type_stats[record["rel_type"]] = record["count"]

                # 获取高度连接的实体（中心性分析）
                centrality_query = """
                MATCH (e:Entity)-[r:RELATED_TO]-(other)
                WITH e, count(r) as degree
                ORDER BY degree DESC
                LIMIT 10
                RETURN e.name as entity_name, e.type as entity_type, degree
                """
                top_entities = []
                for record in session.run(centrality_query):
                    top_entities.append({
                        "name": record["entity_name"],
                        "type": record["entity_type"],
                        "degree": record["degree"]
                    })

                # 获取文档-实体分布
                doc_entity_query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                RETURN d.id as document_id, d.title as document_title,
                       count(DISTINCT e) as entity_count
                ORDER BY entity_count DESC
                LIMIT 10
                """
                doc_entity_dist = []
                for record in session.run(doc_entity_query):
                    doc_entity_dist.append({
                        "document_id": record["document_id"],
                        "document_title": record["document_title"],
                        "entity_count": record["entity_count"]
                    })

                stats.update({
                    "relationship_types": rel_type_stats,
                    "top_connected_entities": top_entities,
                    "document_entity_distribution": doc_entity_dist
                })

                return stats

        except Exception as e:
            logger.error(f"获取增强图谱统计失败: {str(e)}")
            return {}

    async def _extract_entities_with_ner(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """使用NER模型提取实体（简化实现）"""
        # 这里可以集成spaCy、Stanza或其他的NER模型
        import re

        entities = []

        # 提取金融术语
        financial_terms = [
            "股票", "基金", "债券", "期货", "期权", "外汇",
            "收益率", "市盈率", "市净率", "ROE", "ROA"
        ]
        for term in financial_terms:
            if term in text:
                entities.append({
                    "name": term,
                    "type": "FINANCIAL_TERM",
                    "confidence": 0.9,
                    "frequency": text.count(term)
                })

        # 提取公司名称（大写字母组合）
        company_pattern = r'\b([A-Z]{2,})\b'
        companies = re.findall(company_pattern, text)
        for company in companies[:5]:  # 限制数量
            entities.append({
                "name": company,
                "type": "COMPANY",
                "confidence": 0.7,
                "frequency": 1
            })

        # 提取金额
        money_pattern = r'[\$¥]?\s*\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:万|亿|千万|million|billion|K|M)?'
        money_matches = re.findall(money_pattern, text)
        for money in money_matches[:3]:
            entities.append({
                "name": money,
                "type": "MONEY",
                "confidence": 0.8,
                "frequency": 1
            })

        # 提取日期
        date_pattern = r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?|\d{1,2}[-/月]\d{1,2}[日号]?'
        dates = re.findall(date_pattern, text)
        for date in dates[:3]:
            entities.append({
                "name": date,
                "type": "DATE",
                "confidence": 0.9,
                "frequency": 1
            })

        return entities

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.is_connected:
                await self.connect()

            with self.driver.session() as session:
                result = session.run("RETURN 1 as health")
                record = result.single()
                return record["health"] == 1

        except Exception as e:
            logger.error(f"Neo4j健康检查失败: {e}")
            return False

    async def create_knowledge_graph_node(
        self,
        node_id: str,
        node_name: str,
        node_type: str,
        properties: Dict[str, Any],
        document_id: int
    ) -> bool:
        """创建知识图谱节点"""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (n:KnowledgeNode {id: $node_id})
                SET n.name = $node_name,
                    n.type = $node_type,
                    n.properties = $properties,
                    n.document_id = $document_id,
                    n.created_at = datetime(),
                    n.updated_at = datetime()
                RETURN n
                """

                result = session.run(query, {
                    "node_id": node_id,
                    "node_name": node_name,
                    "node_type": node_type,
                    "properties": json.dumps(properties),
                    "document_id": document_id
                })

                return result.single() is not None

        except Exception as e:
            logger.error(f"创建知识图谱节点失败: {e}")
            return False

    async def create_knowledge_graph_relation(
        self,
        relation_id: str,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        properties: Dict[str, Any],
        document_id: int,
        weight: float = 1.0
    ) -> bool:
        """创建知识图谱关系"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source:KnowledgeNode {id: $source_node_id})
                MATCH (target:KnowledgeNode {id: $target_node_id})
                MERGE (source)-[r:RELATION {id: $relation_id}]->(target)
                SET r.type = $relation_type,
                    r.properties = $properties,
                    r.document_id = $document_id,
                    r.weight = $weight,
                    r.created_at = datetime(),
                    r.updated_at = datetime()
                RETURN r
                """

                result = session.run(query, {
                    "relation_id": relation_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "relation_type": relation_type,
                    "properties": json.dumps(properties),
                    "document_id": document_id,
                    "weight": weight
                })

                return result.single() is not None

        except Exception as e:
            logger.error(f"创建知识图谱关系失败: {e}")
            return False

    async def search_entities_and_relations(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索实体和关系"""
        try:
            results = []

            with self.driver.session() as session:
                # 搜索节点
                node_query = """
                MATCH (n:KnowledgeNode)
                WHERE n.name CONTAINS $query
                   OR n.type CONTAINS $query
                OPTIONAL MATCH (n)-[r]-(connected)
                WITH n, collect(DISTINCT connected.name) as connected_nodes,
                     collect(DISTINCT type(r)) as relation_types
                RETURN n.id as node_id,
                       n.name as node_name,
                       n.type as node_type,
                       n.properties as properties,
                       connected_nodes,
                       relation_types,
                       'node' as type,
                       apoc.text.levenshteinSimilarity(n.name, $query) as similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """

                node_result = session.run(node_query, {"query": query, "limit": limit})
                for record in node_result:
                    results.append({
                        "type": "node",
                        "node_id": record["node_id"],
                        "node_name": record["node_name"],
                        "node_type": record["node_type"],
                        "properties": json.loads(record["properties"]) if record["properties"] else {},
                        "connected_nodes": record["connected_nodes"],
                        "relation_types": record["relation_types"],
                        "score": record["similarity"]
                    })

                # 搜索关系
                relation_query = """
                MATCH (source)-[r:RELATION]-(target)
                WHERE r.type CONTAINS $query
                   OR source.name CONTAINS $query
                   OR target.name CONTAINS $query
                RETURN r.id as relation_id,
                       source.name as source_name,
                       source.id as source_node_id,
                       target.name as target_name,
                       target.id as target_node_id,
                       r.type as relation_type,
                       r.weight as weight,
                       r.properties as properties,
                       'relation' as type,
                       CASE
                           WHEN r.type CONTAINS $query THEN 1.0
                           WHEN source.name CONTAINS $query THEN 0.8
                           WHEN target.name CONTAINS $query THEN 0.8
                           ELSE 0.5
                       END as score
                ORDER BY score DESC
                LIMIT $limit
                """

                relation_result = session.run(relation_query, {"query": query, "limit": limit})
                for record in relation_result:
                    results.append({
                        "type": "relation",
                        "relation_id": record["relation_id"],
                        "source_node_id": record["source_node_id"],
                        "source_name": record["source_name"],
                        "target_node_id": record["target_node_id"],
                        "target_name": record["target_name"],
                        "relation_type": record["relation_type"],
                        "weight": record["weight"],
                        "properties": json.loads(record["properties"]) if record["properties"] else {},
                        "score": record["score"]
                    })

            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)

            return results[:limit]

        except Exception as e:
            logger.error(f"搜索实体和关系失败: {e}")
            return []

    async def get_entity_relationships(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """获取实体的关系网络"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (start:KnowledgeNode {id: $entity_id})
                CALL apoc.path.expandConfig(start, {
                    relationshipFilter: "RELATION",
                    maxLevel: $max_depth,
                    uniqueness: "RELATIONSHIP_GLOBAL"
                })
                YIELD path
                UNWIND nodes(path) as node
                UNWIND relationships(path) as rel
                RETURN DISTINCT
                    start.id as center_id,
                    start.name as center_name,
                    collect(DISTINCT {
                        id: node.id,
                        name: node.name,
                        type: node.type,
                        level: length([n in nodes(path) WHERE n = node][0]) - 1
                    }) as nodes,
                    collect(DISTINCT {
                        id: rel.id,
                        type: rel.type,
                        source: startNode(rel).id,
                        target: endNode(rel).id,
                        weight: rel.weight
                    }) as relationships
                LIMIT 1
                """

                result = session.run(query, {"entity_id": entity_id, "max_depth": max_depth})
                record = result.single()

                if record:
                    return {
                        "center_id": record["center_id"],
                        "center_name": record["center_name"],
                        "nodes": record["nodes"],
                        "relationships": record["relationships"]
                    }
                else:
                    return {}

        except Exception as e:
            logger.error(f"获取实体关系网络失败: {e}")
            return {}

    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """查找两个实体之间的最短路径"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = shortestPath((source:KnowledgeNode {id: $source_id})-[*1..%d]-(target:KnowledgeNode {id: $target_id}))
                RETURN [node in nodes(path) | {
                    id: node.id,
                    name: node.name,
                    type: node.type
                }] as nodes,
                [rel in relationships(path) | {
                    id: rel.id,
                    type: rel.type,
                    weight: rel.weight
                }] as relationships,
                length(path) as path_length
                """ % max_depth

                result = session.run(query, {
                    "source_id": source_id,
                    "target_id": target_id
                })

                paths = []
                for record in result:
                    paths.append({
                        "nodes": record["nodes"],
                        "relationships": record["relationships"],
                        "path_length": record["path_length"]
                    })

                return paths

        except Exception as e:
            logger.error(f"查找最短路径失败: {e}")
            return []

    async def get_document_knowledge_graph(
        self,
        document_id: int
    ) -> Dict[str, Any]:
        """获取文档的知识图谱"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:KnowledgeNode {document_id: $document_id})
                OPTIONAL MATCH (n)-[r:RELATION]-(m:KnowledgeNode {document_id: $document_id})
                WITH n, collect(DISTINCT {
                    id: r.id,
                    type: r.type,
                    target_id: m.id,
                    target_name: m.name,
                    weight: r.weight
                }) as relationships
                RETURN collect({
                    id: n.id,
                    name: n.name,
                    type: n.type,
                    properties: n.properties,
                    relationships: relationships
                }) as nodes,
                count(DISTINCT n) as node_count,
                count(DISTINCT r) as relation_count
                """

                result = session.run(query, {"document_id": document_id})
                record = result.single()

                if record:
                    return {
                        "nodes": record["nodes"],
                        "node_count": record["node_count"],
                        "relation_count": record["relation_count"]
                    }
                else:
                    return {"nodes": [], "node_count": 0, "relation_count": 0}

        except Exception as e:
            logger.error(f"获取文档知识图谱失败: {e}")
            return {"nodes": [], "node_count": 0, "relation_count": 0}

    async def update_node_properties(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """更新节点属性"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:KnowledgeNode {id: $node_id})
                SET n.properties = $properties,
                    n.updated_at = datetime()
                RETURN n
                """

                result = session.run(query, {
                    "node_id": node_id,
                    "properties": json.dumps(properties)
                })

                return result.single() is not None

        except Exception as e:
            logger.error(f"更新节点属性失败: {e}")
            return False

    async def delete_node(self, node_id: str) -> bool:
        """删除节点及其所有关系"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:KnowledgeNode {id: $node_id})
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """

                result = session.run(query, {"node_id": node_id})
                record = result.single()

                return record["deleted_count"] > 0

        except Exception as e:
            logger.error(f"删除节点失败: {e}")
            return False

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        try:
            with self.driver.session() as session:
                # 节点统计
                node_stats = session.run("""
                MATCH (n:KnowledgeNode)
                RETURN n.type as node_type, count(n) as count
                """)

                node_statistics = {}
                for record in node_stats:
                    node_statistics[record["node_type"]] = record["count"]

                # 关系统计
                rel_stats = session.run("""
                MATCH ()-[r:RELATION]-()
                RETURN r.type as relation_type, count(r) as count
                """)

                relation_statistics = {}
                for record in rel_stats:
                    relation_statistics[record["relation_type"]] = record["count"]

                # 总体统计
                total_stats = session.run("""
                MATCH (n:KnowledgeNode)
                MATCH ()-[r:RELATION]-()
                RETURN count(DISTINCT n) as total_nodes,
                       count(DISTINCT r) as total_relations
                """)

                total_record = total_stats.single()

                return {
                    "node_statistics": node_statistics,
                    "relation_statistics": relation_statistics,
                    "total_nodes": total_record["total_nodes"],
                    "total_relations": total_record["total_relations"]
                }

        except Exception as e:
            logger.error(f"获取图统计信息失败: {e}")
            return {}

    async def get_related_documents(
        self,
        entity_names: List[str],
        limit: int = 10,
        max_depth: int = 3,
        max_relations: int = 30
    ) -> List[Dict[str, Any]]:
        """
        基于实体列表获取相关文档 - 增强版本

        Args:
            entity_names: 实体名称列表
            limit: 返回文档数量限制
            max_depth: 最大遍历深度
            max_relations: 最大关系数量

        Returns:
            相关文档列表
        """
        if not entity_names:
            return []

        try:
            if not self.driver:
                await self.connect()

            with self.driver.session() as session:
                results = []

                for entity_name in entity_names[:10]:  # 限制实体数量防止查询过大
                    # 使用多跳查询获取相关文档和关系
                    query = f"""
                    MATCH (e:Entity {{name: $entity_name}})
                    OPTIONAL MATCH path = (e)-[r1:APPEARS_IN]->(c:Chunk)-[:BELONGS_TO]->(d:Document)
                    OPTIONAL MATCH (e)-[r2:RELATED_TO*1..{max_depth}]-(related:Entity)-[:APPEARS_IN]->(rc:Chunk)-[:BELONGS_TO]->(rd:Document)
                    WITH d, rc, rd,
                         count(DISTINCT r1) as direct_mentions,
                         count(DISTINCT r2) as indirect_relations,
                         collect(DISTINCT r1) as direct_rels,
                         collect(DISTINCT r2) as indirect_rels
                    RETURN d.id as document_id,
                           d.title as title,
                           rc.content as content,
                           rc.id as chunk_id,
                           (direct_mentions * 0.7 + indirect_relations * 0.3) as score,
                           direct_mentions + indirect_relations as total_relations,
                           [rel in direct_rels | rel.type] as relation_types
                    ORDER BY score DESC
                    LIMIT {limit * 2}
                    """

                    try:
                        result = session.run(query, entity_name=entity_name)

                        for record in result:
                            # 检查是否已经添加过该文档
                            doc_id = record.get("document_id")
                            if not doc_id or any(r.get("document_id") == doc_id for r in results):
                                continue

                            results.append({
                                "document_id": doc_id,
                                "title": record.get("title", ""),
                                "content": record.get("content", ""),
                                "chunk_id": record.get("chunk_id"),
                                "score": record.get("score", 0.5),
                                "relations": record.get("relation_types", []),
                                "total_relations": record.get("total_relations", 0),
                                "entity_source": entity_name
                            })

                            if len(results) >= limit:
                                break

                    except Exception as query_error:
                        logger.warning(f"查询实体 {entity_name} 失败: {query_error}")
                        continue

                    if len(results) >= limit:
                        break

                # 按分数排序
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                return results[:limit]

        except Exception as e:
            logger.error(f"获取相关文档失败: {e}", exc_info=True)
            return []


# 全局Neo4j服务实例
neo4j_service = Neo4jService()