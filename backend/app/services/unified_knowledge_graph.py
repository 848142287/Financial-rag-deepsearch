"""
统一知识图谱管理服务
整合所有修复后的组件，提供统一的图谱构建、查询和管理接口
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

from app.core.graph_config import (
    GraphEntityType,
    GraphRelationType,
    graph_entity_config,
    graph_relation_config,
    graph_storage_config,
    generate_entity_id,
    generate_relation_id,
    normalize_entity_name
)

from app.services.knowledge.graph_db_client import GraphDBClient, Neo4jClientError
from app.services.knowledge.entity_disambiguation import (
    EntityDisambiguationService,
    EntityMention,
    EntityCluster
)
from app.services.knowledge.enhanced_relation_extractor import (
    EnhancedRelationExtractor,
    ExtractedRelation
)
from app.services.knowledge.graph_quality_validator import (
    GraphQualityValidator,
    QualityLevel
)
from app.services.knowledge.graph_schema_manager import GraphSchemaInitializer

logger = logging.getLogger(__name__)


@dataclass
class DocumentKGResult:
    """文档知识图谱处理结果"""
    document_id: str
    success: bool
    entity_count: int
    relation_count: int
    quality_score: float
    errors: List[str]
    warnings: List[str]
    processing_time: float


class UnifiedKnowledgeGraphService:
    """
    统一知识图谱管理服务
    整合实体抽取、关系抽取、消歧、质量验证和存储
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化统一知识图谱服务

        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}

        # 初始化组件
        self.graph_client = GraphDBClient()
        self.entity_disambiguator = EntityDisambiguationService()
        self.relation_extractor = EnhancedRelationExtractor()
        self.quality_validator = GraphQualityValidator()

        # 是否已初始化
        self._initialized = False

        logger.info("统一知识图谱管理服务初始化完成")

    async def initialize(self, force_recreate_schema: bool = False):
        """
        初始化服务

        Args:
            force_recreate_schema: 是否强制重新创建schema
        """
        try:
            # 连接Neo4j
            await self.graph_client.connect()

            # 初始化Schema
            schema_initializer = GraphSchemaInitializer(self.graph_client.driver)
            schema_result = await schema_initializer.initialize_schema(
                database=graph_storage_config.database,
                force_recreate=force_recreate_schema
            )

            if not schema_result["success"]:
                logger.warning(f"Schema初始化有警告: {schema_result['errors']}")
            else:
                logger.info("Schema初始化成功")

            self._initialized = True
            logger.info("统一知识图谱服务已初始化")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    async def process_document(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict] = None
    ) -> DocumentKGResult:
        """
        处理文档，构建知识图谱

        Args:
            document_id: 文档ID
            chunks: 文档块列表
            metadata: 文档元数据（可选）

        Returns:
            处理结果
        """
        start_time = datetime.now()

        result = DocumentKGResult(
            document_id=document_id,
            success=False,
            entity_count=0,
            relation_count=0,
            quality_score=0.0,
            errors=[],
            warnings=[],
            processing_time=0.0
        )

        try:
            if not self._initialized:
                await self.initialize()

            logger.info(f"开始处理文档: {document_id}, 块数: {len(chunks)}")

            # 创建文档节点
            await self._create_document_node(document_id, metadata)

            # 存储所有实体提及
            all_entity_mentions = []

            # 步骤1: 实体抽取
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", f"{document_id}_chunk_{chunk.get('index', 0)}")
                content = chunk["content"]

                # 抽取实体
                entities = await self._extract_entities_from_chunk(
                    content, document_id, chunk_id
                )
                all_entity_mentions.extend(entities)

                # 创建chunk节点
                await self._create_chunk_node(chunk_id, document_id, chunk)

            # 步骤2: 实体消歧和聚类
            entity_clusters = await self.entity_disambiguator.disambiguate_entities(
                all_entity_mentions
            )

            # 步骤3: 存储实体到Neo4j
            entity_id_map = {}
            for cluster in entity_clusters:
                node_id = await self._store_entity_cluster(cluster, document_id)
                entity_id_map[cluster.canonical_id] = node_id

            result.entity_count = len(entity_clusters)

            # 步骤4: 关系抽取
            all_relations = []
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", f"{document_id}_chunk_{chunk.get('index', 0)}")
                content = chunk["content"]

                # 准备实体列表
                chunk_entities = [
                    {
                        "id": e.id,
                        "name": e.name,
                        "type": e.type.value,
                        "start": e.start if hasattr(e, 'start') else 0,
                        "end": e.end if hasattr(e, 'end') else 0
                    }
                    for e in all_entity_mentions
                    if e.chunk_id == chunk_id
                ]

                # 抽取关系
                relations = await self.relation_extractor.extract_relations(
                    content, chunk_entities, chunk_id,
                    use_llm=graph_relation_config.use_llm
                )
                all_relations.extend(relations)

            # 步骤5: 关系消歧和质量验证
            all_relations = await self.quality_validator.resolve_conflicts(
                all_relations,
                strategy=graph_quality_config.resolve_conflicts
            )

            # 步骤6: 存储关系到Neo4j
            for relation in all_relations:
                await self._store_relation(relation, entity_id_map)

            result.relation_count = len(all_relations)

            # 步骤7: 质量验证
            entities_data = [
                self._cluster_to_dict(c, entity_id_map.get(c.canonical_id))
                for c in entity_clusters
            ]
            relations_data = [self._relation_to_dict(r) for r in all_relations]

            quality_report = await self.quality_validator.generate_quality_report(
                entities_data, relations_data
            )
            result.quality_score = quality_report["summary"]["quality_score"]

            # 收集警告
            if quality_report["summary"]["warning_count"] > 0:
                result.warnings.extend(quality_report["recommendations"])

            result.success = True

        except Exception as e:
            error_msg = f"处理文档失败: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

        finally:
            result.processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"文档处理完成: {document_id}, "
            f"实体数: {result.entity_count}, "
            f"关系数: {result.relation_count}, "
            f"质量分: {result.quality_score:.1f}, "
            f"耗时: {result.processing_time:.2f}s"
        )

        return result

    async def _extract_entities_from_chunk(
        self,
        content: str,
        document_id: str,
        chunk_id: str
    ) -> List[EntityMention]:
        """从文档块抽取实体"""
        # 这里应该调用实际的实体抽取器
        # 简化实现：使用规则抽取
        # TODO: 集成实际的实体抽取服务

        entities = []

        # 简单示例：提取公司实体
        import re
        company_pattern = r'[\u4e00-\u9fff]+(?:公司|集团|控股|科技|银行|证券|保险)'
        for match in re.finditer(company_pattern, content):
            entity_name = match.group()
            entities.append(EntityMention(
                id=f"{chunk_id}_{len(entities)}",
                name=entity_name,
                normalized_name=normalize_entity_name(entity_name, GraphEntityType.COMPANY),
                type=GraphEntityType.COMPANY,
                confidence=0.8,
                document_id=document_id,
                chunk_id=chunk_id,
                properties={"start": match.start(), "end": match.end()}
            ))

        return entities

    async def _create_document_node(self, document_id: str, metadata: Optional[Dict]):
        """创建文档节点"""
        properties = {
            "id": document_id,
            "title": metadata.get("title", "") if metadata else "",
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }

        await self.graph_client.create_node("Document", properties)

    async def _create_chunk_node(
        self,
        chunk_id: str,
        document_id: str,
        chunk_data: Dict
    ):
        """创建文档块节点"""
        properties = {
            "id": chunk_id,
            "document_id": document_id,
            "content": chunk_data["content"][:10000],  # 限制长度
            "chunk_index": chunk_data.get("index", 0),
            "created_at": datetime.now().isoformat()
        }

        await self.graph_client.create_node("Chunk", properties)

        # 创建Document-HAS_CHUNK->Chunk关系
        # TODO: 实现关系创建

    async def _store_entity_cluster(
        self,
        cluster: EntityCluster,
        document_id: str
    ) -> str:
        """存储实体簇到Neo4j"""
        properties = {
            "canonical_id": cluster.canonical_id,
            "name": cluster.canonical_name,
            "type": cluster.type.value,
            "aliases": list(cluster.aliases),
            "confidence": cluster.confidence,
            "mention_count": cluster.mention_count,
            "document_count": cluster.document_count,
            "properties": cluster.properties,
            "documents": [document_id]
        }

        # 检查是否已存在
        existing = await self.graph_client.find_entity_by_name(
            cluster.canonical_name,
            cluster.type.value
        )

        if existing:
            # 更新现有实体
            properties["documents"] = existing.get("documents", []) + [document_id]
            properties["mention_count"] = existing.get("mention_count", 0) + cluster.mention_count
            await self.graph_client.update_node(existing["canonical_id"], properties)
            return existing["canonical_id"]
        else:
            # 创建新实体
            return await self.graph_client.create_node("Entity", properties)

    async def _store_relation(
        self,
        relation: ExtractedRelation,
        entity_id_map: Dict[str, str]
    ):
        """存储关系到Neo4j"""
        # 查找源和目标实体ID
        source_id = entity_id_map.get(relation.subject)
        target_id = entity_id_map.get(relation.object)

        if not (source_id and target_id):
            logger.warning(f"关系实体未找到: {relation.subject} -> {relation.object}")
            return

        properties = {
            "confidence": relation.confidence,
            "evidence": relation.evidence,
            "direction": relation.direction.value,
            "source_chunk_id": relation.source_chunk_id,
            "metadata": relation.metadata or {}
        }

        try:
            await self.graph_client.create_relationship(
                source_id,
                target_id,
                relation.relation_type.value,
                properties
            )
        except Neo4jClientError as e:
            logger.warning(f"创建关系失败: {e}")

    def _cluster_to_dict(
        self,
        cluster: EntityCluster,
        node_id: Optional[str]
    ) -> Dict:
        """转换实体簇为字典"""
        return {
            "canonical_id": cluster.canonical_id,
            "name": cluster.canonical_name,
            "type": cluster.type.value,
            "confidence": cluster.confidence,
            "mention_count": cluster.mention_count,
            "aliases": list(cluster.aliases)
        }

    def _relation_to_dict(self, relation: ExtractedRelation) -> Dict:
        """转换关系为字典"""
        return {
            "id": relation.id,
            "subject": relation.subject,
            "object": relation.object,
            "relation_type": relation.relation_type.value,
            "direction": relation.direction.value,
            "confidence": relation.confidence,
            "evidence": relation.evidence
        }

    async def query_entity_network(
        self,
        entity_name: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        查询实体关系网络

        Args:
            entity_name: 实体名称
            max_depth: 最大深度

        Returns:
            网络数据
        """
        try:
            # 查找实体
            entity = await self.graph_client.find_entity_by_name(entity_name)
            if not entity:
                return {"error": f"实体未找到: {entity_name}"}

            entity_id = entity["canonical_id"]

            # 查询关系网络（简化实现）
            # TODO: 使用Cypher查询获取完整网络

            return {
                "center_entity": entity,
                "nodes": [entity],
                "relations": [],
                "depth": max_depth
            }

        except Exception as e:
            logger.error(f"查询实体网络失败: {e}")
            return {"error": str(e)}

    async def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return await self.graph_client.get_statistics()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            stats = await self.get_statistics()
            schema_valid = await self._verify_schema()

            return {
                "status": "healthy" if self._initialized else "uninitialized",
                "schema_valid": schema_valid,
                "total_nodes": stats.get("total_nodes", 0),
                "total_relations": stats.get("total_relationships", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def _verify_schema(self) -> bool:
        """验证schema"""
        try:
            schema_initializer = GraphSchemaInitializer(self.graph_client.driver)
            result = await schema_initializer.verify_schema()
            return result.get("valid", False)
        except Exception:
            return False

    async def close(self):
        """关闭服务"""
        await self.graph_client.disconnect()
        self._initialized = False
        logger.info("统一知识图谱服务已关闭")


# 全局实例
unified_kg_service = UnifiedKnowledgeGraphService()


# 便捷函数
async def get_knowledge_graph_service(
    force_reinit: bool = False
) -> UnifiedKnowledgeGraphService:
    """
    获取统一知识图谱服务实例

    Args:
        force_reinit: 是否强制重新初始化

    Returns:
        UnifiedKnowledgeGraphService 实例
    """
    if not unified_kg_service._initialized or force_reinit:
        await unified_kg_service.initialize()
    return unified_kg_service
