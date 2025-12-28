"""
知识图谱构建任务
使用GraphRAG技术构建金融领域知识图谱
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

from celery import current_task
from celery.exceptions import Retry

from app.celery_app import celery_app
from app.services.graphrag_service import GraphRAGService
from app.services.neo4j_service import Neo4jService
from app.services.entity_linking_service import EntityLinkingService
from app.core.config import settings
from app.db.mysql import get_db
from app.models.document import Document, DocumentChunk, KnowledgeGraphNode, KnowledgeGraphRelation

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,)
)
def build_knowledge_graph(self, document_id: str) -> Dict[str, Any]:
    """
    构建知识图谱任务
    """
    try:
        logger.info(f"Starting knowledge graph construction for document_id: {document_id}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Initializing knowledge graph construction"}
        )

        # 异步执行知识图谱构建
        result = asyncio.run(_build_knowledge_graph_async(document_id, self))

        logger.info(f"Knowledge graph construction completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Knowledge graph construction failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=60)


async def _build_knowledge_graph_async(document_id: str, task) -> Dict[str, Any]:
    """异步构建知识图谱"""
    graphrag_service = GraphRAGService()
    neo4j_service = Neo4jService()
    entity_linking_service = EntityLinkingService()

    # 获取文档块
    async with get_db() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = result.scalars().all()

    if not chunks:
        raise ValueError(f"No chunks found for document {document_id}")

    # 实体抽取和链接
    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Extracting and linking entities"}
    )

    all_entities = []
    all_relations = []
    chunk_entities = {}  # chunk_id -> entities
    chunk_relations = {}  # chunk_id -> relations

    for i, chunk in enumerate(chunks):
        task.update_state(
            state="PROGRESS",
            meta={
                "current": 20 + (i / len(chunks)) * 30,
                "total": 100,
                "status": f"Processing chunk {i+1}/{len(chunks)}"
            }
        )

        # 抽取实体
        entities = await graphrag_service.extract_entities(chunk.content)

        # 链接实体
        linked_entities = await entity_linking_service.link_entities(entities)

        # 抽取关系
        relations = await graphrag_service.extract_relations(chunk.content, linked_entities)

        chunk_entities[chunk.id] = linked_entities
        chunk_relations[chunk.id] = relations

        all_entities.extend(linked_entities)
        all_relations.extend(relations)

    # 去重和合并实体
    task.update_state(
        state="PROGRESS",
        meta={"current": 60, "total": 100, "status": "Merging and deduplicating entities"}
    )

    unique_entities = await _merge_entities(all_entities)
    unique_relations = await _merge_relations(all_relations, unique_entities)

    # 构建图谱结构
    task.update_state(
        state="PROGRESS",
        meta={"current": 75, "total": 100, "status": "Building graph structure"}
    )

    graph_structure = await graphrag_service.build_graph_structure(
        unique_entities,
        unique_relations,
        document_id
    )

    # 存储到Neo4j
    task.update_state(
        state="PROGRESS",
        meta={"current": 85, "total": 100, "status": "Storing to Neo4j"}
    )

    neo4j_results = await neo4j_service.insert_graph_data(
        document_id,
        graph_structure["nodes"],
        graph_structure["relationships"]
    )

    # 保存元数据
    task.update_state(
        state="PROGRESS",
        meta={"current": 95, "total": 100, "status": "Saving metadata"}
    )

    await _save_knowledge_graph_metadata(
        document_id,
        unique_entities,
        unique_relations,
        neo4j_results
    )

    return {
        "status": "completed",
        "document_id": document_id,
        "entities_count": len(unique_entities),
        "relations_count": len(unique_relations),
        "neo4j_nodes": neo4j_results.get("nodes_created", 0),
        "neo4j_relationships": neo4j_results.get("relationships_created", 0),
        "construction_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def update_knowledge_graph(self, document_id: str, chunk_ids: List[str]) -> Dict[str, Any]:
    """
    更新知识图谱（增量更新）
    """
    try:
        logger.info(f"Updating knowledge graph for document_id: {document_id}, chunks: {len(chunk_ids)}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Updating knowledge graph"}
        )

        # 异步执行更新
        result = asyncio.run(_update_knowledge_graph_async(document_id, chunk_ids, self))

        logger.info(f"Knowledge graph update completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Knowledge graph update failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _update_knowledge_graph_async(document_id: str, chunk_ids: List[str], task) -> Dict[str, Any]:
    """异步更新知识图谱"""
    graphrag_service = GraphRAGService()
    neo4j_service = Neo4jService()
    entity_linking_service = EntityLinkingService()

    # 获取要更新的块
    async with get_db() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.id.in_(chunk_ids)
            )
        )
        chunks = result.scalars().all()

    if not chunks:
        raise ValueError(f"No chunks found for update in document {document_id}")

    # 重新抽取实体和关系
    task.update_state(
        state="PROGRESS",
        meta={"current": 30, "total": 100, "status": "Re-extracting entities and relations"}
    )

    new_entities = []
    new_relations = []

    for chunk in chunks:
        entities = await graphrag_service.extract_entities(chunk.content)
        linked_entities = await entity_linking_service.link_entities(entities)
        relations = await graphrag_service.extract_relations(chunk.content, linked_entities)

        new_entities.extend(linked_entities)
        new_relations.extend(relations)

    # 合并到现有图谱
    task.update_state(
        state="PROGRESS",
        meta={"current": 70, "total": 100, "status": "Merging with existing graph"}
    )

    # 获取现有实体
    existing_entities = await neo4j_service.get_document_entities(document_id)

    # 合并实体
    merged_entities = await _merge_with_existing_entities(new_entities, existing_entities)

    # 更新Neo4j
    update_results = await neo4j_service.update_graph_data(
        document_id,
        merged_entities["new_nodes"],
        merged_entities["updated_nodes"],
        new_relations
    )

    return {
        "status": "completed",
        "document_id": document_id,
        "updated_chunks": len(chunks),
        "new_entities": len(merged_entities["new_nodes"]),
        "updated_entities": len(merged_entities["updated_nodes"]),
        "new_relations": len(new_relations),
        "update_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def delete_document_knowledge_graph(self, document_id: str) -> Dict[str, Any]:
    """
    删除文档的知识图谱
    """
    try:
        logger.info(f"Deleting knowledge graph for document_id: {document_id}")

        # 异步执行删除
        result = asyncio.run(_delete_knowledge_graph_async(document_id))

        logger.info(f"Knowledge graph deletion completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Knowledge graph deletion failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _delete_knowledge_graph_async(document_id: str) -> Dict[str, Any]:
    """异步删除知识图谱"""
    neo4j_service = Neo4jService()

    # 从Neo4j删除
    deleted_stats = await neo4j_service.delete_document_graph(document_id)

    # 从数据库删除记录
    async with get_db() as db:
        from sqlalchemy import delete
        await db.execute(
            delete(KnowledgeGraphNode).where(KnowledgeGraphNode.document_id == document_id)
        )
        await db.execute(
            delete(KnowledgeGraphRelation).where(KnowledgeGraphRelation.document_id == document_id)
        )
        await db.commit()

    return {
        "status": "completed",
        "document_id": document_id,
        "deleted_nodes": deleted_stats.get("nodes_deleted", 0),
        "deleted_relationships": deleted_stats.get("relationships_deleted", 0),
        "deletion_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def enrich_knowledge_graph(self, document_id: str) -> Dict[str, Any]:
    """
    知识图谱增强（添加外部知识）
    """
    try:
        logger.info(f"Enriching knowledge graph for document_id: {document_id}")

        # 更新任务状态
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Enriching knowledge graph"}
        )

        # 异步执行增强
        result = asyncio.run(_enrich_knowledge_graph_async(document_id, self))

        logger.info(f"Knowledge graph enrichment completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Knowledge graph enrichment failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _enrich_knowledge_graph_async(document_id: str, task) -> Dict[str, Any]:
    """异步增强知识图谱"""
    neo4j_service = Neo4jService()
    graphrag_service = GraphRAGService()

    # 获取文档中的实体
    task.update_state(
        state="PROGRESS",
        meta={"current": 20, "total": 100, "status": "Loading document entities"}
    )

    entities = await neo4j_service.get_document_entities(document_id)

    # 增强公司实体
    task.update_state(
        state="PROGRESS",
        meta={"current": 40, "total": 100, "status": "Enriching company entities"}
    )

    company_entities = [e for e in entities if e.get("type") == "company"]
    enriched_companies = await graphrag_service.enrich_company_entities(company_entities)

    # 增强人物实体
    task.update_state(
        state="PROGRESS",
        meta={"current": 60, "total": 100, "status": "Enriching person entities"}
    )

    person_entities = [e for e in entities if e.get("type") == "person"]
    enriched_persons = await graphrag_service.enrich_person_entities(person_entities)

    # 增强金融概念实体
    task.update_state(
        state="PROGRESS",
        meta={"current": 80, "total": 100, "status": "Enriching concept entities"}
    )

    concept_entities = [e for e in entities if e.get("type") == "concept"]
    enriched_concepts = await graphrag_service.enrich_concept_entities(concept_entities)

    # 更新Neo4j
    task.update_state(
        state="PROGRESS",
        meta={"current": 90, "total": 100, "status": "Updating enriched entities"}
    )

    all_enriched = enriched_companies + enriched_persons + enriched_concepts
    enrichment_results = await neo4j_service.update_entity_properties(document_id, all_enriched)

    return {
        "status": "completed",
        "document_id": document_id,
        "enriched_companies": len(enriched_companies),
        "enriched_persons": len(enriched_persons),
        "enriched_concepts": len(enriched_concepts),
        "total_enriched": len(all_enriched),
        "enrichment_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def calculate_graph_metrics(self, document_id: str) -> Dict[str, Any]:
    """
    计算知识图谱指标
    """
    try:
        logger.info(f"Calculating graph metrics for document_id: {document_id}")

        # 异步执行计算
        result = asyncio.run(_calculate_graph_metrics_async(document_id, self))

        logger.info(f"Graph metrics calculation completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Graph metrics calculation failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _calculate_graph_metrics_async(document_id: str, task) -> Dict[str, Any]:
    """异步计算图谱指标"""
    neo4j_service = Neo4jService()

    # 计算基础指标
    task.update_state(
        state="PROGRESS",
        meta={"current": 30, "total": 100, "status": "Calculating basic metrics"}
    )

    basic_metrics = await neo4j_service.calculate_basic_metrics(document_id)

    # 计算中心性指标
    task.update_state(
        state="PROGRESS",
        meta={"current": 60, "total": 100, "status": "Calculating centrality metrics"}
    )

    centrality_metrics = await neo4j_service.calculate_centrality_metrics(document_id)

    # 计算社区结构
    task.update_state(
        state="PROGRESS",
        meta={"current": 85, "total": 100, "status": "Detecting communities"}
    )

    community_metrics = await neo4j_service.detect_communities(document_id)

    # 保存指标
    await _save_graph_metrics(document_id, basic_metrics, centrality_metrics, community_metrics)

    return {
        "status": "completed",
        "document_id": document_id,
        "basic_metrics": basic_metrics,
        "centrality_metrics": centrality_metrics,
        "community_metrics": community_metrics,
        "calculation_time": datetime.utcnow().isoformat()
    }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=30
)
def find_graph_patterns(self, document_id: str) -> Dict[str, Any]:
    """
    发现图谱模式
    """
    try:
        logger.info(f"Finding graph patterns for document_id: {document_id}")

        # 异步执行模式发现
        result = asyncio.run(_find_graph_patterns_async(document_id, self))

        logger.info(f"Graph patterns discovery completed for document_id: {document_id}")
        return result

    except Exception as e:
        logger.error(f"Graph patterns discovery failed for document_id: {document_id}, error: {e}")
        raise self.retry(exc=e, countdown=30)


async def _find_graph_patterns_async(document_id: str, task) -> Dict[str, Any]:
    """异步发现图谱模式"""
    neo4j_service = Neo4jService()
    graphrag_service = GraphRAGService()

    # 发现投资关系模式
    task.update_state(
        state="PROGRESS",
        meta={"current": 25, "total": 100, "status": "Finding investment patterns"}
    )

    investment_patterns = await neo4j_service.find_investment_patterns(document_id)

    # 发现人事变动模式
    task.update_state(
        state="PROGRESS",
        meta={"current": 50, "total": 100, "status": "Finding personnel patterns"}
    )

    personnel_patterns = await neo4j_service.find_personnel_patterns(document_id)

    # 发现业务关联模式
    task.update_state(
        state="PROGRESS",
        meta={"current": 75, "total": 100, "status": "Finding business patterns"}
    )

    business_patterns = await neo4j_service.find_business_patterns(document_id)

    # 分析模式意义
    task.update_state(
        state="PROGRESS",
        meta={"current": 90, "total": 100, "status": "Analyzing pattern significance"}
    )

    pattern_insights = await graphrag_service.analyze_pattern_significance(
        investment_patterns,
        personnel_patterns,
        business_patterns
    )

    return {
        "status": "completed",
        "document_id": document_id,
        "investment_patterns": investment_patterns,
        "personnel_patterns": personnel_patterns,
        "business_patterns": business_patterns,
        "pattern_insights": pattern_insights,
        "discovery_time": datetime.utcnow().isoformat()
    }


# 辅助函数
async def _merge_entities(entities: List[Dict]) -> List[Dict]:
    """合并和去重实体"""
    entity_map = {}  # name -> entity

    for entity in entities:
        name = entity.get("name", "").lower().strip()
        if not name:
            continue

        if name in entity_map:
            # 合并实体信息
            existing = entity_map[name]
            # 合并别名
            aliases = set(existing.get("aliases", []))
            aliases.add(entity.get("name", ""))
            if "aliases" in entity:
                aliases.update(entity["aliases"])
            existing["aliases"] = list(aliases)

            # 合并其他属性
            for key, value in entity.items():
                if key not in existing and value:
                    existing[key] = value
        else:
            entity_map[name] = entity

    return list(entity_map.values())


async def _merge_relations(relations: List[Dict], entities: List[Dict]) -> List[Dict]:
    """合并和去重关系"""
    entity_names = {entity.get("name", ""): entity for entity in entities}
    relation_map = {}  # (source, target, type) -> relation

    for relation in relations:
        source = relation.get("source", "")
        target = relation.get("target", "")
        rel_type = relation.get("type", "")

        if not all([source, target, rel_type]):
            continue

        key = (source.lower(), target.lower(), rel_type.lower())
        if key in relation_map:
            # 合并关系信息
            existing = relation_map[key]
            # 合并置信度（取最高）
            if "confidence" in relation and relation["confidence"] > existing.get("confidence", 0):
                existing["confidence"] = relation["confidence"]
        else:
            relation_map[key] = relation

    return list(relation_map.values())


async def _merge_with_existing_entities(
    new_entities: List[Dict],
    existing_entities: List[Dict]
) -> Dict[str, List[Dict]]:
    """与现有实体合并"""
    existing_map = {entity.get("name", "").lower(): entity for entity in existing_entities}

    new_nodes = []
    updated_nodes = []

    for entity in new_entities:
        name = entity.get("name", "").lower()
        if name in existing_map:
            # 更新现有实体
            existing = existing_map[name]
            # 合并属性
            for key, value in entity.items():
                if key not in existing and value:
                    existing[key] = value
            updated_nodes.append(existing)
        else:
            # 新实体
            new_nodes.append(entity)

    return {
        "new_nodes": new_nodes,
        "updated_nodes": updated_nodes
    }


async def _save_knowledge_graph_metadata(
    document_id: str,
    entities: List[Dict],
    relations: List[Dict],
    neo4j_results: Dict[str, Any]
):
    """保存知识图谱元数据"""
    async with get_db() as db:
        # 保存实体记录
        for entity in entities:
            kg_node = KnowledgeGraphNode(
                document_id=document_id,
                neo4j_id=entity.get("neo4j_id"),
                node_type=entity.get("type", "unknown"),
                label=entity.get("name", ""),
                properties=entity,
                created_at=datetime.utcnow()
            )
            db.add(kg_node)

        # 保存关系记录
        for relation in relations:
            kg_relation = KnowledgeGraphRelation(
                document_id=document_id,
                neo4j_id=relation.get("neo4j_id"),
                relation_type=relation.get("type", "unknown"),
                source_node_id=relation.get("source_id"),
                target_node_id=relation.get("target_id"),
                properties=relation,
                created_at=datetime.utcnow()
            )
            db.add(kg_relation)

        # 更新文档记录
        document = await db.get(Document, document_id)
        if document:
            document.doc_metadata = {
                **document.doc_metadata,
                "knowledge_graph": {
                    "entities_count": len(entities),
                    "relations_count": len(relations),
                    "neo4j_nodes": neo4j_results.get("nodes_created", 0),
                    "neo4j_relationships": neo4j_results.get("relationships_created", 0),
                    "constructed_at": datetime.utcnow().isoformat()
                }
            }
            await db.commit()


async def _save_graph_metrics(
    document_id: str,
    basic_metrics: Dict,
    centrality_metrics: Dict,
    community_metrics: Dict
):
    """保存图谱指标"""
    async with get_db() as db:
        document = await db.get(Document, document_id)
        if document:
            document.doc_metadata = {
                **document.doc_metadata,
                "graph_metrics": {
                    "basic": basic_metrics,
                    "centrality": centrality_metrics,
                    "community": community_metrics,
                    "calculated_at": datetime.utcnow().isoformat()
                }
            }
            await db.commit()