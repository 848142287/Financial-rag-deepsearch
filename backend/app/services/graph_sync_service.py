"""
图谱库同步服务
负责将实体关系数据同步到图数据库
"""

import asyncio
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models.document import DocumentChunk
from app.models.synchronization import DocumentSync, GraphSync, EntityLink, SyncStatus, SyncLog
from app.services.sync_state_machine import SyncStateMachine

logger = logging.getLogger(__name__)


class GraphDBClient:
    """图数据库客户端接口"""

    def __init__(self, config: Dict):
        self.config = config
        self.client = None

    async def connect(self):
        """连接到图数据库"""
        # 这里根据实际使用的图数据库实现
        # 例如: Neo4j, ArangoDB, Amazon Neptune等
        pass

    async def create_constraints(self):
        """创建约束和索引"""
        pass

    async def create_node(self, node_type: str, properties: Dict) -> str:
        """创建节点"""
        pass

    async def create_relationship(
        self, source_id: str, target_id: str, rel_type: str, properties: Dict
    ) -> str:
        """创建关系"""
        pass

    async def update_node(self, node_id: str, properties: Dict):
        """更新节点"""
        pass

    async def update_relationship(self, rel_id: str, properties: Dict):
        """更新关系"""
        pass

    async def delete_node(self, node_id: str):
        """删除节点"""
        pass

    async def delete_relationship(self, rel_id: str):
        """删除关系"""
        pass

    async def query_nodes(self, node_type: str, properties: Dict) -> List[Dict]:
        """查询节点"""
        pass

    async def query_relationships(
        self, rel_type: Optional[str] = None, properties: Dict = None
    ) -> List[Dict]:
        """查询关系"""
        pass

    async def find_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Dict]:
        """根据名称查找实体"""
        pass


class EntityExtractor:
    """实体抽取器"""

    def __init__(self, config: Dict):
        self.config = config
        self.entity_types = config.get("entity_types", [
            "company", "person", "financial_product", "concept", "numeric_entity"
        ])
        self.extraction_rules = config.get("extraction_rules", {})

        # 金融实体模式
        self.entity_patterns = {
            "company": [
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Group|Holdings)))",
                r"([^a-z]{2,}(?:\s+[^a-z]{2,})*)",
            ],
            "person": [
                r"([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"(Mr|Mrs|Ms|Dr|Prof)\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            ],
            "financial_product": [
                r"([A-Za-z]+\s+(?:Bond|Stock|ETF|Fund|Option|Future|Derivative))",
                r"([A-Z]{2,6}\s*(?:Index|Future|Option))",
            ],
            "numeric_entity": [
                r"(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand)?)",
                r"(\d{1,2}%|\d+\.\d+%)",
                r"(\d{4}年\d{1,2}月\d{1,2}日)",
            ],
            "concept": [
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Rate|Index|Market|Exchange|Policy))",
                r"([^a-z]*(?:GDP|CPI|PPI|PMI|EPS|P/E|ROE|ROA))",
            ]
        }

        # 关系模式
        self.relation_patterns = [
            (r"(.+?)\s+(?:owns|holds|acquires?)\s+(.+?)", "OWNS"),
            (r"(.+?)\s+(?:invests?\s+in|funds?)\s+(.+?)", "INVESTS_IN"),
            (r"(.+?)\s+(?:partners?\s+with|collaborates?\s+with)\s+(.+?)", "PARTNERS_WITH"),
            (r"(.+?)\s+(?:is\s+(?:a|an)|belongs\s+to)\s+(.+?)", "IS_TYPE_OF"),
            (r"(.+?)\s+(?:reports?\s+to|subsidiary\s+of)\s+(.+?)", "SUBSIDIARY_OF"),
            (r"(.+?)\s+(?:exceeds?|above|below|less\s+than)\s+(.+?)", "COMPARES_TO"),
        ]

    async def extract_entities(self, text: str) -> List[Dict]:
        """提取实体"""
        entities = []

        # 按类型提取实体
        for entity_type, patterns in self.entity_patterns.items():
            if entity_type not in self.entity_types:
                continue

            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(1).strip()
                    if len(entity_name) >= 2:  # 过滤过短的匹配
                        entities.append({
                            "name": entity_name,
                            "type": entity_type,
                            "position": match.span(),
                            "confidence": self._calculate_confidence(entity_name, entity_type)
                        })

        # 实体消歧和归一化
        entities = await self._disambiguate_entities(entities)

        # 去重
        unique_entities = {}
        for entity in entities:
            key = (entity["name"].lower(), entity["type"])
            if key not in unique_entities or entity["confidence"] > unique_entities[key]["confidence"]:
                unique_entities[key] = entity

        return list(unique_entities.values())

    async def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """提取关系"""
        relationships = []
        entity_names = {e["name"].lower() for e in entities}

        for pattern, rel_type in self.relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_name = match.group(1).strip()
                target_name = match.group(2).strip()

                # 检查是否都是已知实体
                if (source_name.lower() in entity_names and
                    target_name.lower() in entity_names):

                    relationships.append({
                        "source": source_name,
                        "target": target_name,
                        "type": rel_type,
                        "position": match.span(),
                        "confidence": self._calculate_relation_confidence(source_name, target_name, rel_type)
                    })

        return relationships

    async def _disambiguate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体消歧"""
        # 公司简称归一化
        company_normalization = self.extraction_rules.get("company_normalization", True)
        if company_normalization:
            company_mapping = {
                "腾讯": "腾讯控股",
                "阿里": "阿里巴巴",
                "百度": "百度公司",
                "京东": "京东集团",
                "美团": "美团点评",
                "滴滴": "滴滴出行",
            }

            for entity in entities:
                if entity["type"] == "company":
                    short_name = entity["name"]
                    if short_name in company_mapping:
                        entity["name"] = company_mapping[short_name]
                        entity["original_name"] = short_name

        return entities

    def _calculate_confidence(self, entity_name: str, entity_type: str) -> float:
        """计算实体置信度"""
        confidence = 0.5  # 基础置信度

        # 根据实体类型调整
        if entity_type == "company":
            if any(keyword in entity_name for keyword in ["公司", "集团", "控股", "Inc", "Corp"]):
                confidence += 0.2
        elif entity_type == "person":
            if len(entity_name.split()) >= 2:
                confidence += 0.2
        elif entity_type == "numeric_entity":
            confidence += 0.3

        # 根据长度调整
        if len(entity_name) > 5:
            confidence += 0.1
        elif len(entity_name) < 2:
            confidence -= 0.3

        return min(1.0, max(0.0, confidence))

    def _calculate_relation_confidence(self, source: str, target: str, rel_type: str) -> float:
        """计算关系置信度"""
        confidence = 0.6  # 基础置信度

        # 根据关系类型调整
        if rel_type in ["OWNS", "SUBSIDIARY_OF"]:
            confidence += 0.2
        elif rel_type in ["INVESTS_IN", "PARTNERS_WITH"]:
            confidence += 0.1

        return min(1.0, confidence)


class GraphSyncService:
    """图谱同步服务"""

    def __init__(self, db: Session, config: Dict):
        self.db = db
        self.config = config
        self.state_machine = SyncStateMachine(db)
        self.graph_client = GraphDBClient(config)
        self.entity_extractor = EntityExtractor(config)

    async def sync_document(self, document_sync_id: int) -> bool:
        """
        同步文档到图数据库

        Args:
            document_sync_id: 文档同步ID

        Returns:
            bool: 是否同步成功
        """
        try:
            # 获取文档同步记录
            document_sync = self.db.query(DocumentSync).filter(
                DocumentSync.id == document_sync_id
            ).first()

            if not document_sync:
                logger.error(f"DocumentSync {document_sync_id} not found")
                return False

            # 获取文档分块
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_sync.document_id
            ).order_by(DocumentChunk.chunk_index).all()

            if not chunks:
                logger.error(f"No chunks found for document {document_sync.document_id}")
                await self.state_machine.transition_state(
                    document_sync_id, SyncStatus.FAILED,
                    "No document chunks found"
                )
                return False

            # 初始化图数据库
            await self.graph_client.connect()
            await self.graph_client.create_constraints()

            total_chunks = len(chunks)
            processed_chunks = 0
            all_entities = {}
            all_relationships = []

            # 处理每个分块
            for chunk in chunks:
                try:
                    # 抽取实体和关系
                    entities = await self.entity_extractor.extract_entities(chunk.content)
                    relationships = await self.entity_extractor.extract_relationships(
                        chunk.content, entities
                    )

                    # 合并实体
                    for entity in entities:
                        key = (entity["name"].lower(), entity["type"])
                        if key not in all_entities:
                            all_entities[key] = {
                                "name": entity["name"],
                                "type": entity["type"],
                                "confidence": entity["confidence"],
                                "chunks": []
                            }
                        all_entities[key]["chunks"].append(chunk.id)

                    # 保存关系
                    for rel in relationships:
                        rel["chunk_id"] = chunk.id
                        all_relationships.append(rel)

                    processed_chunks += 1

                    # 更新进度
                    progress = (processed_chunks / total_chunks) * 50  # 实体抽取占50%
                    await self.state_machine.update_progress(
                        document_sync_id,
                        graph_progress=progress,
                        processed_chunks=processed_chunks
                    )

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.id}: {str(e)}")
                    await self._log_sync_error(
                        document_sync_id, "entity_extraction",
                        f"Error processing chunk {chunk.id}: {str(e)}",
                        {"chunk_id": chunk.id, "error": str(e)}
                    )

            # 同步实体到图数据库
            entity_count = len(all_entities)
            synced_entities = 0

            for entity_key, entity_data in all_entities.items():
                try:
                    # 检查实体是否已存在
                    existing_entity = await self.graph_client.find_entity_by_name(
                        entity_data["name"], entity_data["type"]
                    )

                    if existing_entity:
                        entity_id = existing_entity["id"]
                        # 更新实体信息
                        update_data = {
                            "confidence": max(entity_data["confidence"], existing_entity.get("confidence", 0)),
                            "chunk_count": existing_entity.get("chunk_count", 0) + len(entity_data["chunks"]),
                            "updated_at": datetime.utcnow().isoformat()
                        }
                        await self.graph_client.update_node(entity_id, update_data)
                    else:
                        # 创建新实体
                        entity_id = await self.graph_client.create_node(
                            entity_data["type"], {
                                "name": entity_data["name"],
                                "confidence": entity_data["confidence"],
                                "chunk_count": len(entity_data["chunks"]),
                                "document_id": document_sync.document_id,
                                "created_at": datetime.utcnow().isoformat()
                            }
                        )

                    # 创建图谱同步记录
                    for chunk_id in entity_data["chunks"]:
                        graph_sync = GraphSync(
                            document_sync_id=document_sync_id,
                            chunk_id=chunk_id,
                            entity_id=entity_id,
                            entity_type=entity_data["type"],
                            entity_name=entity_data["name"],
                            sync_status=SyncStatus.COMPLETED,
                            synced_at=datetime.utcnow(),
                            metadata={
                                "confidence": entity_data["confidence"],
                                "created_at": datetime.utcnow().isoformat()
                            }
                        )
                        self.db.add(graph_sync)

                    synced_entities += 1

                    # 更新进度
                    progress = 50 + (synced_entities / entity_count * 30)  # 实体同步占30%
                    await self.state_machine.update_progress(
                        document_sync_id, graph_progress=progress
                    )

                except Exception as e:
                    logger.error(f"Error syncing entity {entity_data['name']}: {str(e)}")
                    await self._log_sync_error(
                        document_sync_id, "entity_sync",
                        f"Error syncing entity {entity_data['name']}: {str(e)}",
                        {"entity": entity_data, "error": str(e)}
                    )

            # 同步关系到图数据库
            rel_count = len(all_relationships)
            synced_rels = 0

            for rel_data in all_relationships:
                try:
                    # 查找源和目标实体
                    source_entity = await self.graph_client.find_entity_by_name(rel_data["source"])
                    target_entity = await self.graph_client.find_entity_by_name(rel_data["target"])

                    if source_entity and target_entity:
                        # 创建关系
                        rel_id = await self.graph_client.create_relationship(
                            source_entity["id"], target_entity["id"], rel_data["type"], {
                                "confidence": rel_data["confidence"],
                                "document_id": document_sync.document_id,
                                "chunk_id": rel_data["chunk_id"],
                                "created_at": datetime.utcnow().isoformat()
                            }
                        )

                        # 创建图谱同步记录
                        graph_sync = GraphSync(
                            document_sync_id=document_sync_id,
                            chunk_id=rel_data["chunk_id"],
                            relation_id=rel_id,
                            relation_type=rel_data["type"],
                            source_entity=rel_data["source"],
                            target_entity=rel_data["target"],
                            sync_status=SyncStatus.COMPLETED,
                            synced_at=datetime.utcnow(),
                            metadata={
                                "confidence": rel_data["confidence"],
                                "created_at": datetime.utcnow().isoformat()
                            }
                        )
                        self.db.add(graph_sync)

                    synced_rels += 1

                    # 更新进度
                    progress = 80 + (synced_rels / rel_count * 20)  # 关系同步占20%
                    await self.state_machine.update_progress(
                        document_sync_id, graph_progress=progress
                    )

                except Exception as e:
                    logger.error(f"Error syncing relationship {rel_data}: {str(e)}")
                    await self._log_sync_error(
                        document_sync_id, "relationship_sync",
                        f"Error syncing relationship: {str(e)}",
                        {"relationship": rel_data, "error": str(e)}
                    )

            self.db.commit()

            # 图谱同步完成，转换到关联建立状态
            await self.state_machine.transition_state(
                document_sync_id, SyncStatus.LINK_ING,
                "Graph synchronization completed"
            )

            logger.info(f"Graph synchronization completed for document {document_sync.document_id}")
            return True

        except Exception as e:
            logger.error(f"Error in graph sync for document_sync {document_sync_id}: {str(e)}")

            # 更新同步状态为失败
            await self.state_machine.transition_state(
                document_sync_id, SyncStatus.FAILED,
                f"Graph synchronization failed: {str(e)}"
            )

            await self._log_sync_error(
                document_sync_id, "graph_sync",
                f"Graph synchronization failed: {str(e)}",
                {"error_details": str(e)}
            )

            return False

    async def update_entity(self, entity_id: str, new_data: Dict) -> bool:
        """
        更新实体

        Args:
            entity_id: 实体ID
            new_data: 新数据

        Returns:
            bool: 是否更新成功
        """
        try:
            # 更新图数据库中的实体
            await self.graph_client.update_node(entity_id, {
                **new_data,
                "updated_at": datetime.utcnow().isoformat()
            })

            # 更新数据库中的同步记录
            graph_syncs = self.db.query(GraphSync).filter(
                GraphSync.entity_id == entity_id
            ).all()

            for gs in graph_syncs:
                gs.synced_at = datetime.utcnow()
                gs.metadata = {
                    **(gs.metadata or {}),
                    "updated_at": datetime.utcnow().isoformat()
                }

            self.db.commit()

            logger.info(f"Successfully updated entity {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating entity {entity_id}: {str(e)}")
            return False

    async def delete_document_graph_data(self, document_id: int) -> bool:
        """
        删除文档的所有图数据

        Args:
            document_id: 文档ID

        Returns:
            bool: 是否删除成功
        """
        try:
            # 获取文档的所有图谱同步记录
            graph_syncs = self.db.query(GraphSync).join(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()

            # 删除关系
            relation_ids = list(set([gs.relation_id for gs in graph_syncs if gs.relation_id]))
            for rel_id in relation_ids:
                await self.graph_client.delete_relationship(rel_id)

            # 删除实体（只删除仅属于此文档的实体）
            entity_ids = list(set([gs.entity_id for gs in graph_syncs if gs.entity_id]))
            for entity_id in entity_ids:
                # 检查实体是否还被其他文档使用
                other_refs = self.db.query(GraphSync).filter(
                    and_(
                        GraphSync.entity_id == entity_id,
                        GraphSync.chunk_id.notin_([gs.chunk_id for gs in graph_syncs])
                    )
                ).count()

                if other_refs == 0:
                    await self.graph_client.delete_node(entity_id)

            # 删除同步记录
            self.db.query(GraphSync).filter(
                GraphSync.id.in_([gs.id for gs in graph_syncs])
            ).delete(synchronize_session=False)

            self.db.commit()

            logger.info(f"Successfully deleted graph data for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting graph data for document {document_id}: {str(e)}")
            return False

    async def query_entity_network(
        self, entity_name: str, depth: int = 2, entity_types: Optional[List[str]] = None
    ) -> Dict:
        """
        查询实体网络

        Args:
            entity_name: 实体名称
            depth: 查询深度
            entity_types: 限制的实体类型

        Returns:
            Dict: 实体网络数据
        """
        try:
            # 查找起始实体
            start_entity = await self.graph_client.find_entity_by_name(entity_name)
            if not start_entity:
                return {}

            # 构建查询（根据具体的图数据库实现）
            network = {
                "nodes": [start_entity],
                "relationships": []
            }

            # 这里实现具体的网络查询逻辑
            # ...

            return network

        except Exception as e:
            logger.error(f"Error querying entity network for {entity_name}: {str(e)}")
            return {}

    async def get_sync_statistics(self, document_id: Optional[int] = None) -> Dict:
        """
        获取同步统计信息

        Args:
            document_id: 文档ID（可选）

        Returns:
            Dict: 统计信息
        """
        try:
            query = self.db.query(GraphSync).join(DocumentChunk)

            if document_id:
                query = query.filter(DocumentChunk.document_id == document_id)

            # 统计实体
            entity_stats = query.filter(GraphSync.entity_id.isnot(None)).all()
            total_entities = len(set([gs.entity_id for gs in entity_stats]))
            completed_entities = len([gs for gs in entity_stats if gs.sync_status == SyncStatus.COMPLETED])
            failed_entities = len([gs for gs in entity_stats if gs.sync_status == SyncStatus.FAILED])

            # 统计关系
            relation_stats = query.filter(GraphSync.relation_id.isnot(None)).all()
            total_relations = len(set([gs.relation_id for gs in relation_stats]))
            completed_relations = len([gs for gs in relation_stats if gs.sync_status == SyncStatus.COMPLETED])
            failed_relations = len([gs for gs in relation_stats if gs.sync_status == SyncStatus.FAILED])

            # 按类型统计
            entity_types = {}
            for gs in entity_stats:
                if gs.entity_type:
                    entity_types[gs.entity_type] = entity_types.get(gs.entity_type, 0) + 1

            relation_types = {}
            for gs in relation_stats:
                if gs.relation_type:
                    relation_types[gs.relation_type] = relation_types.get(gs.relation_type, 0) + 1

            statistics = {
                "entities": {
                    "total": total_entities,
                    "completed": completed_entities,
                    "failed": failed_entities,
                    "by_type": entity_types
                },
                "relationships": {
                    "total": total_relations,
                    "completed": completed_relations,
                    "failed": failed_relations,
                    "by_type": relation_types
                }
            }

            return statistics

        except Exception as e:
            logger.error(f"Error getting sync statistics: {str(e)}")
            return {}

    async def _log_sync_error(
        self, document_sync_id: int, component: str, message: str, details: Dict
    ):
        """记录同步错误日志"""
        try:
            log = SyncLog(
                document_sync_id=document_sync_id,
                log_level="ERROR",
                component=component,
                message=message,
                details=details
            )
            self.db.add(log)
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to log sync error: {str(e)}")