"""
实体关联建立服务
负责建立向量库和图谱库之间的实体关联
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models.synchronization import DocumentSync, VectorSync, GraphSync, EntityLink, SyncStatus, SyncLog
from app.services.sync_state_machine import SyncStateMachine

logger = logging.getLogger(__name__)


class EntityMatcher:
    """实体匹配器"""

    def __init__(self, config: Dict):
        self.config = config
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.match_strategies = config.get("match_strategies", ["text_similarity", "context_similarity"])

    async def match_entities(
        self, vector_entities: List[Dict], graph_entities: List[Dict]
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        匹配向量实体和图谱实体

        Args:
            vector_entities: 向量库实体列表
            graph_entities: 图谱库实体列表

        Returns:
            List[Tuple]: 匹配结果 [(vector_entity, graph_entity, similarity_score)]
        """
        matches = []

        for vector_entity in vector_entities:
            best_match = None
            best_score = 0.0

            for graph_entity in graph_entities:
                # 只匹配相同类型的实体
                if vector_entity.get("entity_type") != graph_entity.get("type"):
                    continue

                # 计算相似度
                similarity_score = await self._calculate_similarity(
                    vector_entity, graph_entity
                )

                if similarity_score > best_score and similarity_score >= self.similarity_threshold:
                    best_match = graph_entity
                    best_score = similarity_score

            if best_match:
                matches.append((vector_entity, best_match, best_score))

        # 按相似度排序
        matches.sort(key=lambda x: x[2], reverse=True)

        return matches

    async def _calculate_similarity(
        self, vector_entity: Dict, graph_entity: Dict
    ) -> float:
        """计算实体相似度"""
        similarity = 0.0
        valid_strategies = 0

        # 文本相似度
        if "text_similarity" in self.match_strategies:
            text_sim = self._text_similarity(
                vector_entity.get("name", ""),
                graph_entity.get("name", "")
            )
            similarity += text_sim
            valid_strategies += 1

        # 上下文相似度
        if "context_similarity" in self.match_strategies:
            context_sim = self._context_similarity(
                vector_entity.get("content", ""),
                graph_entity.get("description", "")
            )
            similarity += context_sim
            valid_strategies += 1

        # 类型权重
        if vector_entity.get("entity_type") == graph_entity.get("type"):
            similarity += 0.1
            valid_strategies += 1

        return similarity / valid_strategies if valid_strategies > 0 else 0.0

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0

        # 简单的字符串相似度（可以使用更复杂的算法）
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        if text1 == text2:
            return 1.0
        elif text1 in text2 or text2 in text1:
            return 0.8
        else:
            # 编辑距离相似度
            distance = self._edit_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def _edit_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _context_similarity(self, content1: str, content2: str) -> float:
        """计算上下文相似度"""
        if not content1 or not content2:
            return 0.0

        # 简单的词汇重叠度
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


class CrossModalAssociation:
    """跨模态关联器"""

    def __init__(self, config: Dict):
        self.config = config
        self.enable_vision_model = config.get("enable_vision_model", False)
        self.table_text_association = config.get("table_text_association", True)

    async def associate_table_with_text(
        self, table_data: Dict, text_chunks: List[str]
    ) -> List[Dict]:
        """
        关联表格数据与文本

        Args:
            table_data: 表格数据
            text_chunks: 文本分块列表

        Returns:
            List[Dict]: 关联结果
        """
        associations = []

        if not self.table_text_association:
            return associations

        # 提取表格中的关键词
        table_keywords = self._extract_table_keywords(table_data)

        # 在文本中搜索相关内容
        for i, chunk in enumerate(text_chunks):
            relevance_score = self._calculate_table_text_relevance(
                table_keywords, chunk
            )

            if relevance_score > 0.3:  # 阈值
                associations.append({
                    "chunk_index": i,
                    "relevance_score": relevance_score,
                    "matched_keywords": [kw for kw in table_keywords if kw.lower() in chunk.lower()],
                    "association_type": "table_text"
                })

        return associations

    def _extract_table_keywords(self, table_data: Dict) -> List[str]:
        """从表格数据中提取关键词"""
        keywords = []

        # 从表头提取
        headers = table_data.get("headers", [])
        keywords.extend(headers)

        # 从数据行提取
        rows = table_data.get("rows", [])
        for row in rows[:5]:  # 只取前5行
            keywords.extend([str(cell) for cell in row if str(cell)])

        # 过滤和清理
        keywords = [kw.strip() for kw in keywords if len(kw.strip()) > 2]
        keywords = list(set(keywords))  # 去重

        return keywords[:20]  # 限制数量

    def _calculate_table_text_relevance(
        self, table_keywords: List[str], text: str
    ) -> float:
        """计算表格与文本的相关性"""
        if not table_keywords or not text:
            return 0.0

        text_lower = text.lower()
        matched_keywords = 0

        for keyword in table_keywords:
            if keyword.lower() in text_lower:
                matched_keywords += 1

        return matched_keywords / len(table_keywords)


class EntityLinkService:
    """实体关联服务"""

    def __init__(self, db: Session, config: Dict):
        self.db = db
        self.config = config
        self.state_machine = SyncStateMachine(db)
        self.entity_matcher = EntityMatcher(config)
        self.cross_modal_associator = CrossModalAssociation(config)

    async def establish_links(self, document_sync_id: int) -> bool:
        """
        建立实体关联

        Args:
            document_sync_id: 文档同步ID

        Returns:
            bool: 是否建立成功
        """
        try:
            # 获取文档同步记录
            document_sync = self.db.query(DocumentSync).filter(
                DocumentSync.id == document_sync_id
            ).first()

            if not document_sync:
                logger.error(f"DocumentSync {document_sync_id} not found")
                return False

            # 获取向量同步记录
            vector_syncs = self.db.query(VectorSync).filter(
                VectorSync.document_sync_id == document_sync_id,
                VectorSync.sync_status == SyncStatus.COMPLETED
            ).all()

            # 获取图谱同步记录
            graph_syncs = self.db.query(GraphSync).filter(
                GraphSync.document_sync_id == document_sync_id,
                GraphSync.sync_status == SyncStatus.COMPLETED,
                GraphSync.entity_id.isnot(None)
            ).all()

            if not vector_syncs or not graph_syncs:
                logger.error("No completed vector or graph syncs found")
                await self.state_machine.transition_state(
                    document_sync_id, SyncStatus.FAILED,
                    "No completed vector or graph syncs found"
                )
                return False

            # 准备实体数据
            vector_entities = []
            for vs in vector_syncs:
                # 从向量元数据中提取实体信息
                metadata = vs.metadata or {}
                if metadata.get("entities"):
                    for entity in metadata["entities"]:
                        vector_entities.append({
                            "vector_id": vs.vector_id,
                            "chunk_id": vs.chunk_id,
                            "entity_name": entity.get("name"),
                            "entity_type": entity.get("type"),
                            "confidence": entity.get("confidence", 0.5),
                            "content": metadata.get("content", "")
                        })

            graph_entities = []
            for gs in graph_syncs:
                if gs.entity_id and gs.entity_name:
                    graph_entities.append({
                        "entity_id": gs.entity_id,
                        "entity_name": gs.entity_name,
                        "entity_type": gs.entity_type,
                        "confidence": gs.metadata.get("confidence", 0.5) if gs.metadata else 0.5,
                        "chunk_id": gs.chunk_id
                    })

            if not vector_entities or not graph_entities:
                logger.warning("No entities found in vector or graph data")
                # 没有实体也算成功，只是没有关联
                await self.state_machine.transition_state(
                    document_sync_id, SyncStatus.COMPLETED,
                    "No entities to link"
                )
                return True

            # 匹配实体
            matches = await self.entity_matcher.match_entities(
                vector_entities, graph_entities
            )

            # 创建实体关联记录
            total_matches = len(matches)
            created_links = 0

            for vector_entity, graph_entity, similarity_score in matches:
                try:
                    # 检查是否已存在关联
                    existing_link = self.db.query(EntityLink).filter(
                        and_(
                            EntityLink.document_sync_id == document_sync_id,
                            EntityLink.vector_entity_id == vector_entity["vector_id"],
                            EntityLink.graph_entity_id == graph_entity["entity_id"]
                        )
                    ).first()

                    if existing_link:
                        # 更新现有关联
                        existing_link.confidence_score = max(
                            existing_link.confidence_score, similarity_score
                        )
                        existing_link.linked_at = datetime.utcnow()
                        existing_link.metadata = {
                            **(existing_link.metadata or {}),
                            "last_updated": datetime.utcnow().isoformat(),
                            "similarity_score": similarity_score
                        }
                    else:
                        # 创建新关联
                        entity_link = EntityLink(
                            document_sync_id=document_sync_id,
                            vector_entity_id=vector_entity["vector_id"],
                            graph_entity_id=graph_entity["entity_id"],
                            link_type="entity_match",
                            confidence_score=similarity_score,
                            link_strength=min(similarity_score, 1.0),
                            sync_status=SyncStatus.COMPLETED,
                            linked_at=datetime.utcnow(),
                            metadata={
                                "vector_entity_name": vector_entity["entity_name"],
                                "graph_entity_name": graph_entity["entity_name"],
                                "entity_type": graph_entity["entity_type"],
                                "similarity_score": similarity_score,
                                "created_at": datetime.utcnow().isoformat()
                            }
                        )
                        self.db.add(entity_link)

                    created_links += 1

                    # 更新进度
                    progress = (created_links / total_matches) * 100
                    await self.state_machine.update_progress(
                        document_sync_id, link_progress=progress
                    )

                except Exception as e:
                    logger.error(f"Error creating entity link: {str(e)}")
                    await self._log_link_error(
                        document_sync_id, "entity_linking",
                        f"Error creating entity link: {str(e)}",
                        {"vector_entity": vector_entity, "graph_entity": graph_entity}
                    )

            # 处理跨模态关联（表格与文本）
            if self.cross_modal_associator.table_text_association:
                await self._handle_cross_modal_associations(document_sync_id)

            self.db.commit()

            # 关联建立完成
            await self.state_machine.transition_state(
                document_sync_id, SyncStatus.COMPLETED,
                f"Entity linking completed: {created_links} links established"
            )

            logger.info(f"Entity linking completed for document_sync {document_sync_id}")
            return True

        except Exception as e:
            logger.error(f"Error in entity linking for document_sync {document_sync_id}: {str(e)}")

            # 更新状态为失败
            await self.state_machine.transition_state(
                document_sync_id, SyncStatus.FAILED,
                f"Entity linking failed: {str(e)}"
            )

            await self._log_link_error(
                document_sync_id, "entity_linking",
                f"Entity linking failed: {str(e)}",
                {"error_details": str(e)}
            )

            return False

    async def _handle_cross_modal_associations(self, document_sync_id: int):
        """处理跨模态关联"""
        try:
            # 这里可以实现表格与文本的关联逻辑
            # 根据具体需求实现
            pass

        except Exception as e:
            logger.error(f"Error handling cross-modal associations: {str(e)}")

    async def update_link_strength(self, link_id: int, new_strength: float) -> bool:
        """
        更新关联强度

        Args:
            link_id: 关联ID
            new_strength: 新的关联强度

        Returns:
            bool: 是否更新成功
        """
        try:
            entity_link = self.db.query(EntityLink).filter(
                EntityLink.id == link_id
            ).first()

            if not entity_link:
                logger.error(f"EntityLink {link_id} not found")
                return False

            # 更新关联强度
            entity_link.link_strength = min(max(new_strength, 0.0), 1.0)
            entity_link.linked_at = datetime.utcnow()

            # 更新元数据
            entity_link.metadata = {
                **(entity_link.metadata or {}),
                "strength_updated": datetime.utcnow().isoformat(),
                "manual_update": True
            }

            self.db.commit()

            logger.info(f"Successfully updated link strength for link {link_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating link strength for link {link_id}: {str(e)}")
            return False

    async def delete_link(self, link_id: int) -> bool:
        """
        删除实体关联

        Args:
            link_id: 关联ID

        Returns:
            bool: 是否删除成功
        """
        try:
            entity_link = self.db.query(EntityLink).filter(
                EntityLink.id == link_id
            ).first()

            if not entity_link:
                logger.error(f"EntityLink {link_id} not found")
                return False

            self.db.delete(entity_link)
            self.db.commit()

            logger.info(f"Successfully deleted entity link {link_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting entity link {link_id}: {str(e)}")
            return False

    async def query_entity_links(
        self, document_sync_id: Optional[int] = None,
        entity_type: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[Dict]:
        """
        查询实体关联

        Args:
            document_sync_id: 文档同步ID（可选）
            entity_type: 实体类型（可选）
            min_confidence: 最小置信度（可选）

        Returns:
            List[Dict]: 关联列表
        """
        try:
            query = self.db.query(EntityLink)

            if document_sync_id:
                query = query.filter(EntityLink.document_sync_id == document_sync_id)

            links = query.all()

            # 转换为字典并过滤
            result = []
            for link in links:
                metadata = link.metadata or {}

                # 过滤实体类型
                if entity_type and metadata.get("entity_type") != entity_type:
                    continue

                # 过滤置信度
                if min_confidence and link.confidence_score < min_confidence:
                    continue

                result.append({
                    "id": link.id,
                    "vector_entity_id": link.vector_entity_id,
                    "graph_entity_id": link.graph_entity_id,
                    "link_type": link.link_type,
                    "confidence_score": link.confidence_score,
                    "link_strength": link.link_strength,
                    "created_at": link.created_at.isoformat() if link.created_at else None,
                    "metadata": metadata
                })

            return result

        except Exception as e:
            logger.error(f"Error querying entity links: {str(e)}")
            return []

    async def get_link_statistics(self, document_sync_id: Optional[int] = None) -> Dict:
        """
        获取关联统计信息

        Args:
            document_sync_id: 文档同步ID（可选）

        Returns:
            Dict: 统计信息
        """
        try:
            query = self.db.query(EntityLink)

            if document_sync_id:
                query = query.filter(EntityLink.document_sync_id == document_sync_id)

            all_links = query.all()
            completed_links = [link for link in all_links if link.sync_status == SyncStatus.COMPLETED]

            # 按类型统计
            link_types = {}
            for link in completed_links:
                link_types[link.link_type] = link_types.get(link.link_type, 0) + 1

            # 按实体类型统计
            entity_types = {}
            for link in completed_links:
                metadata = link.metadata or {}
                entity_type = metadata.get("entity_type", "unknown")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

            # 置信度统计
            confidences = [link.confidence_score for link in completed_links if link.confidence_score]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            statistics = {
                "total_links": len(all_links),
                "completed_links": len(completed_links),
                "completion_rate": (len(completed_links) / len(all_links) * 100) if all_links else 0.0,
                "average_confidence": avg_confidence,
                "link_types": link_types,
                "entity_types": entity_types
            }

            return statistics

        except Exception as e:
            logger.error(f"Error getting link statistics: {str(e)}")
            return {}

    async def _log_link_error(
        self, document_sync_id: int, component: str, message: str, details: Dict
    ):
        """记录关联错误日志"""
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
            logger.error(f"Failed to log link error: {str(e)}")