"""
深度知识图谱提取服务
充分利用文档解析内容（摘要、关键点、表格、主题等）构建丰富的知识图谱
"""
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

from langchain_core.documents import Document

from app.services.parsers.advanced.enhanced_metadata_extractor import (
    ChunkMetadataExtraction,
    DocumentMetadataExtraction,
    KeyPoint,
    ExtractedTable,
    ExtractedTopic
)
from app.services.knowledge.entity_disambiguation import EntityMention, EntityCluster
from app.services.knowledge.enhanced_relation_extractor import ExtractedRelation
from app.core.graph_config import GraphEntityType, GraphRelationType, generate_entity_id

logger = logging.getLogger(__name__)


class EntitySource(str, Enum):
    """实体来源"""
    FROM_SUMMARY = "summary"           # 从摘要中提取
    FROM_KEY_POINTS = "key_points"     # 从关键点中提取
    FROM_CONTENT = "content"           # 从内容中提取
    FROM_TABLE = "table"               # 从表格中提取
    FROM_TOPICS = "topics"             # 从主题中提取


@dataclass
class EnrichedEntity:
    """增强的实体"""
    id: str
    name: str
    type: GraphEntityType
    confidence: float

    # 增强信息
    sources: List[EntitySource] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # 关联信息
    related_key_points: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)

    # 上下文
    summary_context: str = ""
    table_context: str = ""

    # 来源追踪
    document_id: str = ""
    chunk_ids: Set[str] = field(default_factory=set)

    # 质量指标
    mention_count: int = 1
    importance_score: float = 0.0


@dataclass
class EnrichedRelation:
    """增强的关系"""
    id: str
    subject: str
    object: str
    relation_type: GraphRelationType
    confidence: float
    direction: str  # forward, backward, bidirectional

    # 增强信息
    evidence: List[str] = field(default_factory=list)
    key_point_support: List[str] = field(default_factory=list)
    table_support: List[str] = field(default_factory=list)

    # 属性
    attributes: Dict[str, Any] = field(default_factory=dict)
    temporal_info: Optional[str] = None  # 时间信息
    quantitative_info: Dict[str, float] = field(default_factory=dict)

    # 来源追踪
    source_chunk_ids: Set[str] = field(default_factory=set)
    extraction_method: str = ""  # summary, key_point, table, content


@dataclass
class TemporalEvent:
    """时间事件"""
    id: str
    event_type: str  # 投资, 并购, 任职等
    timestamp: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0


@dataclass
class QuantitativeRelation:
    """量化关系（包含数值）"""
    id: str
    subject: str
    object: str
    metric_name: str  # 营收、利润、投资额等
    metric_value: float
    unit: str = ""
    time_period: str = ""
    confidence: float = 0.0


class DeepKnowledgeGraphExtractor:
    """深度知识图谱提取器"""

    def __init__(self):
        # 金融实体模式（增强版）
        self.financial_entity_patterns = self._load_financial_patterns()

        # 关系抽取规则（增强版）
        self.relation_extraction_rules = self._load_relation_rules()

        # 量化指标模式
        self.metric_patterns = self._load_metric_patterns()

    def _load_financial_patterns(self) -> Dict[str, List[str]]:
        """加载金融实体模式"""
        return {
            # 公司模式
            "company": [
                r'[\u4e00-\u9fff]+(?:股份有限公司|有限公司|集团|控股|科技|银行|证券|保险|基金)',
                r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Group|Holding))',
            ],
            # 人物模式（带职位）
            "executive": [
                r'[\u4e00-\u9fff]{2,4}(?:董事长|CEO|总裁|总经理|行长|董事|监事|CFO|CTO)',
                r'(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            ],
            # 金融产品
            "financial_product": [
                r'[\u4e00-\u9fff]+(?:基金|股票|债券|期货|期权|理财产品)',
                r'[A-Z]{2,6}\s+(?:Fund|ETF|Bond|Stock)',
            ],
            # 财务指标
            "financial_metric": [
                r'(?:营业|毛|净|总)?(?:收入|利润|营收)',
                r'(?:总资产|净资产|负债)',
                r'(?:ROE|ROA|毛利率|净利率|市盈率|市净率)',
                r'(?:每股收益|EPS|股息率)',
            ],
        }

    def _load_relation_rules(self) -> Dict[str, List[Dict]]:
        """加载增强的关系抽取规则"""
        return {
            "investment": [
                {
                    "pattern": r'(.+?)(?:投资|参股|入股)(?:了)?(.+?)(?:\d+(?:亿|万|千万)?(?:元|美元)?)',
                    "type": GraphRelationType.INVESTS_IN,
                    "extract_amount": True,
                    "confidence": 0.9
                },
                {
                    "pattern": r'(.+?)对(.+?)进行(?:战略)?投资',
                    "type": GraphRelationType.INVESTS_IN,
                    "extract_amount": False,
                    "confidence": 0.85
                },
            ],
            "acquisition": [
                {
                    "pattern": r'(.+?)(?:收购|并购)(?:了)?(.+?)(?:\d+(?:%|(?:亿|万|千万)?(?:元|美元|股)?)?)',
                    "type": GraphRelationType.SUBSIDIARY_OF,
                    "extract_amount": True,
                    "confidence": 0.9
                },
            ],
            "partnership": [
                {
                    "pattern": r'(.+?)与(.+?)(?:达成|建立|签署)(?:战略)?(?:合作伙伴关系|合作)',
                    "type": GraphRelationType.PARTNER_OF,
                    "extract_amount": False,
                    "confidence": 0.85
                },
            ],
            "executive": [
                {
                    "pattern": r'(.+?)担任(.+?)(?:董事长|CEO|总裁|总经理)',
                    "type": GraphRelationType.CEO_OF,
                    "extract_amount": False,
                    "confidence": 0.95
                },
            ],
        }

    def _load_metric_patterns(self) -> Dict[str, str]:
        """加载量化指标模式"""
        return {
            "revenue": r'(?:营业收入?|营收|总收入)[：:]\s*([\d,]+(?:\.\d+)?)\s*(?:万|亿)?元',
            "profit": r'(?:净利润?|利润总额?)[：:]\s*([\d,]+(?:\.\d+)?)\s*(?:万|亿)?元',
            "assets": r'(?:总资产|资产总额)[：:]\s*([\d,]+(?:\.\d+)?)\s*(?:万|亿)?元',
            "investment_amount": r'(?:投资|参股)(?:金额|总额)?[：:]\s*([\d,]+(?:\.\d+)?)\s*(?:万|亿)?元',
        }

    async def extract_enriched_entities(
        self,
        document_analysis_result: Any
    ) -> List[EnrichedEntity]:
        """
        提取增强的实体
        利用摘要、关键点、表格、主题等丰富信息

        Args:
            document_analysis_result: DocumentAnalysisResult

        Returns:
            EnrichedEntity 列表
        """
        try:
            document_id = document_analysis_result.document_id
            chunks_metadata = document_analysis_result.chunks_metadata

            all_enriched_entities = {}

            # 从多个来源提取实体
            for i, metadata in enumerate(chunks_metadata):
                chunk_id = metadata.chunk_id

                # 1. 从摘要提取实体
                summary_entities = await self._extract_entities_from_summary(
                    metadata.summary,
                    document_id,
                    chunk_id
                )

                # 2. 从关键点提取实体
                keypoint_entities = await self._extract_entities_from_keypoints(
                    metadata.key_points,
                    document_id,
                    chunk_id
                )

                # 3. 从主题提取实体
                topic_entities = await self._extract_entities_from_topics(
                    metadata.topics,
                    document_id,
                    chunk_id
                )

                # 4. 从表格提取实体
                table_entities = await self._extract_entities_from_tables(
                    metadata.tables,
                    document_id,
                    chunk_id
                )

                # 合并实体
                for entity_list in [
                    summary_entities, keypoint_entities, topic_entities, table_entities
                ]:
                    for entity in entity_list:
                        if entity.id not in all_enriched_entities:
                            all_enriched_entities[entity.id] = entity
                        else:
                            # 合并信息
                            existing = all_enriched_entities[entity.id]
                            self._merge_entity_info(existing, entity)

            # 计算重要性分数
            for entity in all_enriched_entities.values():
                entity.importance_score = self._calculate_importance_score(entity)

            logger.info(
                f"从文档 {document_id} 提取了 {len(all_enriched_entities)} 个增强实体"
            )

            return list(all_enriched_entities.values())

        except Exception as e:
            logger.error(f"提取增强实体失败: {e}")
            return []

    async def _extract_entities_from_summary(
        self,
        summary: str,
        document_id: str,
        chunk_id: str
    ) -> List[EnrichedEntity]:
        """从摘要中提取实体"""
        entities = []

        if not summary:
            return entities

        # 使用规则提取
        for entity_type_str, patterns in self.financial_entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, summary)
                for match in matches:
                    entity_name = match.group(0).strip()

                    # 映射实体类型
                    entity_type = self._map_entity_type(entity_type_str)

                    entity = EnrichedEntity(
                        id=generate_entity_id(entity_name, entity_type),
                        name=entity_name,
                        type=entity_type,
                        confidence=0.9,  # 摘要中的实体置信度高
                        sources=[EntitySource.FROM_SUMMARY],
                        descriptions=[summary[:100]],
                        document_id=document_id,
                        chunk_ids={chunk_id}
                    )

                    entities.append(entity)

        return entities

    async def _extract_entities_from_keypoints(
        self,
        key_points: List[KeyPoint],
        document_id: str,
        chunk_id: str
    ) -> List[EnrichedEntity]:
        """从关键点中提取实体"""
        entities = []

        if not key_points:
            return entities

        for kp in key_points:
            # 提取高重要级关键点中的实体
            if kp.importance != "high":
                continue

            for entity_type_str, patterns in self.financial_entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, kp.point)
                    for match in matches:
                        entity_name = match.group(0).strip()
                        entity_type = self._map_entity_type(entity_type_str)

                        entity = EnrichedEntity(
                            id=generate_entity_id(entity_name, entity_type),
                            name=entity_name,
                            type=entity_type,
                            confidence=0.95,  # 高重要性关键点的实体置信度更高
                            sources=[EntitySource.FROM_KEY_POINTS],
                            related_key_points=[kp.point],
                            document_id=document_id,
                            chunk_ids={chunk_id}
                        )

                        entities.append(entity)

        return entities

    async def _extract_entities_from_topics(
        self,
        topics: List[ExtractedTopic],
        document_id: str,
        chunk_id: str
    ) -> List[EnrichedEntity]:
        """从主题中提取实体"""
        entities = []

        if not topics:
            return entities

        for topic in topics:
            # 只处理相关性高的主题
            if topic.relevance_score < 0.7:
                continue

            # 提取主题中的实体
            for entity_type_str, patterns in self.financial_entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, topic.topic)
                    for match in matches:
                        entity_name = match.group(0).strip()
                        entity_type = self._map_entity_type(entity_type_str)

                        entity = EnrichedEntity(
                            id=generate_entity_id(entity_name, entity_type),
                            name=entity_name,
                            type=entity_type,
                            confidence=0.7 + (topic.relevance_score * 0.2),  # 基于相关性调整
                            sources=[EntitySource.FROM_TOPICS],
                            related_topics=[topic.topic],
                            document_id=document_id,
                            chunk_ids={chunk_id}
                        )

                        entities.append(entity)

        return entities

    async def _extract_entities_from_tables(
        self,
        tables: List[ExtractedTable],
        document_id: str,
        chunk_id: str
    ) -> List[EnrichedEntity]:
        """从表格中提取实体"""
        entities = []

        if not tables:
            return entities

        for table in tables:
            # 从表格标题提取实体
            title_entities = await self._extract_entities_from_text(
                table.title,
                EntitySource.FROM_TABLE,
                0.9
            )

            # 从表格数据提取实体
            for row in table.rows[:5]:  # 只处理前5行
                row_text = " | ".join(row)
                row_entities = await self._extract_entities_from_text(
                    row_text,
                    EntitySource.FROM_TABLE,
                    0.8
                )
                title_entities.extend(row_entities)

            # 添加表格上下文
            for entity in title_entities:
                entity.table_context = f"表格: {table.title}, 摘要: {table.summary}"
                entity.sources.append(EntitySource.FROM_TABLE)
                entity.related_tables.append(table.title)
                entity.document_id = document_id
                entity.chunk_ids.add(chunk_id)
                entities.append(entity)

        return entities

    async def _extract_entities_from_text(
        self,
        text: str,
        source: EntitySource,
        base_confidence: float
    ) -> List[EnrichedEntity]:
        """从文本中提取实体（辅助方法）"""
        entities = []

        for entity_type_str, patterns in self.financial_entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_name = match.group(0).strip()
                    entity_type = self._map_entity_type(entity_type_str)

                    entity = EnrichedEntity(
                        id=generate_entity_id(entity_name, entity_type),
                        name=entity_name,
                        type=entity_type,
                        confidence=base_confidence,
                        sources=[source]
                    )

                    entities.append(entity)

        return entities

    def _map_entity_type(self, type_str: str) -> GraphEntityType:
        """映射实体类型"""
        mapping = {
            "company": GraphEntityType.COMPANY,
            "executive": GraphEntityType.PERSON,
            "financial_product": GraphEntityType.PRODUCT,
            "financial_metric": GraphEntityType.FINANCIAL_INDICATOR,
        }
        return mapping.get(type_str, GraphEntityType.CONCEPT)

    def _merge_entity_info(self, existing: EnrichedEntity, new: EnrichedEntity):
        """合并实体信息"""
        # 合并来源
        existing.sources.extend(new.sources)

        # 合并描述
        existing.descriptions.extend(new.descriptions)

        # 合并属性
        existing.attributes.update(new.attributes)

        # 合并关联信息
        existing.related_key_points.extend(new.related_key_points)
        existing.related_topics.extend(new.related_topics)
        existing.related_tables.extend(new.related_tables)

        # 合并上下文
        if new.summary_context:
            existing.summary_context = (
                existing.summary_context + " | " + new.summary_context
            )
        if new.table_context:
            existing.table_context = (
                existing.table_context + " | " + new.table_context
            )

        # 合并chunk IDs
        existing.chunk_ids.update(new.chunk_ids)

        # 更新提及次数
        existing.mention_count += 1

        # 更新置信度（取较高值）
        existing.confidence = max(existing.confidence, new.confidence)

    def _calculate_importance_score(self, entity: EnrichedEntity) -> float:
        """计算实体重要性分数"""
        score = 0.0

        # 基于来源数量
        score += len(set(entity.sources)) * 0.2

        # 基于提及次数
        score += min(entity.mention_count * 0.1, 0.3)

        # 基于描述数量
        score += min(len(entity.descriptions) * 0.05, 0.2)

        # 基于关联信息丰富度
        related_info_count = (
            len(entity.related_key_points) +
            len(entity.related_topics) +
            len(entity.related_tables)
        )
        score += min(related_info_count * 0.1, 0.3)

        return min(1.0, score)

    async def extract_enriched_relations(
        self,
        document_analysis_result: Any,
        enriched_entities: List[EnrichedEntity]
    ) -> List[EnrichedRelation]:
        """
        提取增强的关系
        利用摘要、关键点、表格等信息

        Args:
            document_analysis_result: DocumentAnalysisResult
            enriched_entities: 增强的实体列表

        Returns:
            EnrichedRelation 列表
        """
        try:
            document_id = document_analysis_result.document_id
            chunks_metadata = document_analysis_result.chunks_metadata

            # 构建实体名称到ID的映射
            entity_name_to_id = {
                entity.name: entity.id
                for entity in enriched_entities
            }

            all_relations = {}

            # 从多个来源提取关系
            for metadata in chunks_metadata:
                # 1. 从摘要提取关系
                summary_relations = await self._extract_relations_from_text(
                    metadata.summary,
                    entity_name_to_id,
                    document_id,
                    metadata.chunk_id,
                    "summary",
                    0.9
                )

                # 2. 从关键点提取关系
                keypoint_relations = await self._extract_relations_from_keypoints(
                    metadata.key_points,
                    entity_name_to_id,
                    document_id,
                    metadata.chunk_id
                )

                # 3. 从表格提取关系
                table_relations = await self._extract_relations_from_tables(
                    metadata.tables,
                    entity_name_to_id,
                    document_id,
                    metadata.chunk_id
                )

                # 合并关系
                for relation in summary_relations + keypoint_relations + table_relations:
                    key = (relation.subject, relation.object, relation.relation_type)

                    if key not in all_relations:
                        all_relations[key] = relation
                    else:
                        # 合并关系信息
                        existing = all_relations[key]
                        self._merge_relation_info(existing, relation)

            logger.info(
                f"从文档 {document_id} 提取了 {len(all_relations)} 个增强关系"
            )

            return list(all_relations.values())

        except Exception as e:
            logger.error(f"提取增强关系失败: {e}")
            return []

    async def _extract_relations_from_text(
        self,
        text: str,
        entity_name_to_id: Dict[str, str],
        document_id: str,
        chunk_id: str,
        extraction_method: str,
        base_confidence: float
    ) -> List[EnrichedRelation]:
        """从文本中提取关系"""
        relations = []

        for rule_group in self.relation_extraction_rules.values():
            for rule in rule_group:
                matches = re.finditer(rule["pattern"], text)
                for match in matches:
                    try:
                        if len(match.groups()) < 2:
                            continue

                        subject_name = match.group(1).strip()
                        object_name = match.group(2).strip()

                        # 查找实体ID
                        subject_id = entity_name_to_id.get(subject_name)
                        object_id = entity_name_to_id.get(object_name)

                        if not (subject_id and object_id):
                            continue

                        # 创建关系
                        relation_id = f"{subject_id}_{object_id}_{rule['type'].value}"

                        relation = EnrichedRelation(
                            id=relation_id,
                            subject=subject_id,
                            object=object_id,
                            relation_type=rule["type"],
                            confidence=base_confidence,
                            direction="forward",
                            evidence=[text[:200]],
                            extraction_method=extraction_method,
                            source_chunk_ids={chunk_id}
                        )

                        # 提取量化信息（如果需要）
                        if rule.get("extract_amount"):
                            amount = self._extract_amount_from_text(text)
                            if amount:
                                relation.quantitative_info = {"amount": amount}

                        relations.append(relation)

                    except Exception as e:
                        logger.warning(f"处理关系匹配失败: {e}")
                        continue

        return relations

    async def _extract_relations_from_keypoints(
        self,
        key_points: List[KeyPoint],
        entity_name_to_id: Dict[str, str],
        document_id: str,
        chunk_id: str
    ) -> List[EnrichedRelation]:
        """从关键点中提取关系"""
        relations = []

        if not key_points:
            return relations

        for kp in key_points:
            # 只处理高重要级关键点
            if kp.importance != "high":
                continue

            # 从关键点提取关系
            kp_relations = await self._extract_relations_from_text(
                kp.point,
                entity_name_to_id,
                document_id,
                chunk_id,
                "key_point",
                0.95  # 高重要性关键点中的关系置信度更高
            )

            # 添加关键点支持
            for relation in kp_relations:
                relation.key_point_support.append(kp.point)

            relations.extend(kp_relations)

        return relations

    async def _extract_relations_from_tables(
        self,
        tables: List[ExtractedTable],
        entity_name_to_id: Dict[str, str],
        document_id: str,
        chunk_id: str
    ) -> List[EnrichedRelation]:
        """从表格中提取关系"""
        relations = []

        if not tables:
            return relations

        for table in tables:
            # 从表格中提取实体
            table_entities = await self._extract_entities_from_text(
                table.title,
                EntitySource.FROM_TABLE,
                0.9
            )

            # 尝试从表格数据中识别关系
            # 简化实现：查找投资、持股等关键词
            if any(keyword in table.summary for keyword in ["投资", "持股", "收购", "合作"]):
                # 提取关系
                table_text = f"{table.title} {table.summary} {' '.join([' '.join(row) for row in table.rows[:3]])}"
                table_relations = await self._extract_relations_from_text(
                    table_text,
                    entity_name_to_id,
                    document_id,
                    chunk_id,
                    "table",
                    0.85
                )

                # 添加表格支持
                for relation in table_relations:
                    relation.table_support.append(table.title)

                relations.extend(table_relations)

        return relations

    def _extract_amount_from_text(self, text: str) -> Optional[float]:
        """从文本中提取金额"""
        amount_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:亿|万|千万)?元',
            r'\$([\d,]+(?:\.\d+)?)',
            r'([\d,]+(?:\.\d+)?)\s*(?:million|billion)',
        ]

        for pattern in amount_patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    # 转换单位
                    if '亿' in text:
                        amount *= 100000000
                    elif '万' in text:
                        amount *= 10000
                    elif 'million' in text:
                        amount *= 1000000
                    elif 'billion' in text:
                        amount *= 1000000000
                    return amount
                except ValueError:
                    continue

        return None

    def _merge_relation_info(self, existing: EnrichedRelation, new: EnrichedRelation):
        """合并关系信息"""
        # 合并证据
        existing.evidence.extend(new.evidence)

        # 合并关键点支持
        existing.key_point_support.extend(new.key_point_support)

        # 合并表格支持
        existing.table_support.extend(new.table_support)

        # 合并属性
        existing.attributes.update(new.attributes)

        # 合并量化信息
        existing.quantitative_info.update(new.quantitative_info)

        # 合并来源chunk IDs
        existing.source_chunk_ids.update(new.source_chunk_ids)

        # 更新置信度（取较高值）
        existing.confidence = max(existing.confidence, new.confidence)


# 全局实例
deep_kg_extractor = DeepKnowledgeGraphExtractor()
