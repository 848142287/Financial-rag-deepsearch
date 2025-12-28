"""
金融知识图谱构建器
构建和维护金融领域的知识图谱，支持实体关系抽取和图谱推理

⚠️  **DEPRECATED** - 此服务已废弃

请使用新的增强知识图谱服务：
- `app.services.knowledge.deep_graph_extractor.DeepKnowledgeGraphExtractor`
- `app.services.unified_knowledge_graph.UnifiedKnowledgeGraphService`
- `app.services.fusion_service.fusion_document_service`

迁移原因：
- 使用 NetworkX 内存图谱，无法持久化大规模数据
- 实体ID包含文档ID，导致跨文档实体重复
- 缺少LLM增强提取和多源数据融合
- 无实体消歧和质量验证机制

迁移指南：请参阅 `docs/代码迁移指南.md`

此文件保留用于向后兼容，将在未来版本中移除。
"""

import warnings
warnings.warn(
    "FinancialKnowledgeGraph 已废弃，请使用 DeepKnowledgeGraphExtractor "
    "和 UnifiedKnowledgeGraphService。"
    "详见 docs/代码迁移指南.md",
    DeprecationWarning,
    stacklevel=2
)

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import networkx as nx
from datetime import datetime
import re

from neo4j import GraphDatabase
from rdflib import Graph, URIRef, Literal, Namespace
from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty
from app.core.config import settings

from .financial_llm_service import financial_llm_service, FinancialEntity
from .data_balancer import DocumentCategory

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """实体类型"""
    COMPANY = "Company"  # 公司
    PERSON = "Person"  # 人员
    STOCK = "Stock"  # 股票
    INDUSTRY = "Industry"  # 行业
    PRODUCT = "Product"  # 产品
    LOCATION = "Location"  # 地点
    EVENT = "Event"  # 事件
    CONCEPT = "Concept"  # 概念
    REGULATION = "Regulation"  # 法规
    FINANCIAL_INDICATOR = "FinancialIndicator"  # 财务指标


class RelationType(Enum):
    """关系类型"""
    OWNS = "owns"  # 拥有
    WORKS_FOR = "works_for"  # 任职于
    SUBSIDIARY_OF = "subsidiary_of"  # 子公司
    COMPETITOR_OF = "competitor_of"  # 竞争对手
    PARTNER_OF = "partner_of"  # 合作伙伴
    INVESTS_IN = "invests_in"  # 投资于
    LOCATED_IN = "located_in"  # 位于
    BELONGS_TO = "belongs_to"  # 属于
    AFFECTS = "affects"  # 影响
    REGULATED_BY = "regulated_by"  # 受...监管
    MEASURES = "measures"  # 衡量
    INVOLVES = "involves"  # 涉及


@dataclass
class KGraphEntity:
    """知识图谱实体"""
    id: str
    type: EntityType
    name: str
    aliases: List[str]
    properties: Dict[str, Any]
    confidence: float
    source_documents: Set[str]


@dataclass
class KGraphRelation:
    """知识图谱关系"""
    id: str
    subject_id: str
    object_id: str
    relation_type: RelationType
    properties: Dict[str, Any]
    confidence: float
    source_documents: Set[str]


@dataclass
class ExtractionResult:
    """抽取结果"""
    entities: List[KGraphEntity]
    relations: List[KGraphRelation]
    confidence: float
    metadata: Dict[str, Any]


class FinancialKnowledgeGraph:
    """金融知识图谱构建器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化知识图谱构建器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # Neo4j配置 - 使用settings中的配置
        self.neo4j_uri = self.config.get("neo4j_uri", settings.neo4j_uri)
        self.neo4j_user = self.config.get("neo4j_user", settings.neo4j_user)
        self.neo4j_password = self.config.get("neo4j_password", settings.neo4j_password)

        # 初始化图结构
        self.graph = nx.MultiDiGraph()

        # 实体和关系存储
        self.entities: Dict[str, KGraphEntity] = {}
        self.relations: Dict[str, KGraphRelation] = {}

        # 金融领域词汇
        self.financial_vocabulary = self._load_financial_vocabulary()

        # 关系抽取规则
        self.relation_patterns = self._load_relation_patterns()

    def _load_financial_vocabulary(self) -> Dict[str, Any]:
        """加载金融领域词汇"""
        return {
            "company_indicators": [
                "公司", "企业", "集团", "股份", "有限", "责任", "公司", "集团", "控股",
                "corporation", "inc", "ltd", "co", "group", "holding"
            ],
            "person_indicators": [
                "先生", "女士", "董事长", "ceo", "总裁", "总经理", "董事", "监事",
                "chairman", "ceo", "president", "director", "manager"
            ],
            "stock_indicators": [
                "股票", "股份", "代码", "上市", "退市", "涨停", "跌停",
                "stock", "share", "ticker", "listed", "delisted"
            ],
            "financial_metrics": [
                "营收", "利润", "资产", "负债", "现金流", "roe", "roa", "毛利率",
                "revenue", "profit", "asset", "liability", "cash_flow", "margin"
            ],
            "industry_terms": [
                "银行", "保险", "证券", "房地产", "制造业", "科技", "医疗", "能源",
                "banking", "insurance", "securities", "real_estate", "manufacturing"
            ]
        }

    def _load_relation_patterns(self) -> Dict[str, List[str]]:
        """加载关系抽取规则"""
        return {
            RelationType.OWNS: [
                r"(.+?)拥有(.+?)",
                r"(.+?)持有(.+?)",
                r"(.+?)owns?(.+?)",
                r"(.+?)holds?(.+?)"
            ],
            RelationType.WORKS_FOR: [
                r"(.+?)任职于(.+?)",
                r"(.+?)担任(.+?)",
                r"(.+?)在(.+?)工作",
                r"(.+?)works? for (.+?)",
                r"(.+?)employed by (.+?)"
            ],
            RelationType.SUBSIDIARY_OF: [
                r"(.+?)是(.+?)的子公司",
                r"(.+?)是(.+?)的附属公司",
                r"(.+?)subsidiary of (.+?)"
            ],
            RelationType.INVESTS_IN: [
                r"(.+?)投资(.+?)",
                r"(.+?)入股(.+?)",
                r"(.+?)invests? in (.+?)"
            ],
            RelationType.COMPETITOR_OF: [
                r"(.+?)与(.+?)竞争",
                r"(.+?)是(.+?)的竞争对手",
                r"(.+?)competes? with (.+?)"
            ],
            RelationType.PARTNER_OF: [
                r"(.+?)与(.+?)合作",
                r"(.+?)partner with (.+?)",
                r"(.+?)和(.+?)建立合作关系"
            ]
        }

    async def build_from_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        从文档构建知识图谱

        Args:
            documents: 文档列表

        Returns:
            是否构建成功
        """
        try:
            logger.info(f"开始从{len(documents)}个文档构建知识图谱")

            for doc in documents:
                await self._process_document(doc)

            # 实体消歧和合并
            await self._entity_disambiguation()

            # 关系验证和清理
            await self._relation_validation()

            logger.info(f"知识图谱构建完成: {len(self.entities)}个实体, {len(self.relations)}个关系")
            return True

        except Exception as e:
            logger.error(f"知识图谱构建失败: {str(e)}")
            return False

    async def _process_document(self, document: Dict[str, Any]):
        """处理单个文档"""
        try:
            doc_id = document.get("id", "")
            content = document.get("content", "")
            category = document.get("category", "")

            # 使用金融模型抽取实体
            financial_entities = await self._extract_financial_entities(content, doc_id)

            # 抽取关系
            relations = await self._extract_relations(content, financial_entities, doc_id)

            # 添加到图谱
            for entity in financial_entities:
                await self._add_entity(entity)

            for relation in relations:
                await self._add_relation(relation)

        except Exception as e:
            logger.error(f"文档处理失败: {str(e)}")

    async def _extract_financial_entities(self, text: str, doc_id: str) -> List[KGraphEntity]:
        """抽取金融实体"""
        entities = []

        try:
            # 使用金融LLM服务进行实体识别
            entity_result = await financial_llm_service.extract_entities(text)

            # 转换为知识图谱实体
            for financial_entity in entity_result.result:
                entity_type = self._map_entity_type(financial_entity.entity_type)
                if entity_type:
                    kg_entity = KGraphEntity(
                        id=f"{entity_type.value}_{financial_entity.entity}_{doc_id}",
                        type=entity_type,
                        name=financial_entity.entity,
                        aliases=[financial_entity.entity],
                        properties={
                            "confidence": financial_entity.confidence,
                            "start_pos": financial_entity.start_pos,
                            "end_pos": financial_entity.end_pos,
                            "source_doc": doc_id
                        },
                        confidence=financial_entity.confidence,
                        source_documents={doc_id}
                    )
                    entities.append(kg_entity)

            # 使用规则进行补充实体抽取
            rule_entities = await self._rule_based_entity_extraction(text, doc_id)
            entities.extend(rule_entities)

        except Exception as e:
            logger.error(f"金融实体抽取失败: {str(e)}")

        return entities

    async def _rule_based_entity_extraction(self, text: str, doc_id: str) -> List[KGraphEntity]:
        """基于规则的实体抽取"""
        entities = []

        try:
            # 抽取公司实体
            company_pattern = r"([^\s，。！？]+(?:公司|企业|集团|股份|有限公司|Corporation|Inc\.|Ltd\.|Group|Holding))"
            for match in re.finditer(company_pattern, text):
                company_name = match.group(1).strip()
                if len(company_name) > 2:  # 过滤过短的匹配
                    entity = KGraphEntity(
                        id=f"Company_{company_name}_{doc_id}",
                        type=EntityType.COMPANY,
                        name=company_name,
                        aliases=[company_name],
                        properties={
                            "confidence": 0.8,
                            "source_doc": doc_id
                        },
                        confidence=0.8,
                        source_documents={doc_id}
                    )
                    entities.append(entity)

            # 抽取股票代码
            stock_pattern = r"(\d{6})|[A-Z]{1,4}\s?(?:股票|股份)?"
            for match in re.finditer(stock_pattern, text):
                stock_code = match.group(1) if match.group(1) else match.group(0)
                entity = KGraphEntity(
                    id=f"Stock_{stock_code}_{doc_id}",
                    type=EntityType.STOCK,
                    name=stock_code,
                    aliases=[stock_code],
                    properties={
                        "confidence": 0.9,
                        "source_doc": doc_id
                    },
                    confidence=0.9,
                    source_documents={doc_id}
                )
                entities.append(entity)

            # 抽取财务指标
            metric_pattern = r"([^\s，。！？]*(?:营收|利润|资产|负债|现金流|ROE|ROA|毛利率|净利率)[^\s，。！？]*)"
            for match in re.finditer(metric_pattern, text):
                metric_name = match.group(1).strip()
                if len(metric_name) > 3:
                    entity = KGraphEntity(
                        id=f"FinancialIndicator_{metric_name}_{doc_id}",
                        type=EntityType.FINANCIAL_INDICATOR,
                        name=metric_name,
                        aliases=[metric_name],
                        properties={
                            "confidence": 0.7,
                            "source_doc": doc_id
                        },
                        confidence=0.7,
                        source_documents={doc_id}
                    )
                    entities.append(entity)

        except Exception as e:
            logger.error(f"规则实体抽取失败: {str(e)}")

        return entities

    def _map_entity_type(self, entity_type: str) -> Optional[EntityType]:
        """映射实体类型"""
        type_mapping = {
            "COMPANY": EntityType.COMPANY,
            "PERSON": EntityType.PERSON,
            "STOCK": EntityType.STOCK,
            "ORGANIZATION": EntityType.COMPANY,
            "LOCATION": EntityType.LOCATION,
            "AMOUNT": EntityType.CONCEPT
        }
        return type_mapping.get(entity_type.upper())

    async def _extract_relations(
        self,
        text: str,
        entities: List[KGraphEntity],
        doc_id: str
    ) -> List[KGraphRelation]:
        """抽取实体关系"""
        relations = []

        try:
            # 基于规则的关系抽取
            for relation_type, patterns in self.relation_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        subject = match.group(1).strip()
                        object = match.group(2).strip()

                        # 查找对应的实体
                        subject_entity = self._find_matching_entity(subject, entities)
                        object_entity = self._find_matching_entity(object, entities)

                        if subject_entity and object_entity:
                            relation = KGraphRelation(
                                id=f"{relation_type.value}_{subject_entity.id}_{object_entity.id}_{doc_id}",
                                subject_id=subject_entity.id,
                                object_id=object_entity.id,
                                relation_type=relation_type,
                                properties={
                                    "text": match.group(0),
                                    "confidence": 0.7,
                                    "source_doc": doc_id
                                },
                                confidence=0.7,
                                source_documents={doc_id}
                            )
                            relations.append(relation)

        except Exception as e:
            logger.error(f"关系抽取失败: {str(e)}")

        return relations

    def _find_matching_entity(self, name: str, entities: List[KGraphEntity]) -> Optional[KGraphEntity]:
        """查找匹配的实体"""
        name = name.strip()

        for entity in entities:
            if name == entity.name:
                return entity
            # 检查别名
            if name in entity.aliases:
                return entity
            # 模糊匹配
            if name in entity.name or entity.name in name:
                return entity

        return None

    async def _add_entity(self, entity: KGraphEntity):
        """添加实体到图谱"""
        try:
            # 检查是否已存在相似实体
            existing_entity = self._find_similar_entity(entity)
            if existing_entity:
                # 合并实体
                await self._merge_entities(existing_entity, entity)
            else:
                # 添加新实体
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, **entity.__dict__)

        except Exception as e:
            logger.error(f"实体添加失败: {str(e)}")

    async def _add_relation(self, relation: KGraphRelation):
        """添加关系到图谱"""
        try:
            # 检查是否已存在相同关系
            existing_relation = self._find_similar_relation(relation)
            if existing_relation:
                # 合并关系
                await self._merge_relations(existing_relation, relation)
            else:
                # 添加新关系
                self.relations[relation.id] = relation
                self.graph.add_edge(
                    relation.subject_id,
                    relation.object_id,
                    relation_type=relation.relation_type.value,
                    **relation.__dict__
                )

        except Exception as e:
            logger.error(f"关系添加失败: {str(e)}")

    def _find_similar_entity(self, entity: KGraphEntity) -> Optional[KGraphEntity]:
        """查找相似实体"""
        for existing_id, existing_entity in self.entities.items():
            if existing_id == entity.id:
                continue

            # 同类型且名称相似
            if (existing_entity.type == entity.type and
                self._calculate_similarity(existing_entity.name, entity.name) > 0.8):
                return existing_entity

        return None

    def _find_similar_relation(self, relation: KGraphRelation) -> Optional[KGraphRelation]:
        """查找相似关系"""
        for existing_id, existing_relation in self.relations.items():
            if existing_id == relation.id:
                continue

            # 相同的主体、客体和关系类型
            if (existing_relation.subject_id == relation.subject_id and
                existing_relation.object_id == relation.object_id and
                existing_relation.relation_type == relation.relation_type):
                return existing_relation

        return None

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        # 简化实现：使用编辑距离
        if str1 == str2:
            return 1.0

        len1, len2 = len(str1), len(str2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # 计算公共字符比例
        common_chars = set(str1) & set(str2)
        similarity = len(common_chars) / max(len(set(str1)), len(set(str2)))

        return similarity

    async def _merge_entities(self, existing: KGraphEntity, new: KGraphEntity):
        """合并实体"""
        try:
            # 合并别名
            existing.aliases.extend([alias for alias in new.aliases if alias not in existing.aliases])

            # 合并属性
            existing.properties.update(new.properties)

            # 合并来源文档
            existing.source_documents.update(new.source_documents)

            # 更新置信度（取较高值）
            existing.confidence = max(existing.confidence, new.confidence)

        except Exception as e:
            logger.error(f"实体合并失败: {str(e)}")

    async def _merge_relations(self, existing: KGraphRelation, new: KGraphRelation):
        """合并关系"""
        try:
            # 合并属性
            existing.properties.update(new.properties)

            # 合并来源文档
            existing.source_documents.update(new.source_documents)

            # 更新置信度（取较高值）
            existing.confidence = max(existing.confidence, new.confidence)

        except Exception as e:
            logger.error(f"关系合并失败: {str(e)}")

    async def _entity_disambiguation(self):
        """实体消歧"""
        try:
            logger.info("开始实体消歧")

            # 移除低置信度实体
            low_confidence_entities = [
                entity_id for entity_id, entity in self.entities.items()
                if entity.confidence < 0.3
            ]

            for entity_id in low_confidence_entities:
                del self.entities[entity_id]
                if entity_id in self.graph:
                    self.graph.remove_node(entity_id)

            logger.info(f"移除了{len(low_confidence_entities)}个低置信度实体")

        except Exception as e:
            logger.error(f"实体消歧失败: {str(e)}")

    async def _relation_validation(self):
        """关系验证"""
        try:
            logger.info("开始关系验证")

            # 移除不存在实体的关系
            invalid_relations = []

            for relation_id, relation in self.relations.items():
                if (relation.subject_id not in self.entities or
                    relation.object_id not in self.entities):
                    invalid_relations.append(relation_id)

            for relation_id in invalid_relations:
                del self.relations[relation_id]
                if self.graph.has_edge(relation.subject_id, relation.object_id):
                    self.graph.remove_edge(relation.subject_id, relation.object_id)

            logger.info(f"移除了{len(invalid_relations)}个无效关系")

        except Exception as e:
            logger.error(f"关系验证失败: {str(e)}")

    async def query_entity(self, entity_name: str, entity_type: Optional[EntityType] = None) -> List[KGraphEntity]:
        """查询实体"""
        try:
            results = []

            for entity in self.entities.values():
                # 名称匹配
                if (entity_name.lower() in entity.name.lower() or
                    any(entity_name.lower() in alias.lower() for alias in entity.aliases)):

                    # 类型过滤
                    if entity_type is None or entity.type == entity_type:
                        results.append(entity)

            # 按置信度排序
            results.sort(key=lambda x: x.confidence, reverse=True)

            return results

        except Exception as e:
            logger.error(f"实体查询失败: {str(e)}")
            return []

    async def get_entity_relations(self, entity_id: str) -> List[KGraphRelation]:
        """获取实体的关系"""
        try:
            relations = []

            for relation in self.relations.values():
                if relation.subject_id == entity_id or relation.object_id == entity_id:
                    relations.append(relation)

            return relations

        except Exception as e:
            logger.error(f"实体关系查询失败: {str(e)}")
            return []

    async def get_neighbors(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """获取实体邻居"""
        try:
            neighbors = {
                "entities": {},
                "relations": [],
                "paths": []
            }

            if entity_id not in self.entities:
                return neighbors

            # 使用NetworkX查找邻居
            if entity_id in self.graph:
                # 获取指定深度的邻居
                for target, paths in nx.single_source_shortest_path(
                    self.graph, entity_id, cutoff=max_depth
                ).items():
                    if target != entity_id:
                        path_length = len(paths) - 1
                        neighbors["paths"].append({
                            "target_entity": target,
                            "path_length": path_length,
                            "path": paths
                        })

                        # 添加路径中的实体和关系
                        for i in range(len(paths) - 1):
                            source = paths[i]
                            target_node = paths[i + 1]

                            if source not in neighbors["entities"] and source in self.entities:
                                neighbors["entities"][source] = self.entities[source]

                            if target_node not in neighbors["entities"] and target_node in self.entities:
                                neighbors["entities"][target_node] = self.entities[target_node]

                            # 查找对应的关系
                            for relation in self.relations.values():
                                if ((relation.subject_id == source and relation.object_id == target_node) or
                                    (relation.subject_id == target_node and relation.object_id == source)):
                                    if relation not in neighbors["relations"]:
                                        neighbors["relations"].append(relation)

            return neighbors

        except Exception as e:
            logger.error(f"邻居查询失败: {str(e)}")
            return {"entities": {}, "relations": [], "paths": []}

    async def reason_about_entities(self, entity_ids: List[str]) -> Dict[str, Any]:
        """实体推理"""
        try:
            reasoning_results = {
                "direct_relations": [],
                "indirect_relations": [],
                "patterns": [],
                "insights": []
            }

            # 获取实体间的直接关系
            for i, entity_id1 in enumerate(entity_ids):
                for entity_id2 in entity_ids[i+1:]:
                    direct_relations = [
                        relation for relation in self.relations.values()
                        if ((relation.subject_id == entity_id1 and relation.object_id == entity_id2) or
                            (relation.subject_id == entity_id2 and relation.object_id == entity_id1))
                    ]

                    if direct_relations:
                        reasoning_results["direct_relations"].extend(direct_relations)

            # 获取实体间的间接关系（通过共同邻居）
            common_neighbors = set()
            for entity_id in entity_ids:
                neighbors = set(self.graph.neighbors(entity_id))
                if not common_neighbors:
                    common_neighbors = neighbors
                else:
                    common_neighbors &= neighbors

            for neighbor in common_neighbors:
                if neighbor in self.entities:
                    reasoning_results["indirect_relations"].append({
                        "entity": neighbor,
                        "type": "common_neighbor"
                    })

            # 生成洞察
            if reasoning_results["direct_relations"]:
                reasoning_results["insights"].append("实体间存在直接业务关系")

            if reasoning_results["indirect_relations"]:
                reasoning_results["insights"].append("实体通过共同实体产生间接关联")

            return reasoning_results

        except Exception as e:
            logger.error(f"实体推理失败: {str(e)}")
            return {"direct_relations": [], "indirect_relations": [], "patterns": [], "insights": []}

    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        try:
            entity_type_counts = {}
            for entity in self.entities.values():
                entity_type_counts[entity.type.value] = entity_type_counts.get(entity.type.value, 0) + 1

            relation_type_counts = {}
            for relation in self.relations.values():
                relation_type_counts[relation.relation_type.value] = relation_type_counts.get(relation.relation_type.value, 0) + 1

            return {
                "total_entities": len(self.entities),
                "total_relations": len(self.relations),
                "entity_types": entity_type_counts,
                "relation_types": relation_type_counts,
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "connected_components": nx.number_weakly_connected_components(self.graph)
            }

        except Exception as e:
            logger.error(f"统计信息获取失败: {str(e)}")
            return {}

    async def export_to_rdf(self, output_path: str):
        """导出为RDF格式"""
        try:
            # 创建RDF图
            g = Graph()

            # 定义命名空间
            fin = Namespace("http://example.org/financial#")
            g.bind("fin", fin)

            # 添加实体
            for entity in self.entities.values():
                entity_uri = fin[entity.id]

                # 添加类型
                g.add((entity_uri, fin.type, Literal(entity.type.value)))
                g.add((entity_uri, fin.name, Literal(entity.name)))
                g.add((entity_uri, fin.confidence, Literal(entity.confidence)))

                # 添加属性
                for prop_key, prop_value in entity.properties.items():
                    g.add((entity_uri, fin[prop_key], Literal(str(prop_value))))

            # 添加关系
            for relation in self.relations.values():
                subject_uri = fin[relation.subject_id]
                object_uri = fin[relation.object_id]

                # 添加关系边
                g.add((subject_uri, fin[relation.relation_type.value], object_uri))
                g.add((subject_uri, fin.confidence, Literal(relation.confidence)))

            # 保存到文件
            g.serialize(destination=output_path, format="turtle")
            logger.info(f"RDF导出完成: {output_path}")

        except Exception as e:
            logger.error(f"RDF导出失败: {str(e)}")


# 全局金融知识图谱实例
financial_knowledge_graph = FinancialKnowledgeGraph()