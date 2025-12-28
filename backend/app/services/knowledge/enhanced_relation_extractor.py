"""
增强的关系抽取器
支持方向性判断、依存句法分析和LLM增强
"""
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.core.graph_config import (
    GraphRelationType,
    GraphEntityType,
    graph_relation_config,
    generate_relation_id
)

logger = logging.getLogger(__name__)


class RelationDirection(Enum):
    """关系方向"""
    FORWARD = "forward"       # entity1 -> relation -> entity2
    BACKWARD = "backward"     # entity2 -> relation -> entity1
    BIDIRECTIONAL = "bidirectional"  # 双向关系
    UNKNOWN = "unknown"       # 无法确定方向


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    id: str
    subject: str                    # 主体实体
    object: str                     # 客体实体
    relation_type: GraphRelationType
    direction: RelationDirection
    confidence: float
    evidence: str                   # 证据文本
    source_chunk_id: str
    metadata: Dict = None


class EnhancedRelationExtractor:
    """增强的关系抽取器"""

    def __init__(self):
        self.config = graph_relation_config

        # 关系方向性指示词
        self.direction_indicators = self._load_direction_indicators()

        # 关系模式（增强版）
        self.relation_patterns = self._load_relation_patterns()

    def _load_direction_indicators(self) -> Dict[GraphRelationType, Dict[str, str]]:
        """
        加载关系方向性指示词

        Returns:
            {
                relation_type: {
                    "forward": [指示词列表],  # entity1 -> relation -> entity2
                    "backward": [指示词列表]  # entity2 -> relation -> entity1
                }
            }
        """
        return {
            GraphRelationType.OWNS: {
                "forward": ["拥有", "持有", "控制", "owns", "holds"],
                "backward": ["被拥有", "由...拥有", "owned by", "held by"]
            },
            GraphRelationType.INVESTS_IN: {
                "forward": ["投资", "参股", "入股", "invests in", "funds"],
                "backward": ["被投资", "获...投资", "backed by", "funded by"]
            },
            GraphRelationType.SUBSIDIARY_OF: {
                "forward": ["是...的子公司", "隶属于", "属于", "subsidiary of"],
                "backward": ["拥有", "是...的母公司", "parent of"]
            },
            GraphRelationType.WORKS_FOR: {
                "forward": ["任职于", "担任", "在...工作", "works for", "employed by"],
                "backward": ["雇佣", "聘请", "employs", "hires"]
            },
            GraphRelationType.CEO_OF: {
                "forward": ["是...的CEO", "担任...CEO", "CEO of"],
                "backward": ["CEO是", "CEO为"]
            },
            GraphRelationType.PARTNER_OF: {
                "forward": ["与...合作", "与...合资", "partners with"],
                "backward": ["与...合作", "与...合资", "partners with"]  # 双向
            },
            GraphRelationType.COMPETITOR_OF: {
                "forward": ["与...竞争", "是...的竞争对手", "competes with"],
                "backward": ["与...竞争", "是...的竞争对手"]  # 双向
            },
            GraphRelationType.LOCATED_IN: {
                "forward": ["位于", "在", "总部在", "located in"],
                "backward": ["包含", "管辖", "contains"]
            },
            GraphRelationType.AFFECTS: {
                "forward": ["影响", "作用于", "affects", "impacts"],
                "backward": ["受...影响", "被影响", "affected by"]
            },
        }

    def _load_relation_patterns(self) -> List[Dict]:
        """
        加载增强的关系抽取模式

        Returns:
            关系模式列表
        """
        return [
            # 投资关系
            {
                "type": GraphRelationType.INVESTS_IN,
                "patterns": [
                    r"(.+?)投资(?:了)?(.+?)(?:\d+(?:亿|万|千万)?(?:元|美元)?)?",
                    r"(.+?)入股(.+?)",
                    r"(.+?)参股(.+?)",
                    r"(.+?)对(.+?)进行投资",
                    r"(.+?)invests?(?:\s+\$?\d+\s+(?:million|billion)?)?\s+in(.+?)",
                ],
                "entity_types": [(GraphEntityType.COMPANY, GraphEntityType.COMPANY),
                                (GraphEntityType.COMPANY, GraphEntityType.STOCK),
                                (GraphEntityType.PERSON, GraphEntityType.COMPANY)]
            },
            # 收购关系
            {
                "type": GraphRelationType.SUBSIDIARY_OF,
                "patterns": [
                    r"(.+?)收购(?:了)?(.+?)(?:\d+(?:%|(?:亿|万|千万)?(?:元|美元)?)?)?",
                    r"(.+?)并购(.+?)",
                    r"(.+?)是(.+?)的子公司",
                    r"(.+?)隶属于(.+?)",
                    r"(.+?)由(.+?)控股",
                ],
                "entity_types": [(GraphEntityType.COMPANY, GraphEntityType.COMPANY)]
            },
            # 任职关系
            {
                "type": GraphRelationType.WORKS_FOR,
                "patterns": [
                    r"(.+?)任职(?:于)?(.+?)",
                    r"(.+?)担任(.+?)(?:董事长|CEO|总裁|总经理|行长|总监)?",
                    r"(.+?)是(.+?)的(?:董事长|CEO|总裁|总经理|行长)",
                    r"(.+?)works?\s+for(?:\s+the\s+)?(.+?)",
                    r"(.+?)was\s+appointed\s+(?:as\s+)?(?:CEO|President|Director)\s+of\s+(.+?)",
                ],
                "entity_types": [(GraphEntityType.PERSON, GraphEntityType.COMPANY),
                                (GraphEntityType.PERSON, GraphEntityType.ORGANIZATION)]
            },
            # 合作关系
            {
                "type": GraphRelationType.PARTNER_OF,
                "patterns": [
                    r"(.+?)与(.+?)合作",
                    r"(.+?)和(.+?)建立(?:战略)?合作伙伴关系",
                    r"(.+?)与(.+?)合资(?:成立)?",
                    r"(.+?)partners?\s+with(.+?)",
                    r"(.+?)joint\s+venture\s+with(.+?)",
                ],
                "entity_types": [(GraphEntityType.COMPANY, GraphEntityType.COMPANY)]
            },
            # 竞争关系
            {
                "type": GraphRelationType.COMPETITOR_OF,
                "patterns": [
                    r"(.+?)与(.+?)竞争",
                    r"(.+?)是(.+?)的竞争对手",
                    r"(.+?)与(.+?)展开竞争",
                    r"(.+?)competes?\s+with(.+?)",
                ],
                "entity_types": [(GraphEntityType.COMPANY, GraphEntityType.COMPANY)]
            },
            # 地理位置关系
            {
                "type": GraphRelationType.LOCATED_IN,
                "patterns": [
                    r"(.+?)位于(.+?)",
                    r"(.+?)在(.+?)(?:市|省|县|区)?",
                    r"(.+?)总部在(.+?)",
                    r"(.+?)(?:市|省|县|区)的(.+?)",
                ],
                "entity_types": [(GraphEntityType.COMPANY, GraphEntityType.LOCATION),
                                (GraphEntityType.PERSON, GraphEntityType.LOCATION)]
            },
            # 拥有关系
            {
                "type": GraphRelationType.OWNS,
                "patterns": [
                    r"(.+?)拥有(.+?)",
                    r"(.+?)持有(.+?)(?:\d+%?|\d+(?:亿|万|千万)?(?:股|份))?",
                    r"(.+?)owns?(?:\s+\d+%?)?\s+(.+?)",
                    r"(.+?)holds?(?:\s+\d+%?)?\s+(.+?)",
                ],
                "entity_types": [(GraphEntityType.COMPANY, GraphEntityType.STOCK),
                                (GraphEntityType.COMPANY, GraphEntityType.COMPANY),
                                (GraphEntityType.PERSON, GraphEntityType.STOCK)]
            },
            # 监管关系
            {
                "type": GraphRelationType.REGULATED_BY,
                "patterns": [
                    r"(.+?)受(.+?)监管",
                    r"(.+?)由(.+?)(?:监管|管辖)",
                    r"(.+?)遵守(.+?)的规定",
                    r"(.+?)regulated\s+by(.+?)",
                ],
                "entity_types": [(GraphEntityType.COMPANY, GraphEntityType.REGULATION),
                                (GraphEntityType.COMPANY, GraphEntityType.ORGANIZATION)]
            },
        ]

    async def extract_relations(
        self,
        text: str,
        entities: List[Dict],
        chunk_id: str,
        use_llm: bool = True
    ) -> List[ExtractedRelation]:
        """
        抽取实体间的关系

        Args:
            text: 文本内容
            entities: 实体列表（包含name, type, start, end等）
            chunk_id: 文档块ID
            use_llm: 是否使用LLM增强抽取

        Returns:
            抽取的关系列表
        """
        relations = []

        try:
            # 方法1: 基于模式的关系抽取
            if self.config.use_rule_based:
                rule_relations = await self._extract_by_patterns(
                    text, entities, chunk_id
                )
                relations.extend(rule_relations)

            # 方法2: 依存句法分析（如果启用）
            if self.config.use_dependency_parsing:
                dependency_relations = await self._extract_by_dependencies(
                    text, entities, chunk_id
                )
                relations.extend(dependency_relations)

            # 方法3: LLM增强抽取（如果启用）
            if use_llm and self.config.use_llm:
                llm_relations = await self._extract_by_llm(
                    text, entities, chunk_id
                )
                relations.extend(llm_relations)

            # 去重和合并
            relations = await self._deduplicate_relations(relations)

            logger.info(f"抽取到 {len(relations)} 个关系 (chunk: {chunk_id})")
            return relations

        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []

    async def _extract_by_patterns(
        self,
        text: str,
        entities: List[Dict],
        chunk_id: str
    ) -> List[ExtractedRelation]:
        """基于模式的关系抽取"""
        relations = []

        # 构建实体索引
        entity_index = {
            (e["start"], e["end"]): e
            for e in entities
        }

        # 遍历所有关系模式
        for pattern_info in self.relation_patterns:
            relation_type = pattern_info["type"]
            patterns = pattern_info["patterns"]
            allowed_types = pattern_info.get("entity_types", [])

            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    try:
                        # 提取主体和客体
                        if len(match.groups()) >= 2:
                            subject_text = match.group(1).strip()
                            object_text = match.group(2).strip()

                            # 查找对应的实体
                            subject_entity = self._find_entity_by_text(
                                subject_text, entities
                            )
                            object_entity = self._find_entity_by_text(
                                object_text, entities
                            )

                            if not subject_entity or not object_entity:
                                continue

                            # 检查实体类型是否匹配
                            if allowed_types:
                                type_match = any(
                                    subject_entity["type"] == t1 and
                                    object_entity["type"] == t2
                                    for t1, t2 in allowed_types
                                )
                                if not type_match:
                                    continue

                            # 判断方向
                            direction = self._detect_direction(
                                match.group(0),
                                relation_type
                            )

                            # 计算置信度
                            confidence = self._calculate_pattern_confidence(
                                match,
                                pattern,
                                relation_type,
                                direction
                            )

                            # 过滤低置信度关系
                            if confidence < self.config.min_confidence:
                                continue

                            # 创建关系
                            relation = ExtractedRelation(
                                id=generate_relation_id(
                                    subject_entity["id"],
                                    object_entity["id"],
                                    relation_type,
                                    match.group(0)
                                ),
                                subject=subject_entity["name"],
                                object=object_entity["name"],
                                relation_type=relation_type,
                                direction=direction,
                                confidence=confidence,
                                evidence=match.group(0),
                                source_chunk_id=chunk_id,
                                metadata={
                                    "extraction_method": "pattern",
                                    "pattern": pattern,
                                    "subject_type": subject_entity["type"],
                                    "object_type": object_entity["type"]
                                }
                            )

                            relations.append(relation)

                    except Exception as e:
                        logger.warning(f"处理匹配失败: {e}")
                        continue

        return relations

    def _find_entity_by_text(
        self,
        text: str,
        entities: List[Dict]
    ) -> Optional[Dict]:
        """根据文本查找实体（支持模糊匹配）"""
        # 精确匹配
        for entity in entities:
            if entity["name"] == text:
                return entity

        # 模糊匹配（包含关系）
        for entity in entities:
            if text in entity["name"] or entity["name"] in text:
                return entity

        # 部分匹配（至少3个字符）
        if len(text) >= 3:
            for entity in entities:
                if text[:3] in entity["name"] or entity["name"][:3] in text:
                    return entity

        return None

    def _detect_direction(
        self,
        relation_text: str,
        relation_type: GraphRelationType
    ) -> RelationDirection:
        """
        检测关系方向

        Args:
            relation_text: 关系文本
            relation_type: 关系类型

        Returns:
            关系方向
        """
        if not self.config.enable_direction_detection:
            return RelationDirection.UNKNOWN

        # 获取方向性指示词
        indicators = self.direction_indicators.get(relation_type, {})

        if not indicators:
            return RelationDirection.UNKNOWN

        # 检查前向指示词
        forward_keywords = indicators.get("forward", [])
        for keyword in forward_keywords:
            if keyword in relation_text:
                return RelationDirection.FORWARD

        # 检查后向指示词
        backward_keywords = indicators.get("backward", [])
        for keyword in backward_keywords:
            if keyword in relation_text:
                return RelationDirection.BACKWARD

        # 双向关系
        bidirectional_types = [
            GraphRelationType.PARTNER_OF,
            GraphRelationType.COMPETITOR_OF
        ]
        if relation_type in bidirectional_types:
            return RelationDirection.BIDIRECTIONAL

        return RelationDirection.UNKNOWN

    def _calculate_pattern_confidence(
        self,
        match: re.Match,
        pattern: str,
        relation_type: GraphRelationType,
        direction: RelationDirection
    ) -> float:
        """计算模式匹配的置信度"""
        base_confidence = 0.7

        # 模式复杂度加成
        if len(pattern) > 50:
            base_confidence += 0.1  # 复杂模式，高置信度

        # 方向性加成
        if direction != RelationDirection.UNKNOWN:
            base_confidence += 0.1

        # 关系类型特定加成
        high_confidence_types = [
            GraphRelationType.INVESTS_IN,
            GraphRelationType.SUBSIDIARY_OF,
            GraphRelationType.WORKS_FOR
        ]
        if relation_type in high_confidence_types:
            base_confidence += 0.05

        return min(1.0, base_confidence)

    async def _extract_by_dependencies(
        self,
        text: str,
        entities: List[Dict],
        chunk_id: str
    ) -> List[ExtractedRelation]:
        """基于依存句法的关系抽取（简化实现）"""
        # 这里应该集成依存句法分析库（如spaCy、Stanza）
        # 简化实现：返回空列表
        # TODO: 实现完整的依存句法分析
        return []

    async def _extract_by_llm(
        self,
        text: str,
        entities: List[Dict],
        chunk_id: str
    ) -> List[ExtractedRelation]:
        """使用LLM进行关系抽取"""
        try:
            from app.services.qwen_embedding_service import QwenEmbeddingService

            # 构建实体列表描述
            entity_desc = "\n".join([
                f"- {i+1}. {e['name']} (类型: {e['type']})"
                for i, e in enumerate(entities)
            ])

            # 构建prompt
            prompt = f"""
你是一个金融领域的专家，擅长分析实体间的关系。

请从以下文本中抽取实体间的关系。

实体列表：
{entity_desc}

文本内容：
{text[:2000]}

请识别实体间的关系，包括：
1. 投资关系 (INVESTS_IN)
2. 任职关系 (WORKS_FOR, CEO_OF)
3. 所有权关系 (OWNS, SUBSIDIARY_OF)
4. 合作关系 (PARTNER_OF)
5. 竞争关系 (COMPETITOR_OF)
6. 地理关系 (LOCATED_IN)
7. 监管关系 (REGULATED_BY)

请以JSON格式返回，格式如下：
{{
    "relations": [
        {{
            "subject": "实体1名称",
            "object": "实体2名称",
            "relation_type": "关系类型",
            "direction": "forward|backward|bidirectional",
            "confidence": 0.0-1.0,
            "evidence": "证据文本"
        }}
    ]
}}
"""

            # 调用LLM（使用已有的LLM服务）
            llm_service = QwenEmbeddingService()
            response = await llm_service.simple_chat(
                prompt=prompt,
                temperature=0.1
            )

            # 解析响应
            relations = await self._parse_llm_response(
                response,
                entities,
                chunk_id
            )

            return relations

        except Exception as e:
            logger.error(f"LLM关系抽取失败: {e}")
            return []

    async def _parse_llm_response(
        self,
        response: str,
        entities: List[Dict],
        chunk_id: str
    ) -> List[ExtractedRelation]:
        """解析LLM响应"""
        import json

        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("LLM响应中未找到JSON")
                return []

            result = json.loads(json_match.group(0))
            relations_data = result.get("relations", [])

            relations = []
            for rel_data in relations_data:
                try:
                    # 映射关系类型
                    relation_type_str = rel_data.get("relation_type", "")
                    try:
                        relation_type = GraphRelationType[relation_type_str]
                    except KeyError:
                        logger.warning(f"未知的关系类型: {relation_type_str}")
                        continue

                    # 映射方向
                    direction_str = rel_data.get("direction", "unknown")
                    try:
                        direction = RelationDirection[direction_str.upper()]
                    except KeyError:
                        direction = RelationDirection.UNKNOWN

                    # 验证实体存在
                    subject_name = rel_data.get("subject", "")
                    object_name = rel_data.get("object", "")
                    if not subject_name or not object_name:
                        continue

                    # 检查实体是否在列表中
                    subject_entity = self._find_entity_by_text(subject_name, entities)
                    object_entity = self._find_entity_by_text(object_name, entities)

                    if not subject_entity or not object_entity:
                        continue

                    # 计算置信度（LLM加成）
                    base_confidence = float(rel_data.get("confidence", 0.7))
                    confidence = min(1.0, base_confidence + self.config.llm_confidence_boost)

                    # 创建关系
                    relation = ExtractedRelation(
                        id=generate_relation_id(
                            subject_entity["id"],
                            object_entity["id"],
                            relation_type,
                            rel_data.get("evidence", "")
                        ),
                        subject=subject_name,
                        object=object_name,
                        relation_type=relation_type,
                        direction=direction,
                        confidence=confidence,
                        evidence=rel_data.get("evidence", ""),
                        source_chunk_id=chunk_id,
                        metadata={
                            "extraction_method": "llm",
                            "subject_type": subject_entity["type"],
                            "object_type": object_entity["type"]
                        }
                    )

                    relations.append(relation)

                except Exception as e:
                    logger.warning(f"解析单个关系失败: {e}")
                    continue

            logger.info(f"LLM抽取到 {len(relations)} 个关系")
            return relations

        except json.JSONDecodeError as e:
            logger.error(f"解析LLM响应JSON失败: {e}")
            return []

    async def _deduplicate_relations(
        self,
        relations: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """去重和合并关系"""
        if not relations:
            return []

        # 使用字典去重（key: subject+object+type）
        relation_dict = {}

        for relation in relations:
            key = (relation.subject, relation.object, relation.relation_type)

            if key not in relation_dict:
                relation_dict[key] = relation
            else:
                # 合并：保留置信度更高的
                existing = relation_dict[key]
                if relation.confidence > existing.confidence:
                    relation_dict[key] = relation

        return list(relation_dict.values())


# 全局实例
enhanced_relation_extractor = EnhancedRelationExtractor()
