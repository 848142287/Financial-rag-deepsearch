"""
金融关系抽取器 - 券商研报专用
提取金融实体之间的语义关系
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class FinancialRelationType(str, Enum):
    """金融关系类型"""
    # 预测关系
    PREDICTS = "predicts"  # A预测B
    FORECASTS = "forecasts"  # A预测B

    # 相关关系
    POSITIVELY_CORRELATED = "positively_correlated"  # 正相关
    NEGATIVELY_CORRELATED = "negatively_correlated"  # 负相关
    INFLUENCES = "influences"  # A影响B
    DRIVEN_BY = "driven_by"  # A由B驱动

    # 包含关系
    INCLUDES = "includes"  # A包含B
    CONSISTS_OF = "consists_of"  # A由B组成

    # 比较关系
    OUTPERFORMS = "outperforms"  # A优于B
    UNDERPERFORMS = "underperforms"  # A弱于B
    EXCEEDS = "exceeds"  # A超过B
    LAGS = "lags"  # A滞后于B

    # 因果关系
    CAUSES = "causes"  # A导致B
    RESULTS_IN = "results_in"  # A导致B结果

    # 依赖关系
    DEPENDS_ON = "depends_on"  # A依赖B
    SENSITIVE_TO = "sensitive_to"  # A对B敏感

    # 属性关系
    HAS_PROPERTY = "has_property"  # A具有属性B
    MEASURED_BY = "measured_by"  # A用B衡量

@dataclass
class FinancialRelation:
    """金融关系"""
    subject: str  # 主体实体
    relation: FinancialRelationType  # 关系类型
    object: str  # 客体实体
    confidence: float  # 置信度
    source_text: str  # 源文本
    evidence: str = ""  # 证据文本
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RelationExtractionResult:
    """关系抽取结果"""
    relations: List[FinancialRelation]
    stats: Dict[str, Any]
    graph: Dict[str, List[str]] = field(default_factory=dict)  # 简化图结构

class FinancialRelationExtractor:
    """
    金融关系抽取器

    功能：
    1. 基于规则的关系抽取
    2. 基于模式的关系抽取
    3. 共现关系推断
    4. 关系强度计算
    """

    def __init__(self):
        self._compile_patterns()
        self._load_relation_templates()
        self._load_financial_verbs()

    def _compile_patterns(self):
        """预编译关系抽取模式"""
        # 关系指示词
        self.relation_indicators = {
            FinancialRelationType.PREDICTS: [
                r'(预测|预计|预期|展望|forecast|predict)',
                r'将(达到|增长至|rise to)',
            ],
            FinancialRelationType.POSITIVELY_CORRELATED: [
                r'(正相关|positive correlation)',
                r'(随着.*增加.*也增加|rises with)',
                r'(同步上涨|move together)',
            ],
            FinancialRelationType.NEGATIVELY_CORRELATED: [
                r'(负相关|negative correlation)',
                r'(随着.*增加.*而下降|falls as)',
            ],
            FinancialRelationType.INFLUENCES: [
                r'(影响|influence|impact)',
                r'(驱动|drive|driven by)',
                r'(带动|lead to)',
            ],
            FinancialRelationType.OUTPERFORMS: [
                r'(优于|跑赢|outperform|beat)',
                r'(超过|surpass|exceed)',
                r'(强于|stronger than)',
            ],
            FinancialRelationType.UNDERPERFORMS: [
                r'(弱于|跑输|underperform)',
                r'(落后于|lag behind)',
            ],
            FinancialRelationType.CAUSES: [
                r'(导致|引起|cause|lead to)',
                r'(使得|result in)',
                r'(造成|create)',
            ],
            FinancialRelationType.DEPENDS_ON: [
                r'(依赖|depend on|rely on)',
                r'(受.*影响|sensitive to)',
                r'(基于|based on)',
            ],
            FinancialRelationType.HAS_PROPERTY: [
                r'(具有|possess|have)',
                r'(表现出|exhibit|show)',
                r'(呈现|display)',
            ],
        }

    def _load_relation_templates(self):
        """加载关系抽取模板"""
        self.relation_templates = [
            # A预测B
            {
                "type": FinancialRelationType.PREDICTS,
                "patterns": [
                    r'({})\s*(预测|预计|预期)\s*({})',
                    r'({})\s*(将|will)\s*({})',
                ],
                "confidence": 0.85,
            },
            # A影响B
            {
                "type": FinancialRelationType.INFLUENCES,
                "patterns": [
                    r'({})\s*(影响|驱动|impact|drive)\s*({})',
                    r'({})\s*(带动|lead to)\s*({})',
                ],
                "confidence": 0.80,
            },
            # A优于B
            {
                "type": FinancialRelationType.OUTPERFORMS,
                "patterns": [
                    r'({})\s*(优于|跑赢|outperform)\s*({})',
                    r'({})\s*(超过|surpass)\s*({})',
                ],
                "confidence": 0.90,
            },
            # A依赖B
            {
                "type": FinancialRelationType.DEPENDS_ON,
                "patterns": [
                    r'({})\s*(依赖|depend on)\s*({})',
                    r'({})\s*(受.*影响|sensitive to)\s*({})',
                ],
                "confidence": 0.85,
            },
            # A包含B
            {
                "type": FinancialRelationType.INCLUDES,
                "patterns": [
                    r'({})\s*(包括|包含|include)\s*({})',
                    r'({})\s*(由.*组成|consist of)\s*({})',
                ],
                "confidence": 0.90,
            },
        ]

    def _load_financial_verbs(self):
        """加载金融动词"""
        self.financial_verbs = {
            # 增长类
            "growth": ["增长", "上升", "上涨", "increase", "rise", "grow"],
            "decline": ["下降", "下跌", "回落", "decrease", "decline", "fall"],
            # 波动类
            "fluctuate": ["波动", "震荡", "fluctuate", "oscillate"],
            # 稳定类
            "stabilize": ["稳定", "持平", "stabilize", "remain flat"],
            # 预测类
            "predict": ["预测", "预计", "预期", "forecast", "predict", "project"],
            # 影响类
            "impact": ["影响", "驱动", "带动", "impact", "drive", "lead"],
        }

    def extract_relations(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> RelationExtractionResult:
        """
        从文本中提取金融关系

        Args:
            text: 输入文本
            entities: 实体列表（如果为None则自动提取）
            config: 配置参数

        Returns:
            RelationExtractionResult
        """
        config = config or {}
        min_confidence = config.get("min_confidence", 0.7)

        relations = []
        stats = {
            "total_relations": 0,
            "by_type": defaultdict(int),
        }

        # 基于模板抽取
        template_relations = self._extract_by_templates(text, entities)
        relations.extend(template_relations)

        # 基于共现抽取
        if entities and len(entities) >= 2:
            cooccurrence_relations = self._extract_by_cooccurrence(
                text, entities, config.get("cooccurrence_window", 100)
            )
            relations.extend(cooccurrence_relations)

        # 基于动词抽取
        verb_relations = self._extract_by_verbs(text, entities)
        relations.extend(verb_relations)

        # 过滤低置信度关系
        relations = [r for r in relations if r.confidence >= min_confidence]

        # 去重
        relations = self._deduplicate_relations(relations)

        # 统计
        for relation in relations:
            stats["by_type"][relation.relation.value] += 1

        stats["total_relations"] = len(relations)

        # 构建简单图结构
        graph = self._build_relation_graph(relations)

        logger.info(
            f"提取到 {stats['total_relations']} 个金融关系, "
            f"类型分布: {dict(stats['by_type'])}"
        )

        return RelationExtractionResult(
            relations=relations,
            stats=dict(stats),
            graph=graph
        )

    def _extract_by_templates(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> List[FinancialRelation]:
        """基于模板抽取关系"""
        relations = []

        for template in self.relation_templates:
            relation_type = template["type"]
            base_confidence = template["confidence"]

            for pattern in template["patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    try:
                        subject = match.group(1).strip()
                        obj = match.group(3).strip()

                        # 验证实体（如果提供了实体列表）
                        if entities:
                            if subject not in entities and obj not in entities:
                                continue

                        relation = FinancialRelation(
                            subject=subject,
                            relation=relation_type,
                            object=obj,
                            confidence=base_confidence,
                            source_text=match.group(),
                            evidence=text[max(0, match.start() - 50):min(len(text), match.end() + 50)],
                        )

                        relations.append(relation)

                    except IndexError:
                        continue

        return relations

    def _extract_by_cooccurrence(
        self,
        text: str,
        entities: List[str],
        window: int = 100
    ) -> List[FinancialRelation]:
        """基于共现抽取关系"""
        relations = []
        entity_positions = {}

        # 查找所有实体位置
        for entity in entities:
            positions = [m.start() for m in re.finditer(re.escape(entity), text)]
            if positions:
                entity_positions[entity] = positions

        # 检测共现实体对
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1 not in entity_positions or entity2 not in entity_positions:
                    continue

                # 检查是否在窗口内共现
                for pos1 in entity_positions[entity1]:
                    for pos2 in entity_positions[entity2]:
                        distance = abs(pos1 - pos2)

                        if distance <= window:
                            # 推断关系类型
                            relation_type = self._infer_relation_from_context(
                                text, pos1, pos2
                            )

                            relation = FinancialRelation(
                                subject=entity1,
                                relation=relation_type,
                                object=entity2,
                                confidence=0.6,  # 共现关系置信度较低
                                source_text=text[min(pos1, pos2):max(pos1, pos2) + window],
                            )

                            relations.append(relation)
                            break  # 避免重复

        return relations

    def _infer_relation_from_context(
        self,
        text: str,
        pos1: int,
        pos2: int
    ) -> FinancialRelationType:
        """从上下文推断关系类型"""
        start = min(pos1, pos2)
        end = max(pos1, pos2)
        context = text[start:end + 50]

        # 检查关系指示词
        for relation_type, patterns in self.relation_indicators.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    return relation_type

        # 默认返回影响关系
        return FinancialRelationType.INFLUENCES

    def _extract_by_verbs(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> List[FinancialRelation]:
        """基于动词抽取关系"""
        relations = []

        # 简单实现：查找 "A 影响 B" 模式
        for verb_category, verbs in self.financial_verbs.items():
            for verb in verbs:
                # 查找动词附近的实体
                pattern = rf'([^，。]{0,20})\s*{verb}\s*([^，。]{0,20})'
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    subject = match.group(1).strip()
                    obj = match.group(2).strip()

                    # 简单过滤
                    if len(subject) < 2 or len(obj) < 2:
                        continue

                    # 映射到关系类型
                    relation_type = self._map_verb_to_relation(verb_category)

                    relation = FinancialRelation(
                        subject=subject,
                        relation=relation_type,
                        object=obj,
                        confidence=0.65,
                        source_text=match.group(),
                    )

                    relations.append(relation)

        return relations

    def _map_verb_to_relation(
        self,
        verb_category: str
    ) -> FinancialRelationType:
        """将动词类别映射到关系类型"""
        mapping = {
            "growth": FinancialRelationType.PREDICTS,
            "decline": FinancialRelationType.PREDICTS,
            "predict": FinancialRelationType.PREDICTS,
            "impact": FinancialRelationType.INFLUENCES,
        }

        return mapping.get(verb_category, FinancialRelationType.INFLUENCES)

    def _deduplicate_relations(
        self,
        relations: List[FinancialRelation]
    ) -> List[FinancialRelation]:
        """去重关系（保留最高置信度）"""
        relation_map = {}

        for relation in relations:
            key = (relation.subject, relation.relation, relation.object)

            if key not in relation_map or relation.confidence > relation_map[key].confidence:
                relation_map[key] = relation

        return list(relation_map.values())

    def _build_relation_graph(
        self,
        relations: List[FinancialRelation]
    ) -> Dict[str, List[str]]:
        """构建关系图"""
        graph = defaultdict(list)

        for relation in relations:
            graph[relation.subject].append(relation.object)
            # 如果需要无向图，可以添加反向边
            # graph[relation.object].append(relation.subject)

        return dict(graph)

    def get_relation_patterns(
        self,
        relations: List[FinancialRelation]
    ) -> Dict[str, Any]:
        """
        分析关系模式

        Args:
            relations: 关系列表

        Returns:
            模式分析结果
        """
        patterns = {
            "most_common_relations": defaultdict(int),
            "most_connected_entities": defaultdict(int),
            "relation_types_distribution": defaultdict(int),
        }

        # 统计最常见的关系
        for relation in relations:
            key = f"{relation.relation.value}"
            patterns["most_common_relations"][key] += 1

            # 统计实体连接数
            patterns["most_connected_entities"][relation.subject] += 1
            patterns["relation_types_distribution"][relation.relation.value] += 1

        # 排序
        patterns["most_common_relations"] = dict(
            sorted(
                patterns["most_common_relations"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        )

        patterns["most_connected_entities"] = dict(
            sorted(
                patterns["most_connected_entities"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        )

        return patterns

    async def extract_relations_batch(
        self,
        texts: List[str],
        entities_list: Optional[List[List[str]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> List[RelationExtractionResult]:
        """
        批量提取关系

        Args:
            texts: 文本列表
            entities_list: 实体列表列表（与texts一一对应）
            config: 配置参数

        Returns:
            RelationExtractionResult列表
        """
        results = []

        for i, text in enumerate(texts):
            entities = entities_list[i] if entities_list else None
            result = self.extract_relations(text, entities, config)
            results.append(result)

        return results

# 全局实例
_financial_relation_extractor: Optional[FinancialRelationExtractor] = None

def get_financial_relation_extractor() -> FinancialRelationExtractor:
    """获取全局金融关系抽取器"""
    global _financial_relation_extractor
    if _financial_relation_extractor is None:
        _financial_relation_extractor = FinancialRelationExtractor()
    return _financial_relation_extractor
