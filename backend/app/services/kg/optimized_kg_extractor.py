"""
优化的知识图谱抽取服务
提升Neo4j知识图谱质量和准确性
"""

from dataclasses import dataclass
from enum import Enum
import re
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

class EntityType(Enum):
    """实体类型"""
    COMPANY = "公司"
    PERSON = "人物"
    PRODUCT = "产品"
    FINANCIAL_INDICATOR = "财务指标"
    TIME_PERIOD = "时间段"
    DEPARTMENT = "部门"
    PROJECT = "项目"
    LOCATION = "地点"

class RelationType(Enum):
    """关系类型"""
    INVEST = "投资"
    COOPERATE = "合作"
    SUPPLY = "供应"
    COMPETE = "竞争"
    MANAGEMENT = "管理"
    SUBSIDIARY = "子公司"
    PARTNERSHIP = "合伙"
    EMPLOYMENT = "雇佣"

@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    type: EntityType
    properties: Dict[str, Any]
    document_id: str
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'properties': self.properties,
            'document_id': self.document_id,
            'confidence': self.confidence
        }

@dataclass
class Relationship:
    """关系"""
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any]
    document_id: str
    confidence: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type.value,
            'properties': self.properties,
            'document_id': self.document_id,
            'confidence': self.confidence
        }

class OptimizedKGExtractor:
    """
    优化的知识图谱抽取器

    优化点：
    1. 基于规则的实体识别（高准确率）
    2. 上下文感知的关系抽取
    3. 财务领域专用模式
    4. 实体去重和合并
    5. 关系置信度评分
    """

    def __init__(self):
        """初始化抽取器"""
        # 编译正则表达式
        self._compile_patterns()

    def _compile_patterns(self):
        """预编译正则表达式模式"""

        # 1. 公司名称模式
        self.company_patterns = [
            re.compile(r'([\u4e00-\u9fa5]{2,10})(股份有限公司|集团有限公司|有限公司|科技公司|实业公司)'),
            re.compile(r'([\u4e00-\u9fa5]{2,15})公司'),
        ]

        # 2. 人物模式
        self.person_patterns = [
            re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)'),  # 英文名
            re.compile(r'([\u4e00-\u9fa5]{2,4})(先生|女士|总裁|CEO|董事长|总经理|总监|经理)'),
        ]

        # 3. 财务指标模式
        self.metric_patterns = [
            re.compile(r'(营业收入|净利润|总资产|净资产|现金流|毛利率|净利率|ROE|ROA)[：:]\s*([0-9.,]+\s*[亿元千百万元%]?)'),
            re.compile(r'([0-9.,]+\s*[亿元千百万元%]?)\s*(的|的)?(营业收入|净利润|总资产|净资产|现金流|增长率)'),
        ]

        # 4. 时间模式
        self.time_patterns = [
            re.compile(r'(\d{4}年\d{1,2}季度|\d{4}年\d{1,2}月|\d{4}年上半年|\d{4}年下半年)'),
            re.compile(r'(本期|上期|去年同期|本季度|上季度)'),
        ]

        # 5. 关系模式
        self.relation_patterns = [
            # 投资关系
            re.compile(r'([\u4e00-\u9fa5]{2,10}公司)(投资|控股|参股|持股)([\u4e00-\u9fa5]{2,10}公司)'),
            # 合作关系
            re.compile(r'([\u4e00-\u9fa5]{2,10}公司)(与|和)([\u4e00-\u9fa5]{2,10}公司)(合作|达成|签署)'),
            # 人员关系
            re.compile(r'([\u4e00-\u9fa5]{2,4})(担任|出任|任命|被选为)([\u4e00-\u9fa5]{2,10})(总裁|CEO|董事长|总经理)'),
        ]

    async def extract_entities(
        self,
        text: str,
        document_id: str
    ) -> List[Entity]:
        """
        抽取实体

        Args:
            text: 文本内容
            document_id: 文档ID

        Returns:
            实体列表
        """
        entities = []
        entity_names: Set[str] = set()

        # 1. 抽取公司实体
        companies = self._extract_companies(text, document_id)
        for entity in companies:
            if entity.name not in entity_names:
                entities.append(entity)
                entity_names.add(entity.name)

        # 2. 抽取人物实体
        persons = self._extract_persons(text, document_id)
        for entity in persons:
            if entity.name not in entity_names:
                entities.append(entity)
                entity_names.add(entity.name)

        # 3. 抽取财务指标
        metrics = self._extract_financial_metrics(text, document_id)
        for entity in metrics:
            if entity.name not in entity_names:
                entities.append(entity)
                entity_names.add(entity.name)

        # 4. 抽取时间实体
        times = self._extract_times(text, document_id)
        for entity in times:
            if entity.name not in entity_names:
                entities.append(entity)
                entity_names.add(entity.name)

        logger.info(f"✅ 抽取到 {len(entities)} 个实体 (公司:{len(companies)}, 人物:{len(persons)}, 指标:{len(metrics)}, 时间:{len(times)})")

        return entities

    def _extract_companies(self, text: str, document_id: str) -> List[Entity]:
        """抽取公司实体"""
        companies = []

        for pattern in self.company_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                company_name = match.group(0)
                # 去除常见噪音
                if len(company_name) < 3:
                    continue
                if '该公司' in company_name or '公司公司' in company_name:
                    continue

                entity_id = f"{document_id}_company_{hash(company_name) % 1000000}"

                companies.append(Entity(
                    id=entity_id,
                    name=company_name,
                    type=EntityType.COMPANY,
                    properties={
                        'matched_text': company_name,
                        'position': match.span()
                    },
                    document_id=document_id,
                    confidence=0.85
                ))

        return companies

    def _extract_persons(self, text: str, document_id: str) -> List[Entity]:
        """抽取人物实体"""
        persons = []

        for pattern in self.person_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                person_name = match.group(1)
                if len(person_name) < 2:
                    continue

                entity_id = f"{document_id}_person_{hash(person_name) % 1000000}"

                persons.append(Entity(
                    id=entity_id,
                    name=person_name,
                    type=EntityType.PERSON,
                    properties={
                        'matched_text': match.group(0),
                        'position': match.span()
                    },
                    document_id=document_id,
                    confidence=0.80
                ))

        return persons

    def _extract_financial_metrics(self, text: str, document_id: str) -> List[Entity]:
        """抽取财务指标"""
        metrics = []

        for pattern in self.metric_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                metric_name = match.group(1)
                metric_value = match.group(2) if len(match.groups()) >= 2 else ""

                entity_id = f"{document_id}_metric_{hash(metric_name) % 1000000}"

                metrics.append(Entity(
                    id=entity_id,
                    name=metric_name,
                    type=EntityType.FINANCIAL_INDICATOR,
                    properties={
                        'value': metric_value,
                        'matched_text': match.group(0),
                        'position': match.span()
                    },
                    document_id=document_id,
                    confidence=0.90
                ))

        return metrics

    def _extract_times(self, text: str, document_id: str) -> List[Entity]:
        """抽取时间实体"""
        times = []

        for pattern in self.time_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                time_expr = match.group(0)

                entity_id = f"{document_id}_time_{hash(time_expr) % 1000000}"

                times.append(Entity(
                    id=entity_id,
                    name=time_expr,
                    type=EntityType.TIME_PERIOD,
                    properties={
                        'matched_text': time_expr,
                        'position': match.span()
                    },
                    document_id=document_id,
                    confidence=0.95
                ))

        return times

    async def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        document_id: str
    ) -> List[Relationship]:
        """
        抽取关系

        Args:
            text: 文本内容
            entities: 已抽取的实体
            document_id: 文档ID

        Returns:
            关系列表
        """
        relationships = []

        # 构建实体名称到ID的映射
        entity_map = {entity.name: entity.id for entity in entities}

        # 使用关系模式抽取
        for pattern in self.relation_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                groups = match.groups()

                if len(groups) >= 3:
                    source_name = groups[0]
                    relation_text = groups[1] if len(groups) > 1 else ""
                    target_name = groups[2] if len(groups) > 2 else ""

                    # 检查实体是否存在
                    if source_name not in entity_map or target_name not in entity_map:
                        continue

                    # 推断关系类型
                    relation_type = self._infer_relation_type(relation_text)

                    source_id = entity_map[source_name]
                    target_id = entity_map[target_name]

                    relationships.append(Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        type=relation_type,
                        properties={
                            'matched_text': match.group(0),
                            'relation_text': relation_text,
                            'position': match.span()
                        },
                        document_id=document_id,
                        confidence=0.75
                    ))

        logger.info(f"✅ 抽取到 {len(relationships)} 个关系")

        return relationships

    def _infer_relation_type(self, relation_text: str) -> RelationType:
        """根据关系文本推断关系类型"""
        if '投资' in relation_text or '控股' in relation_text or '参股' in relation_text:
            return RelationType.INVEST
        elif '合作' in relation_text or '签署' in relation_text or '达成' in relation_text:
            return RelationType.COOPERATE
        elif '担任' in relation_text or '出任' in relation_text or '任命' in relation_text:
            return RelationType.EMPLOYMENT
        else:
            return RelationType.COOPERATE  # 默认合作关系

def get_optimized_kg_extractor() -> OptimizedKGExtractor:
    """获取优化的知识图谱抽取器实例"""
    return OptimizedKGExtractor()

__all__ = [
    'OptimizedKGExtractor',
    'get_optimized_kg_extractor',
    'Entity',
    'Relationship',
    'EntityType',
    'RelationType'
]
