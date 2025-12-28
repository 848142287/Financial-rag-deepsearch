"""
Neo4j 知识图谱统一配置
确保所有服务使用一致的配置和ID生成策略
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import uuid


class GraphEntityType(str, Enum):
    """图谱实体类型"""
    COMPANY = "Company"
    PERSON = "Person"
    STOCK = "Stock"
    BOND = "Bond"
    FUND = "Fund"
    FINANCIAL_INDICATOR = "FinancialIndicator"
    FINANCIAL_TERM = "FinancialTerm"
    MONEY = "Money"
    PERCENTAGE = "Percentage"
    LOCATION = "Location"
    INDUSTRY = "Industry"
    PRODUCT = "Product"
    EVENT = "Event"
    CONCEPT = "Concept"
    REGULATION = "Regulation"
    TIME = "Time"


class GraphRelationType(str, Enum):
    """图谱关系类型"""
    # 所有权关系
    OWNS = "OWNS"                    # 拥有
    HOLDS = "HOLDS"                  # 持有
    SUBSIDIARY_OF = "SUBSIDIARY_OF"  # 子公司关系
    PARENT_OF = "PARENT_OF"          # 父公司关系

    # 任职关系
    WORKS_FOR = "WORKS_FOR"          # 任职于
    CEO_OF = "CEO_OF"                # CEO
    DIRECTOR_OF = "DIRECTOR_OF"      # 董事
    EMPLOYEE_OF = "EMPLOYEE_OF"      # 员工

    # 投资关系
    INVESTS_IN = "INVESTS_IN"        # 投资于
    BACKED_BY = "BACKED_BY"          # 被投资

    # 合作关系
    PARTNER_OF = "PARTNER_OF"        # 合作伙伴
    JOINT_VENTURE_WITH = "JOINT_VENTURE_WITH"  # 合资
    ALLIES_WITH = "ALLIES_WITH"      # 联盟

    # 竞争关系
    COMPETITOR_OF = "COMPETITOR_OF"  # 竞争对手
    RIVAL_OF = "RIVAL_OF"            # 对手

    # 地理位置
    LOCATED_IN = "LOCATED_IN"        # 位于
    HEADQUARTERED_IN = "HEADQUARTERED_IN"  # 总部在
    OPERATES_IN = "OPERATES_IN"      # 运营于

    # 分类关系
    BELONGS_TO = "BELONGS_TO"        # 属于
    IS_TYPE_OF = "IS_TYPE_OF"        # 是...的类型
    CATEGORY_OF = "CATEGORY_OF"      # 类别

    # 影响/因果关系
    AFFECTS = "AFFECTS"              # 影响
    CAUSES = "CAUSES"                # 导致
    INFLUENCES = "INFLUENCES"        # 影响

    # 监管关系
    REGULATED_BY = "REGULATED_BY"    # 受...监管
    REGULATES = "REGULATES"          # 监管

    # 度量关系
    MEASURES = "MEASURES"            # 衡量
    INDICATES = "INDICATES"          # 指示

    # 涉及关系
    INVOLVES = "INVOLVES"            # 涉及
    PARTICIPATES_IN = "PARTICIPATES_IN"  # 参与

    # 事件关系
    OCCURRED_AT = "OCCURRED_AT"      # 发生于
    RELATED_TO = "RELATED_TO"        # 相关


@dataclass
class EntityExtractionConfig:
    """实体抽取配置"""
    # 抽取方法
    use_rule_based: bool = True           # 使用规则抽取
    use_regex: bool = True                # 使用正则表达式
    use_ner: bool = True                  # 使用NER模型
    use_llm: bool = True                  # 使用LLM抽取（默认启用！）

    # 置信度阈值
    min_confidence: float = 0.5           # 最低置信度
    llm_confidence_boost: float = 0.2     # LLM置信度加成

    # 实体类型限制
    enabled_entity_types: Set[GraphEntityType] = field(default_factory=lambda: {
        GraphEntityType.COMPANY,
        GraphEntityType.PERSON,
        GraphEntityType.STOCK,
        GraphEntityType.FINANCIAL_INDICATOR,
        GraphEntityType.FINANCIAL_TERM,
        GraphEntityType.LOCATION,
        GraphEntityType.INDUSTRY,
        GraphEntityType.TIME,
        GraphEntityType.MONEY,
        GraphEntityType.PERCENTAGE
    })

    # 消歧配置
    enable_cross_document_disambiguation: bool = True   # 跨文档消歧
    entity_similarity_threshold: float = 0.85           # 实体相似度阈值
    enable_alias_normalization: bool = True              # 别名归一化

    # 批处理配置
    batch_size: int = 10                  # 批处理大小
    max_entities_per_chunk: int = 50      # 每个chunk最多抽取实体数

    # NER模型配置
    spacy_model: str = "zh_core_web_sm"   # spaCy中文模型
    fallback_spacy_model: str = "en_core_web_sm"  # 备用英文模型


@dataclass
class RelationExtractionConfig:
    """关系抽取配置"""
    # 抽取方法
    use_rule_based: bool = True           # 使用规则抽取
    use_dependency_parsing: bool = True   # 使用依存句法
    use_llm: bool = True                  # 使用LLM抽取（默认启用！）

    # 方向性判断
    enable_direction_detection: bool = True  # 启用方向性判断
    min_direction_confidence: float = 0.6    # 方向性最低置信度

    # 置信度阈值
    min_confidence: float = 0.5           # 最低置信度
    llm_confidence_boost: float = 0.3     # LLM置信度加成

    # 关系类型限制
    enabled_relation_types: Set[GraphRelationType] = field(default_factory=set)

    # 窗口大小
    max_distance: int = 100               # 实体间最大距离（字符数）
    max_sentence_distance: int = 3        # 实体间最大句子距离

    # 批处理配置
    batch_size: int = 10
    max_relations_per_chunk: int = 100    # 每个chunk最多抽取关系数


@dataclass
class GraphStorageConfig:
    """图谱存储配置"""
    # Neo4j连接
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"

    # 数据库名称
    database: str = "neo4j"

    # 约束和索引
    create_constraints: bool = True
    create_indexes: bool = True

    # 批处理
    batch_size: int = 100
    transaction_timeout: int = 30  # 秒

    # 节点标签
    document_label: str = "Document"
    chunk_label: str = "Chunk"
    entity_label: str = "Entity"
    knowledge_node_label: str = "KnowledgeNode"


@dataclass
class GraphQualityConfig:
    """图谱质量配置"""
    # 实体质量
    min_entity_mentions: int = 1         # 实体最少出现次数
    min_entity_confidence: float = 0.5   # 实体最低置信度

    # 关系质量
    min_relation_mentions: int = 1       # 关系最少出现次数
    min_relation_confidence: float = 0.5 # 关系最低置信度

    # 孤立节点清理
    remove_isolated_nodes: bool = False  # 是否移除孤立节点
    min_connections: int = 1             # 最少连接数

    # 冲突检测
    detect_conflicting_relations: bool = True  # 检测冲突关系
    resolve_conflicts: str = "highest_confidence"  # 冲突解决策略

    # 数据验证
    validate_on_insert: bool = True
    validation_strict_mode: bool = False  # 严格模式：验证失败时拒绝插入


# 全局配置实例
graph_entity_config = EntityExtractionConfig()
graph_relation_config = RelationExtractionConfig()
graph_storage_config = GraphStorageConfig()
graph_quality_config = GraphQualityConfig()


def generate_entity_id(entity_name: str, entity_type: GraphEntityType) -> str:
    """
    生成统一的实体ID
    使用哈希确保跨文档一致性

    Args:
        entity_name: 实体名称
        entity_type: 实体类型

    Returns:
        实体ID (格式: {entity_type}:{hash})
    """
    import hashlib

    # 标准化实体名称（小写、去空格）
    normalized_name = entity_name.strip().lower()

    # 生成哈希
    hash_value = hashlib.md5(
        f"{entity_type.value}:{normalized_name}".encode()
    ).hexdigest()[:16]

    # 返回ID
    return f"{entity_type.value}:{hash_value}"


def generate_relation_id(
    source_id: str,
    target_id: str,
    relation_type: GraphRelationType,
    context: Optional[str] = None
) -> str:
    """
    生成统一的关系ID
    使用哈希确保唯一性

    Args:
        source_id: 源实体ID
        target_id: 目标实体ID
        relation_type: 关系类型
        context: 上下文（可选，用于区分同一对实体的多个关系）

    Returns:
        关系ID
    """
    import hashlib

    # 生成哈希
    if context:
        hash_input = f"{source_id}:{target_id}:{relation_type.value}:{context}"
    else:
        hash_input = f"{source_id}:{target_id}:{relation_type.value}"

    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:16]

    return f"REL:{hash_value}"


def normalize_entity_name(entity_name: str, entity_type: GraphEntityType) -> str:
    """
    标准化实体名称
    处理简称、全称、别名等

    Args:
        entity_name: 原始实体名称
        entity_type: 实体类型

    Returns:
        标准化后的实体名称
    """
    # 去除多余空格
    normalized = entity_name.strip()

    # 公司类型归一化
    if entity_type == GraphEntityType.COMPANY:
        # 移除常见后缀的冗余
        suffixes_to_remove = ["有限公司", "股份有限公司", "Co., Ltd.", "Inc.", "Corp."]
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
                break

        # 常见公司简称映射
        company_aliases = {
            "腾讯": "腾讯控股有限公司",
            "阿里巴巴": "阿里巴巴集团控股有限公司",
            "阿里": "阿里巴巴集团控股有限公司",
            "百度": "百度在线网络技术公司",
            "京东": "京东集团",
            "美团": "美团点评",
            "滴滴": "滴滴出行",
        }

        if normalized in company_aliases:
            normalized = company_aliases[normalized]

    return normalized


# 默认启用所有关系类型
graph_relation_config.enabled_relation_types = set(GraphRelationType)
