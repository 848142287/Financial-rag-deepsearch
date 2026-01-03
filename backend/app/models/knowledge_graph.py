"""
知识图谱数据模型
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class NodeType(str, enum.Enum):
    # 基础实体类型
    ENTITY = "entity"
    CONCEPT = "concept"
    EVENT = "event"
    ORGANIZATION = "organization"
    PERSON = "person"
    LOCATION = "location"
    DATE = "date"
    AMOUNT = "amount"

    # 金融研报专用类型
    STRATEGY = "strategy"  # 策略类型（择时、选股、均线、动量等）
    INDICATOR = "indicator"  # 金融指标（PE、PB、ROE、夏普比率、最大回撤等）
    MARKET_CONCEPT = "market_concept"  # 市场概念（行业、牛市、轮动、流动性等）
    TIME_PERIOD = "time_period"  # 时间周期（月、周、短期、长期等）
    QUANT_METHOD = "quant_method"  # 量化方法（趋势、震荡、移动平均、回归分析等）
    METRIC = "metric"  # 数值指标（具体的数值，如"胜率76.9%"）


class RelationType(str, enum.Enum):
    # 通用关系
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    BELONGS_TO = "belongs_to"
    CONTAINS = "contains"
    LOCATED_IN = "located_in"

    # 金融研报专用关系（基于历史数据分布）
    PREDICTS = "predicts"  # 预测关系 (30.89%)
    POSITIVELY_CORRELATED = "positively_correlated"  # 正相关关系 (30.87%)
    INFLUENCES = "influences"  # 影响关系 (22.48%)
    INCLUDES = "includes"  # 包含关系 (6.03%)
    OUTPERFORMS = "outperforms"  # 优于关系 (5.87%)

    # 其他保留关系
    OWNS = "owns"
    WORKS_FOR = "works_for"
    INVESTS_IN = "invests_in"
    ACQUIRES = "acquires"
    MERGES_WITH = "merges_with"
    COLLABORATES_WITH = "collaborates_with"
    REPORTS_TO = "reports_to"
    REGULATED_BY = "regulated_by"


class KnowledgeGraphNode(Base):
    """知识图谱节点表"""
    __tablename__ = "knowledge_graph_nodes"

    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(Integer, nullable=False, index=True)  # 图谱ID
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    # 节点标识
    node_id = Column(String(255), nullable=False, unique=True, index=True)  # Neo4j中的节点ID
    neo4j_id = Column(String(255), index=True)  # Neo4j内部ID

    # 节点基本信息
    node_type = Column(Enum(NodeType), nullable=False, index=True)
    node_name = Column(String(1000), nullable=False, index=True)
    node_label = Column(String(500))  # 节点标签
    node_alias = Column(JSON)  # 别名列表

    # 节点属性
    properties = Column(JSON)  # 节点属性
    attributes = Column(JSON)  # 详细属性
    confidence = Column(Float)  # 节点提取置信度
    importance = Column(Float)  # 重要性评分

    # 位置信息
    source_text = Column(Text)  # 源文本
    page_number = Column(Integer)
    position = Column(JSON)  # 在文档中的位置
    context = Column(Text)  # 上下文

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document", backref="kg_nodes")

    def __repr__(self):
        return f"<KnowledgeGraphNode(id={self.id}, node_id='{self.node_id}', type='{self.node_type}', name='{self.node_name}')>"


class KnowledgeGraphRelation(Base):
    """知识图谱关系表"""
    __tablename__ = "knowledge_graph_edges"

    id = Column(Integer, primary_key=True, index=True)
    graph_id = Column(Integer, nullable=False, index=True)  # 图谱ID

    # 关系两端（简化版）
    source_id = Column(String(255), nullable=False, index=True)  # 源节点ID
    target_id = Column(String(255), nullable=False, index=True)  # 目标节点ID

    # 关系类型（存储为字符串，与RelationType枚举值对应）
    edge_type = Column(String(100), index=True)  # 关系类型

    # 关系属性（JSON格式存储所有额外信息）
    properties = Column(JSON)  # 包含document_id, confidence, weight等

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<KnowledgeGraphRelation(id={self.id}, source_id='{self.source_id}', target_id='{self.target_id}', type='{self.edge_type}')>"


class KnowledgeGraphPath(Base):
    """知识图谱路径表"""
    __tablename__ = "knowledge_graph_paths"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    # 路径信息
    path_id = Column(String(255), nullable=False, unique=True, index=True)
    source_node_id = Column(String(255), nullable=False, index=True)
    target_node_id = Column(String(255), nullable=False, index=True)

    # 路径详情
    path_length = Column(Integer, nullable=False)  # 路径长度
    node_sequence = Column(JSON)  # 节点序列
    relation_sequence = Column(JSON)  # 关系序列

    # 路径属性
    path_type = Column(String(100))  # 路径类型
    confidence = Column(Float)  # 路径置信度
    weight = Column(Float)  # 路径权重
    importance = Column(Float)  # 重要性评分

    # 路径描述
    description = Column(Text)  # 路径描述
    summary = Column(Text)  # 路径摘要

    # 元数据
    graph_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    document = relationship("Document", backref="kg_paths")

    def __repr__(self):
        return f"<KnowledgeGraphPath(id={self.id}, path_id='{self.path_id}', length={self.path_length})>"


class KnowledgeGraphEntity(Base):
    """实体表（节点的详细实体信息）"""
    __tablename__ = "knowledge_graph_entities"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(255), ForeignKey("knowledge_graph_nodes.node_id"), nullable=False, unique=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    # 实体基本信息
    entity_name = Column(String(1000), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False, index=True)
    entity_subtype = Column(String(100))  # 实体子类型
    entity_category = Column(String(100))  # 实体类别

    # 实体属性
    canonical_name = Column(String(1000))  # 规范名称
    synonyms = Column(JSON)  # 同义词列表
    abbreviations = Column(JSON)  # 缩写列表
    aliases = Column(JSON)  # 别名列表

    # 描述信息
    description = Column(Text)  # 实体描述
    definition = Column(Text)  # 实体定义
    characteristics = Column(JSON)  # 特征列表

    # 外部链接
    external_ids = Column(JSON)  # 外部ID映射 (WikiData, DBpedia等)
    external_links = Column(JSON)  # 外部链接
    references = Column(JSON)  # 参考链接

    # 统计信息
    mention_count = Column(Integer, default=0)  # 提及次数
    relation_count = Column(Integer, default=0)  # 关系数目
    confidence = Column(Float)  # 总体置信度

    # 元数据
    graph_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    node = relationship("KnowledgeGraphNode", backref="entity_details")
    document = relationship("Document", backref="entities")

    def __repr__(self):
        return f"<KnowledgeGraphEntity(id={self.id}, entity_name='{self.entity_name}', type='{self.entity_type}')>"


class KnowledgeGraphCluster(Base):
    """知识图谱聚类表"""
    __tablename__ = "knowledge_graph_clusters"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    # 聚类信息
    cluster_id = Column(String(255), nullable=False, unique=True, index=True)
    cluster_name = Column(String(1000))
    cluster_type = Column(String(100))  # 聚类类型

    # 聚类属性
    center_node_id = Column(String(255), ForeignKey("knowledge_graph_nodes.node_id"))  # 中心节点
    member_nodes = Column(JSON)  # 成员节点列表
    member_count = Column(Integer, default=0)  # 成员数量

    # 聚类特征
    keywords = Column(JSON)  # 关键词
    topics = Column(JSON)  # 主题
    summary = Column(Text)  # 聚类摘要
    description = Column(Text)  # 聚类描述

    # 聚类质量
    cohesion_score = Column(Float)  # 内聚性评分
    separation_score = Column(Float)  # 分离性评分
    quality_score = Column(Float)  # 质量评分

    # 元数据
    graph_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document", backref="kg_clusters")
    center_node = relationship("KnowledgeGraphNode")

    def __repr__(self):
        return f"<KnowledgeGraphCluster(id={self.id}, cluster_id='{self.cluster_id}', members={self.member_count})>"