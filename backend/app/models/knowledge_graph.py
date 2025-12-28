"""
知识图谱数据模型
"""

from sqlalchemy import Column, Integer, String, Text, BigInteger, DateTime, ForeignKey, Enum, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class NodeType(str, enum.Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    RELATION = "relation"
    EVENT = "event"
    ORGANIZATION = "organization"
    PERSON = "person"
    LOCATION = "location"
    DATE = "date"
    AMOUNT = "amount"


class RelationType(str, enum.Enum):
    OWNS = "owns"
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
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
    __tablename__ = "knowledge_graph_relations"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    # 关系标识
    relation_id = Column(String(255), nullable=False, unique=True, index=True)  # Neo4j中的关系ID
    neo4j_id = Column(String(255), index=True)  # Neo4j内部ID

    # 关系两端
    source_node_id = Column(String(255), nullable=False, index=True)  # 源节点ID
    target_node_id = Column(String(255), nullable=False, index=True)  # 目标节点ID
    source_node_neo4j_id = Column(String(255), index=True)
    target_node_neo4j_id = Column(String(255), index=True)

    # 关系信息
    relation_type = Column(Enum(RelationType), nullable=False, index=True)
    relation_label = Column(String(500))  # 关系标签
    description = Column(Text)  # 关系描述

    # 关系属性
    properties = Column(JSON)  # 关系属性
    attributes = Column(JSON)  # 详细属性
    weight = Column(Float, default=1.0)  # 关系权重
    confidence = Column(Float)  # 关系提取置信度
    direction = Column(String(20), default="directed")  # directed, undirected

    # 证据信息
    evidence = Column(Text)  # 支持证据
    source_text = Column(Text)  # 源文本
    page_number = Column(Integer)
    position = Column(JSON)  # 在文档中的位置
    context = Column(Text)  # 上下文

    # 验证信息
    is_verified = Column(Integer, default=0)  # 是否已验证 (0/1)
    verification_method = Column(String(100))  # 验证方法
    verification_confidence = Column(Float)  # 验证置信度

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document", backref="kg_relations")

    def __repr__(self):
        return f"<KnowledgeGraphRelation(id={self.id}, relation_id='{self.relation_id}', type='{self.relation_type}')>"


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