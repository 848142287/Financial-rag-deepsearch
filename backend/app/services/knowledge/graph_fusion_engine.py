"""
知识图谱融合引擎
处理实体对齐、关系融合和社区检测
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import networkx as nx
from collections import defaultdict

from .entity_aligner import EntityAligner
from .relation_merger import RelationMerger
from .community_detector import CommunityDetector

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """融合策略"""
    STRICT = "strict"  # 严格模式，高相似度阈值
    MODERATE = "moderate"  # 中等模式
    LOOSE = "loose"  # 宽松模式，低相似度阈值


@dataclass
class EntityNode:
    """实体节点"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    source_documents: Set[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass
class RelationEdge:
    """关系边"""
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]
    weight: float = 1.0
    confidence: float = 1.0
    source_documents: Set[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class GraphFusionResult:
    """图谱融合结果"""
    entities: List[EntityNode]
    relations: List[RelationEdge]
    communities: Dict[str, List[str]]
    statistics: Dict[str, Any]
    alignment_results: Dict[str, Any]
    metadata: Dict[str, Any]


class GraphFusionEngine:
    """知识图谱融合引擎"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_aligner = EntityAligner(config or {})
        self.relation_merger = RelationMerger(config or {})
        self.community_detector = CommunityDetector(config or {})

        # 融合参数
        self.similarity_thresholds = {
            FusionStrategy.STRICT: 0.9,
            FusionStrategy.MODERATE: 0.8,
            FusionStrategy.LOOSE: 0.7
        }

        # 图数据结构
        self.graph = nx.DiGraph()

    async def fuse_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        strategy: FusionStrategy = FusionStrategy.MODERATE,
        context: Optional[Dict[str, Any]] = None
    ) -> GraphFusionResult:
        """
        融合知识图谱
        """
        try:
            logger.info(f"开始融合知识图谱，实体数: {len(entities)}, 关系数: {len(relations)}")

            # 1. 创建实体节点
            entity_nodes = self._create_entity_nodes(entities)

            # 2. 创建关系边
            relation_edges = self._create_relation_edges(relations, entity_nodes)

            # 3. 构建图
            self._build_graph(entity_nodes, relation_edges)

            # 4. 实体对齐
            alignment_results = await self._align_entities(entity_nodes, strategy)

            # 5. 关系融合
            merged_relations = await self._merge_relations(relation_edges, alignment_results)

            # 6. 社区检测
            communities = await self._detect_communities(alignment_results["merged_entities"], merged_relations)

            # 7. 计算统计信息
            statistics = self._calculate_graph_statistics(alignment_results["merged_entities"], merged_relations, communities)

            # 8. 生成元数据
            metadata = self._generate_fusion_metadata(entities, relations, strategy, context)

            logger.info(f"知识图谱融合完成，最终实体数: {len(alignment_results['merged_entities'])}, 关系数: {len(merged_relations)}")

            return GraphFusionResult(
                entities=alignment_results["merged_entities"],
                relations=merged_relations,
                communities=communities,
                statistics=statistics,
                alignment_results=alignment_results,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"知识图谱融合失败: {str(e)}")
            raise

    def _create_entity_nodes(self, entities: List[Dict[str, Any]]) -> List[EntityNode]:
        """创建实体节点"""
        entity_nodes = []

        for entity_data in entities:
            node = EntityNode(
                id=entity_data.get('id', f"entity_{len(entity_nodes)}"),
                label=entity_data.get('text', entity_data.get('label', '')),
                type=entity_data.get('type', 'unknown'),
                properties=entity_data.get('properties', {}),
                embedding=entity_data.get('embedding'),
                source_documents=set(entity_data.get('source_documents', [])),
                confidence=entity_data.get('confidence', 1.0),
                metadata=entity_data.get('metadata', {})
            )
            entity_nodes.append(node)

        return entity_nodes

    def _create_relation_edges(
        self,
        relations: List[Dict[str, Any]],
        entity_nodes: List[EntityNode]
    ) -> List[RelationEdge]:
        """创建关系边"""
        # 创建实体ID映射
        entity_id_map = {node.id: node for node in entity_nodes}

        relation_edges = []
        for rel_data in relations:
            # 查找源和目标实体
            source_id = rel_data.get('source') or rel_data.get('subject')
            target_id = rel_data.get('target') or rel_data.get('object')

            if source_id in entity_id_map and target_id in entity_id_map:
                edge = RelationEdge(
                    id=rel_data.get('id', f"relation_{len(relation_edges)}"),
                    source=source_id,
                    target=target_id,
                    type=rel_data.get('predicate', rel_data.get('type', 'unknown')),
                    properties=rel_data.get('properties', {}),
                    weight=rel_data.get('weight', 1.0),
                    confidence=rel_data.get('confidence', 1.0),
                    source_documents=set(rel_data.get('source_documents', [])),
                    metadata=rel_data.get('metadata', {})
                )
                relation_edges.append(edge)

        return relation_edges

    def _build_graph(self, entity_nodes: List[EntityNode], relation_edges: List[RelationEdge]):
        """构建图结构"""
        self.graph.clear()

        # 添加节点
        for node in entity_nodes:
            self.graph.add_node(
                node.id,
                label=node.label,
                type=node.type,
                properties=node.properties,
                confidence=node.confidence
            )

        # 添加边
        for edge in relation_edges:
            self.graph.add_edge(
                edge.source,
                edge.target,
                type=edge.type,
                weight=edge.weight,
                confidence=edge.confidence
            )

    async def _align_entities(
        self,
        entity_nodes: List[EntityNode],
        strategy: FusionStrategy
    ) -> Dict[str, Any]:
        """对齐实体"""
        logger.info(f"开始实体对齐，策略: {strategy.value}")

        # 使用实体对齐器
        alignment_result = await self.entity_aligner.align_entities(
            entity_nodes,
            threshold=self.similarity_thresholds[strategy]
        )

        # 更新实体信息
        merged_entities = []
        entity_id_map = {}

        for cluster_id, cluster in alignment_result["clusters"].items():
            if len(cluster) == 1:
                # 没有重复的实体
                entity = cluster[0]
                entity.id = f"entity_{len(merged_entities)}"
            else:
                # 合并重复实体
                merged_entity = self._merge_entity_cluster(cluster)
                entity = merged_entity

            merged_entities.append(entity)
            entity_id_map[entity.id] = entity

        logger.info(f"实体对齐完成，原始实体: {len(entity_nodes)}, 合并后: {len(merged_entities)}")

        return {
            "original_entities": entity_nodes,
            "merged_entities": merged_entities,
            "alignment_clusters": alignment_result["clusters"],
            "entity_id_map": entity_id_map
        }

    def _merge_entity_cluster(self, cluster: List[EntityNode]) -> EntityNode:
        """合并实体簇"""
        if len(cluster) == 1:
            return cluster[0]

        # 选择置信度最高的实体作为主实体
        main_entity = max(cluster, key=lambda e: e.confidence)

        # 合并属性
        merged_properties = defaultdict(list)
        all_types = set()
        all_source_docs = set()
        total_confidence = 0

        for entity in cluster:
            # 合并属性
            for key, value in entity.properties.items():
                if value not in merged_properties[key]:
                    merged_properties[key].append(value)

            # 合并类型
            all_types.add(entity.type)

            # 合并源文档
            all_source_docs.update(entity.source_documents or [])

            # 累计置信度
            total_confidence += entity.confidence

        # 解决属性冲突（选择最频繁的值）
        final_properties = {}
        for key, values in merged_properties.items():
            if len(values) == 1:
                final_properties[key] = values[0]
            else:
                # 选择出现次数最多的值
                from collections import Counter
                final_properties[key] = Counter(values).most_common(1)[0][0]

        # 合并嵌入向量（平均）
        merged_embedding = None
        embeddings = [e.embedding for e in cluster if e.embedding is not None]
        if embeddings:
            merged_embedding = np.mean(embeddings, axis=0)

        # 创建合并后的实体
        merged_entity = EntityNode(
            id=f"merged_{main_entity.id}",
            label=main_entity.label,
            type=main_entity.type if len(all_types) == 1 else "merged",
            properties=final_properties,
            embedding=merged_embedding,
            source_documents=all_source_docs,
            confidence=min(total_confidence / len(cluster), 1.0),
            metadata={
                "merged_from": [e.id for e in cluster],
                "merge_count": len(cluster),
                "original_types": list(all_types)
            }
        )

        return merged_entity

    async def _merge_relations(
        self,
        relation_edges: List[RelationEdge],
        alignment_results: Dict[str, Any]
    ) -> List[RelationEdge]:
        """融合关系"""
        logger.info("开始关系融合")

        # 使用关系融合器
        merge_result = await self.relation_merger.merge_relations(
            relation_edges,
            alignment_results["entity_id_map"]
        )

        logger.info(f"关系融合完成，原始关系: {len(relation_edges)}, 合并后: {len(merge_result['merged_relations'])}")

        return merge_result["merged_relations"]

    async def _detect_communities(
        self,
        entities: List[EntityNode],
        relations: List[RelationEdge]
    ) -> Dict[str, List[str]]:
        """检测社区"""
        logger.info("开始社区检测")

        # 构建社区检测图
        community_graph = nx.Graph()

        # 添加节点
        for entity in entities:
            community_graph.add_node(entity.id, type=entity.type)

        # 添加边（无向）
        for relation in relations:
            community_graph.add_edge(
                relation.source,
                relation.target,
                weight=relation.weight
            )

        # 使用社区检测器
        communities = await self.community_detector.detect_communities(community_graph)

        logger.info(f"社区检测完成，发现 {len(communities)} 个社区")

        return communities

    def _calculate_graph_statistics(
        self,
        entities: List[EntityNode],
        relations: List[RelationEdge],
        communities: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """计算图统计信息"""
        stats = {
            "node_count": len(entities),
            "edge_count": len(relations),
            "community_count": len(communities),
            "entity_types": {},
            "relation_types": {},
            "graph_density": 0.0,
            "avg_clustering": 0.0,
            "max_path_length": 0,
            "entity_confidence_stats": {},
            "relation_confidence_stats": {}
        }

        # 统计实体类型
        for entity in entities:
            entity_type = entity.type
            stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1

        # 统计关系类型
        for relation in relations:
            rel_type = relation.type
            stats["relation_types"][rel_type] = stats["relation_types"].get(rel_type, 0) + 1

        # 计算图密度
        if len(entities) > 1:
            max_edges = len(entities) * (len(entities) - 1)
            stats["graph_density"] = len(relations) / max_edges

        # 计算平均聚集系数
        if len(self.graph.nodes) > 0:
            stats["avg_clustering"] = nx.average_clustering(self.graph.to_undirected())

        # 计算最大路径长度（连通分量）
        if nx.is_connected(self.graph.to_undirected()):
            stats["max_path_length"] = nx.diameter(self.graph.to_undirected())

        # 计算置信度统计
        entity_confidences = [e.confidence for e in entities]
        if entity_confidences:
            stats["entity_confidence_stats"] = {
                "mean": np.mean(entity_confidences),
                "min": np.min(entity_confidences),
                "max": np.max(entity_confidences),
                "std": np.std(entity_confidences)
            }

        relation_confidences = [r.confidence for r in relations]
        if relation_confidences:
            stats["relation_confidence_stats"] = {
                "mean": np.mean(relation_confidences),
                "min": np.min(relation_confidences),
                "max": np.max(relation_confidences),
                "std": np.std(relation_confidences)
            }

        return stats

    def _generate_fusion_metadata(
        self,
        original_entities: List[Dict[str, Any]],
        original_relations: List[Dict[str, Any]],
        strategy: FusionStrategy,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成融合元数据"""
        metadata = {
            "fusion_timestamp": datetime.now().isoformat(),
            "fusion_strategy": strategy.value,
            "similarity_threshold": self.similarity_thresholds[strategy],
            "original_counts": {
                "entities": len(original_entities),
                "relations": len(original_relations)
            },
            "context": context or {},
            "config": self.config
        }

        return metadata

    async def update_graph(
        self,
        existing_graph: GraphFusionResult,
        new_entities: List[Dict[str, Any]],
        new_relations: List[Dict[str, Any]],
        strategy: FusionStrategy = FusionStrategy.MODERATE
    ) -> GraphFusionResult:
        """更新现有图谱"""
        logger.info("开始增量更新知识图谱")

        # 转换现有图谱为节点和边的格式
        existing_entities = [
            {
                "id": e.id,
                "text": e.label,
                "type": e.type,
                "properties": e.properties,
                "embedding": e.embedding,
                "confidence": e.confidence
            }
            for e in existing_graph.entities
        ]

        existing_relations = [
            {
                "id": r.id,
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "weight": r.weight,
                "confidence": r.confidence
            }
            for r in existing_graph.relations
        ]

        # 合并新旧实体和关系
        all_entities = existing_entities + new_entities
        all_relations = existing_relations + new_relations

        # 重新融合
        updated_result = await self.fuse_knowledge_graph(
            all_entities,
            all_relations,
            strategy
        )

        # 标记为增量更新
        updated_result.metadata["update_type"] = "incremental"
        updated_result.metadata["previous_update"] = existing_graph.metadata.get("fusion_timestamp")

        logger.info("增量更新完成")

        return updated_result

    def export_graph(self, format: str = "json") -> Dict[str, Any]:
        """导出图谱"""
        if format == "json":
            return {
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "type": node.type,
                        "properties": node.properties,
                        "confidence": node.confidence
                    }
                    for node in self.graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "source": edge[0],
                        "target": edge[1],
                        "type": edge[2]["type"],
                        "weight": edge[2]["weight"]
                    }
                    for edge in self.graph.edges(data=True)
                ]
            }
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def get_graph_summary(self) -> Dict[str, Any]:
        """获取图谱摘要"""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "is_connected": nx.is_connected(self.graph.to_undirected()),
            "density": nx.density(self.graph),
            "entity_types": self._get_entity_type_distribution()
        }

    def _get_entity_type_distribution(self) -> Dict[str, int]:
        """获取实体类型分布"""
        type_count = defaultdict(int)
        for node in self.graph.nodes(data=True):
            type_count[node[1].get("type", "unknown")] += 1
        return dict(type_count)