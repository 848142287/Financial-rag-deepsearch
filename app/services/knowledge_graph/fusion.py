"""
知识图谱融合模块
提供统一的图谱融合接口
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from datetime import datetime

from .graph_fusion_engine import GraphFusionEngine, EntityNode, RelationEdge, FusionStrategy
from .entity_aligner import EntityAligner
from .relation_merger import RelationMerger
from .community_detector import CommunityDetector

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """融合配置"""
    strategy: FusionStrategy = FusionStrategy.MODERATE
    enable_entity_alignment: bool = True
    enable_relation_merging: bool = True
    enable_community_detection: bool = True
    similarity_threshold: float = 0.8
    min_cluster_size: int = 3


class KnowledgeGraphFusioner:
    """知识图谱融合器

    提供统一的图谱融合接口，整合实体对齐、关系融合和社区检测功能
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """初始化融合器

        Args:
            config: 融合配置
        """
        self.config = config or FusionConfig()

        # 初始化各个组件
        self.fusion_engine = GraphFusionEngine(
            strategy=self.config.strategy,
            similarity_threshold=self.config.similarity_threshold
        )

        self.entity_aligner = EntityAligner()
        self.relation_merger = RelationMerger()
        self.community_detector = CommunityDetector()

    async def fuse_knowledge_graphs(self,
                                  graphs: List[nx.Graph],
                                  source_documents: Optional[List[str]] = None) -> Tuple[nx.Graph, Dict[str, Any]]:
        """融合多个知识图谱

        Args:
            graphs: 要融合的图谱列表
            source_documents: 源文档列表

        Returns:
            融合后的图谱和融合统计信息
        """
        if not graphs:
            return nx.Graph(), {"message": "没有输入图谱"}

        try:
            logger.info(f"开始融合 {len(graphs)} 个知识图谱")

            # 1. 实体对齐
            aligned_entities = []
            if self.config.enable_entity_alignment:
                for i, graph1 in enumerate(graphs):
                    for j, graph2 in enumerate(graphs[i+1:], i+1):
                        alignment_result = await self.entity_aligner.align_entities(
                            list(graph1.nodes(data=True)),
                            list(graph2.nodes(data=True))
                        )
                        aligned_entities.extend(alignment_result.alignments)

            # 2. 使用融合引擎进行图谱融合
            fusion_result = await self.fusion_engine.fuse_graphs(
                graphs,
                aligned_entities,
                source_documents
            )

            fused_graph = fusion_result.fused_graph

            # 3. 关系融合
            if self.config.enable_relation_merging:
                merge_result = await self.relation_merger.merge_relations(
                    fused_graph,
                    strategy="similarity_based"
                )
                # 应用关系合并结果到融合图谱
                self._apply_relation_merges(fused_graph, merge_result)

            # 4. 社区检测
            communities = []
            if self.config.enable_community_detection:
                detection_result = await self.community_detector.detect_communities(
                    fused_graph,
                    algorithm="louvain",
                    min_cluster_size=self.config.min_cluster_size
                )
                communities = detection_result.communities

                # 将社区信息添加到图谱
                for i, community in enumerate(communities):
                    for node_id in community.nodes:
                        if fused_graph.has_node(node_id):
                            fused_graph.nodes[node_id]['community'] = i

            # 5. 计算融合统计
            fusion_stats = self._calculate_fusion_stats(
                graphs, fused_graph, aligned_entities, communities
            )

            logger.info(f"知识图谱融合完成: {fusion_stats}")
            return fused_graph, fusion_stats

        except Exception as e:
            logger.error(f"知识图谱融合失败: {e}")
            return nx.Graph(), {"error": str(e)}

    def _apply_relation_merges(self, graph: nx.Graph, merge_result: Any):
        """应用关系合并结果到图谱"""
        # 这里需要根据实际的关系合并结果结构来实现
        # 简化实现
        pass

    def _calculate_fusion_stats(self,
                               original_graphs: List[nx.Graph],
                               fused_graph: nx.Graph,
                               aligned_entities: List[Any],
                               communities: List[Any]) -> Dict[str, Any]:
        """计算融合统计信息"""
        original_nodes = sum(len(g.nodes()) for g in original_graphs)
        original_edges = sum(len(g.edges()) for g in original_graphs)

        return {
            "original_graphs_count": len(original_graphs),
            "original_nodes": original_nodes,
            "original_edges": original_edges,
            "fused_nodes": len(fused_graph.nodes()),
            "fused_edges": len(fused_graph.edges()),
            "node_reduction_rate": (original_nodes - len(fused_graph.nodes())) / max(original_nodes, 1),
            "aligned_entities_count": len(aligned_entities),
            "communities_detected": len(communities),
            "fusion_timestamp": datetime.now().isoformat(),
            "config": {
                "strategy": self.config.strategy.value,
                "similarity_threshold": self.config.similarity_threshold
            }
        }

    async def align_entities_only(self,
                                entities1: List[EntityNode],
                                entities2: List[EntityNode]) -> List[Any]:
        """仅进行实体对齐"""
        alignment_result = await self.entity_aligner.align_entities(entities1, entities2)
        return alignment_result.alignments

    async def detect_communities_only(self,
                                    graph: nx.Graph,
                                    algorithm: str = "louvain") -> List[Any]:
        """仅进行社区检测"""
        detection_result = await self.community_detector.detect_communities(
            graph, algorithm=algorithm
        )
        return detection_result.communities

    def get_fusion_summary(self, fused_graph: nx.Graph) -> Dict[str, Any]:
        """获取融合图谱摘要"""
        if not fused_graph.nodes():
            return {"message": "空图谱"}

        # 基本统计
        node_types = {}
        edge_types = {}
        communities = set()

        for node, data in fused_graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            if 'community' in data:
                communities.add(data['community'])

        for u, v, data in fused_graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        return {
            "nodes_count": len(fused_graph.nodes()),
            "edges_count": len(fused_graph.edges()),
            "node_types": node_types,
            "edge_types": edge_types,
            "communities_count": len(communities),
            "density": nx.density(fused_graph),
            "is_connected": nx.is_connected(fused_graph),
            "components_count": nx.number_connected_components(fused_graph)
        }