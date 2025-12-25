"""
社区检测器
使用多种算法检测知识图谱中的社区结构
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import json

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logging.warning("python-louvain未安装，Louvain算法不可用")

logger = logging.getLogger(__name__)


class DetectionAlgorithm(Enum):
    """社区检测算法"""
    LOUVAIN = "louvain"  # Louvain模块度优化
    LABEL_PROPAGATION = "label_propagation"  # 标签传播
    GREEDY_MODULARITY = "greedy_modularity"  # 贪婪模块度
    INFOMAP = "infomap"  # Infomap算法
    WALKTRAP = "walktrap"  # Walktrap算法
    SPECTRAL = "spectral"  # 谱聚类
    LEIDEN = "leiden"  # Leiden算法


@dataclass
class Community:
    """社区"""
    id: str
    nodes: List[str]
    size: int
    density: float
    modularity: Optional[float] = None
    internal_edges: int = 0
    external_edges: int = 0
    central_nodes: List[str] = None
    labels: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class DetectionResult:
    """检测结果"""
    communities: Dict[str, Community]
    algorithm: str
    total_communities: int
    modularity: float
    coverage: float
    performance: float
    execution_time: float
    statistics: Dict[str, Any]


class CommunityDetector:
    """社区检测器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 检测参数
        self.resolution = self.config.get('resolution', 1.0)  # 分辨率参数
        self.random_state = self.config.get('random_state', 42)
        self.min_community_size = self.config.get('min_community_size', 3)
        self.weight_attribute = self.config.get('weight_attribute', 'weight')

        # 算法权重
        self.algorithm_weights = {
            DetectionAlgorithm.LOUVAIN: 0.3,
            DetectionAlgorithm.LABEL_PROPAGATION: 0.2,
            DetectionAlgorithm.GREEDY_MODULARITY: 0.2,
            DetectionAlgorithm.SPECTRAL: 0.15,
            DetectionAlgorithm.WALKTRAP: 0.15
        }

    async def detect_communities(
        self,
        graph: nx.Graph,
        algorithm: str = "louvain",
        **kwargs
    ) -> DetectionResult:
        """检测社区"""
        try:
            start_time = asyncio.get_event_loop().time()

            logger.info(f"开始使用 {algorithm} 算法检测社区，图节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")

            # 验证图
            if graph.number_of_nodes() == 0:
                raise ValueError("图为空")

            # 选择算法
            detection_algorithm = DetectionAlgorithm(algorithm)

            # 执行社区检测
            if detection_algorithm == DetectionAlgorithm.LOUVAIN:
                communities = await self._detect_louvain(graph)
            elif detection_algorithm == DetectionAlgorithm.LABEL_PROPAGATION:
                communities = await self._detect_label_propagation(graph)
            elif detection_algorithm == DetectionAlgorithm.GREEDY_MODULARITY:
                communities = await self._detect_greedy_modularity(graph)
            elif detection_algorithm == DetectionAlgorithm.SPECTRAL:
                communities = await self._detect_spectral(graph)
            elif detection_algorithm == DetectionAlgorithm.WALKTRAP:
                communities = await self._detect_walktrap(graph)
            else:
                raise ValueError(f"不支持的算法: {algorithm}")

            # 计算社区指标
            communities_with_metrics = await self._calculate_community_metrics(graph, communities)

            # 计算全局指标
            modularity = await self._calculate_modularity(graph, communities_with_metrics)
            coverage = await self._calculate_coverage(graph, communities_with_metrics)
            performance = await self._calculate_performance(graph, communities_with_metrics)

            # 计算执行时间
            execution_time = asyncio.get_event_loop().time() - start_time

            # 生成统计信息
            statistics = await self._generate_detection_statistics(
                graph, communities_with_metrics, detection_algorithm
            )

            logger.info(f"社区检测完成，发现 {len(communities_with_metrics)} 个社区，模块度: {modularity:.3f}")

            return DetectionResult(
                communities=communities_with_metrics,
                algorithm=algorithm,
                total_communities=len(communities_with_metrics),
                modularity=modularity,
                coverage=coverage,
                performance=performance,
                execution_time=execution_time,
                statistics=statistics
            )

        except Exception as e:
            logger.error(f"社区检测失败: {str(e)}")
            raise

    async def _detect_louvain(self, graph: nx.Graph) -> Dict[str, Community]:
        """Louvain算法"""
        if not LOUVAIN_AVAILABLE:
            logger.warning("Louvain算法不可用，使用标签传播算法")
            return await self._detect_label_propagation(graph)

        try:
            # 使用python-louvain库
            partition = community_louvain.best_partition(
                graph,
                resolution=self.resolution,
                random_state=self.random_state,
                weight=self.weight_attribute
            )

            # 转换为社区格式
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)

            # 创建Community对象
            community_objects = {}
            for community_id, nodes in communities.items():
                if len(nodes) >= self.min_community_size:
                    community_obj = Community(
                        id=f"louvain_{community_id}",
                        nodes=nodes,
                        size=len(nodes),
                        density=0.0,  # 稍后计算
                        metadata={'algorithm': 'louvain', 'resolution': self.resolution}
                    )
                    community_objects[community_obj.id] = community_obj

            return community_objects

        except Exception as e:
            logger.error(f"Louvain算法执行失败: {str(e)}")
            # 回退到其他算法
            return await self._detect_label_propagation(graph)

    async def _detect_label_propagation(self, graph: nx.Graph) -> Dict[str, Community]:
        """标签传播算法"""
        try:
            # 使用NetworkX的标签传播
            communities = nx.algorithms.community.label_propagation_communities(
                graph,
                weight=self.weight_attribute
            )

            # 转换为字典格式
            community_objects = {}
            for i, community_nodes in enumerate(communities):
                if len(community_nodes) >= self.min_community_size:
                    community_obj = Community(
                        id=f"lp_{i}",
                        nodes=list(community_nodes),
                        size=len(community_nodes),
                        density=0.0,  # 稍后计算
                        metadata={'algorithm': 'label_propagation'}
                    )
                    community_objects[community_obj.id] = community_obj

            return community_objects

        except Exception as e:
            logger.error(f"标签传播算法执行失败: {str(e)}")
            # 回退到简单划分
            return await self._detect_simple_partition(graph)

    async def _detect_greedy_modularity(self, graph: nx.Graph) -> Dict[str, Community]:
        """贪婪模块度算法"""
        try:
            # 使用NetworkX的贪婪模块度优化
            communities = nx.algorithms.community.greedy_modularity_communities(
                graph,
                weight=self.weight_attribute
            )

            community_objects = {}
            for i, community_nodes in enumerate(communities):
                if len(community_nodes) >= self.min_community_size:
                    community_obj = Community(
                        id=f"greedy_{i}",
                        nodes=list(community_nodes),
                        size=len(community_nodes),
                        density=0.0,  # 稍后计算
                        metadata={'algorithm': 'greedy_modularity'}
                    )
                    community_objects[community_obj.id] = community_obj

            return community_objects

        except Exception as e:
            logger.error(f"贪婪模块度算法执行失败: {str(e)}")
            return await self._detect_simple_partition(graph)

    async def _detect_spectral(self, graph: nx.Graph) -> Dict[str, Community]:
        """谱聚类算法"""
        try:
            # 获取图的邻接矩阵
            adj_matrix = nx.adjacency_matrix(graph, weight=self.weight_attribute).toarray()

            # 计算拉普拉斯矩阵
            degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
            laplacian = degree_matrix - adj_matrix

            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

            # 选择前k个最小的特征值对应的特征向量
            n_communities = min(10, graph.number_of_nodes() // 10)  # 启发式选择社区数
            feature_vectors = eigenvectors[:, :n_communities]

            # K-means聚类
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_communities, random_state=self.random_state)
            labels = kmeans.fit_predict(feature_vectors)

            # 构建社区
            communities = defaultdict(list)
            for node, label in zip(list(graph.nodes()), labels):
                communities[label].append(node)

            community_objects = {}
            for community_id, nodes in communities.items():
                if len(nodes) >= self.min_community_size:
                    community_obj = Community(
                        id=f"spectral_{community_id}",
                        nodes=nodes,
                        size=len(nodes),
                        density=0.0,
                        metadata={'algorithm': 'spectral'}
                    )
                    community_objects[community_obj.id] = community_obj

            return community_objects

        except Exception as e:
            logger.error(f"谱聚类算法执行失败: {str(e)}")
            return await self._detect_simple_partition(graph)

    async def _detect_walktrap(self, graph: nx.Graph) -> Dict[str, Community]:
        """Walktrap算法（简化实现）"""
        try:
            # 这里实现一个简化版的walktrap算法
            # 实际应用中可以使用igraph库的完整实现

            # 基于随机游走的相似度矩阵
            n_nodes = graph.number_of_nodes()
            nodes = list(graph.nodes())
            node_index = {node: i for i, node in enumerate(nodes)}

            # 构建转移矩阵
            adj_matrix = nx.to_numpy_array(graph, weight=self.weight_attribute)
            degree_matrix = np.diag(adj_matrix.sum(axis=1))
            with np.errstate(divide='ignore', invalid='ignore'):
                transition_matrix = np.linalg.inv(degree_matrix) @ adj_matrix

            # 计算随机游走相似度
            similarity_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i, n_nodes):
                    # 计算从节点i和节点j开始的随机游走的相似度
                    similarity = 1.0 / (1 + np.linalg.norm(transition_matrix[i] - transition_matrix[j]))
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity

            # 使用层次聚类
            from sklearn.cluster import AgglomerativeClustering
            n_communities = min(20, n_nodes // 5)  # 启发式选择
            clustering = AgglomerativeClustering(
                n_clusters=n_communities,
                affinity='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(1 - similarity_matrix)  # 转换为距离

            # 构建社区
            communities = defaultdict(list)
            for node, label in zip(nodes, labels):
                communities[label].append(node)

            community_objects = {}
            for community_id, nodes in communities.items():
                if len(nodes) >= self.min_community_size:
                    community_obj = Community(
                        id=f"walktrap_{community_id}",
                        nodes=nodes,
                        size=len(nodes),
                        density=0.0,
                        metadata={'algorithm': 'walktrap'}
                    )
                    community_objects[community_obj.id] = community_obj

            return community_objects

        except Exception as e:
            logger.error(f"Walktrap算法执行失败: {str(e)}")
            return await self._detect_simple_partition(graph)

    async def _detect_simple_partition(self, graph: nx.Graph) -> Dict[str, Community]:
        """简单分区算法（回退方案）"""
        # 基于连通分量的简单划分
        connected_components = list(nx.connected_components(graph))

        community_objects = {}
        for i, component in enumerate(connected_components):
            if len(component) >= self.min_community_size:
                community_obj = Community(
                    id=f"component_{i}",
                    nodes=list(component),
                    size=len(component),
                    density=0.0,
                    metadata={'algorithm': 'connected_components'}
                )
                community_objects[community_obj.id] = community_obj

        return community_objects

    async def _calculate_community_metrics(
        self,
        graph: nx.Graph,
        communities: Dict[str, Community]
    ) -> Dict[str, Community]:
        """计算社区指标"""
        for community_id, community in communities.items():
            try:
                # 计算子图
                subgraph = graph.subgraph(community.nodes)

                # 计算密度
                if len(community.nodes) > 1:
                    community.density = nx.density(subgraph)
                else:
                    community.density = 0.0

                # 计算内部边数
                community.internal_edges = subgraph.number_of_edges()

                # 计算外部边数
                external_edges = 0
                for node in community.nodes:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in community.nodes:
                            external_edges += 1
                community.external_edges = external_edges // 2  # 每条边被计算了两次

                # 找出中心节点（度中心性最高的节点）
                if community.nodes:
                    degree_centrality = nx.degree_centrality(subgraph)
                    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                    community.central_nodes = [node for node, _ in sorted_nodes[:3]]  # 前3个中心节点

                # 提取社区标签
                community.labels = await self._extract_community_labels(graph, community)

            except Exception as e:
                logger.error(f"计算社区 {community_id} 指标失败: {str(e)}")

        return communities

    async def _extract_community_labels(self, graph: nx.Graph, community: Community) -> List[str]:
        """提取社区标签"""
        labels = []

        # 从节点属性中提取标签
        for node in community.nodes:
            if graph.nodes[node].get('type'):
                labels.append(graph.nodes[node]['type'])
            if graph.nodes[node].get('label'):
                labels.append(graph.nodes[node]['label'])

        # 从边属性中提取关系类型
        for node in community.nodes:
            for neighbor in graph.neighbors(node):
                if neighbor in community.nodes:
                    edge_data = graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('type'):
                        labels.append(edge_data['type'])

        # 去重并返回最常见的标签
        label_counts = Counter(labels)
        return [label for label, _ in label_counts.most_common(5)]

    async def _calculate_modularity(
        self,
        graph: nx.Graph,
        communities: Dict[str, Community]
    ) -> float:
        """计算模块度"""
        try:
            # 构建社区划分字典
            partition = {}
            for community_id, community in communities.items():
                for node in community.nodes:
                    partition[node] = community_id

            # 计算模块度
            modularity = nx.algorithms.community.modularity(
                graph,
                [set(c.nodes) for c in communities.values()],
                weight=self.weight_attribute
            )

            return modularity

        except Exception as e:
            logger.error(f"计算模块度失败: {str(e)}")
            return 0.0

    async def _calculate_coverage(
        self,
        graph: nx.Graph,
        communities: Dict[str, Community]
    ) -> float:
        """计算覆盖率"""
        try:
            # 计算社区内边数
            internal_edges = sum(community.internal_edges for community in communities.values())
            total_edges = graph.number_of_edges()

            coverage = internal_edges / total_edges if total_edges > 0 else 0.0
            return coverage

        except Exception as e:
            logger.error(f"计算覆盖率失败: {str(e)}")
            return 0.0

    async def _calculate_performance(
        self,
        graph: nx.Graph,
        communities: Dict[str, Community]
    ) -> float:
        """计算性能指标"""
        try:
            # 性能 = (社区内边数 + 社区间非边数) / 所有可能的边对数
            n = graph.number_of_nodes()
            total_pairs = n * (n - 1) / 2

            internal_edges = sum(community.internal_edges for community in communities.values())

            # 计算社区间可能的边数
            external_possible = 0
            for i, community1 in enumerate(communities.values()):
                for community2 in list(communities.values())[i+1:]:
                    external_possible += len(community1.nodes) * len(community2.nodes)

            # 社区间实际存在的边数
            external_edges = sum(community.external_edges for community in communities.values())

            # 社区间非边数
            external_non_edges = external_possible - external_edges

            performance = (internal_edges + external_non_edges) / total_pairs
            return performance

        except Exception as e:
            logger.error(f"计算性能指标失败: {str(e)}")
            return 0.0

    async def _generate_detection_statistics(
        self,
        graph: nx.Graph,
        communities: Dict[str, Community],
        algorithm: DetectionAlgorithm
    ) -> Dict[str, Any]:
        """生成检测统计信息"""
        stats = {
            'algorithm': algorithm.value,
            'graph_info': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_connected': nx.is_connected(graph)
            },
            'community_info': {
                'total_communities': len(communities),
                'avg_size': np.mean([c.size for c in communities.values()]) if communities else 0,
                'min_size': min([c.size for c in communities.values()]) if communities else 0,
                'max_size': max([c.size for c in communities.values()]) if communities else 0,
                'size_distribution': Counter([c.size for c in communities.values()])
            },
            'quality_metrics': {
                'avg_density': np.mean([c.density for c in communities.values()]) if communities else 0,
                'avg_internal_edges': np.mean([c.internal_edges for c in communities.values()]) if communities else 0,
                'avg_external_edges': np.mean([c.external_edges for c in communities.values()]) if communities else 0
            },
            'label_analysis': {}
        }

        # 标签分析
        all_labels = []
        for community in communities.values():
            if community.labels:
                all_labels.extend(community.labels)

        if all_labels:
            label_counter = Counter(all_labels)
            stats['label_analysis'] = {
                'total_unique_labels': len(label_counter),
                'most_common_labels': label_counter.most_common(10)
            }

        return stats

    async def compare_algorithms(
        self,
        graph: nx.Graph,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, DetectionResult]:
        """比较不同算法的结果"""
        if algorithms is None:
            algorithms = ["louvain", "label_propagation", "greedy_modularity"]

        results = {}

        for algorithm in algorithms:
            try:
                logger.info(f"运行 {algorithm} 算法")
                result = await self.detect_communities(graph, algorithm)
                results[algorithm] = result
            except Exception as e:
                logger.error(f"算法 {algorithm} 执行失败: {str(e)}")
                continue

        return results

    async def ensemble_detection(
        self,
        graph: nx.Graph,
        algorithms: Optional[List[str]] = None,
        voting_threshold: float = 0.5
    ) -> DetectionResult:
        """集成多种算法的检测结果"""
        if algorithms is None:
            algorithms = ["louvain", "label_propagation", "greedy_modularity"]

        # 运行所有算法
        algorithm_results = await self.compare_algorithms(graph, algorithms)

        if not algorithm_results:
            raise RuntimeError("所有算法都执行失败")

        # 构建节点-社区共现矩阵
        node_community_votes = defaultdict(lambda: defaultdict(int))

        for algorithm, result in algorithm_results.items():
            weight = self.algorithm_weights.get(DetectionAlgorithm(algorithm), 1.0)
            for community in result.communities.values():
                for node in community.nodes:
                    node_community_votes[node][community.id] += weight

        # 基于投票结果重新构建社区
        final_communities = {}
        assigned_nodes = set()

        # 处理高置信度的分配
        for node, votes in node_community_votes.items():
            if votes:
                best_community, max_votes = max(votes.items(), key=lambda x: x[1])
                total_votes = sum(votes.values())

                if max_votes / total_votes >= voting_threshold:
                    if best_community not in final_communities:
                        final_communities[best_community] = []
                    final_communities[best_community].append(node)
                    assigned_nodes.add(node)

        # 处理未分配的节点
        unassigned_nodes = set(graph.nodes()) - assigned_nodes
        if unassigned_nodes:
            # 为未分配的节点创建小社区或分配到最近的社区
            for node in unassigned_nodes:
                # 找到最近的社区（基于连接）
                best_community = None
                max_connections = 0

                for community_id, community_nodes in final_communities.items():
                    connections = sum(1 for neighbor in graph.neighbors(node) if neighbor in community_nodes)
                    if connections > max_connections:
                        max_connections = connections
                        best_community = community_id

                if best_community:
                    final_communities[best_community].append(node)
                else:
                    # 创建新社区
                    new_community_id = f"singleton_{node}"
                    final_communities[new_community_id] = [node]

        # 转换为Community对象
        community_objects = {}
        for community_id, nodes in final_communities.items():
            if len(nodes) >= self.min_community_size:
                community_obj = Community(
                    id=community_id,
                    nodes=nodes,
                    size=len(nodes),
                    density=0.0,
                    metadata={'algorithm': 'ensemble', 'source_algorithms': algorithms}
                )
                community_objects[community_id] = community_obj

        # 计算指标
        community_objects = await self._calculate_community_metrics(graph, community_objects)
        modularity = await self._calculate_modularity(graph, community_objects)
        coverage = await self._calculate_coverage(graph, community_objects)
        performance = await self._calculate_performance(graph, community_objects)

        # 生成统计信息
        statistics = {
            'ensemble_info': {
                'algorithms_used': algorithms,
                'voting_threshold': voting_threshold,
                'algorithm_weights': {alg: self.algorithm_weights.get(DetectionAlgorithm(alg), 1.0) for alg in algorithms}
            },
            'individual_results': {alg: {
                'communities': result.total_communities,
                'modularity': result.modularity
            } for alg, result in algorithm_results.items()}
        }

        return DetectionResult(
            communities=community_objects,
            algorithm="ensemble",
            total_communities=len(community_objects),
            modularity=modularity,
            coverage=coverage,
            performance=performance,
            execution_time=sum(r.execution_time for r in algorithm_results.values()),
            statistics=statistics
        )