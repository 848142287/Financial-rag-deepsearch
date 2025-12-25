"""
任务规划模块
根据查询理解结果生成执行计划
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .query_understanding import QueryUnderstanding, QueryIntent
from .agent_engine import QueryPlan, QueryType

logger = logging.getLogger(__name__)


@dataclass
class RetrievalStrategy:
    """检索策略"""
    name: str
    weight: float
    parameters: Dict[str, Any]
    priority: int


@dataclass
class SubTask:
    """子任务"""
    task_id: str
    description: str
    retrieval_strategies: List[RetrievalStrategy]
    dependencies: List[str]
    expected_output: str
    priority: int


class TaskPlanner:
    """任务规划器"""

    def __init__(self):
        # 策略配置
        self.strategy_configs = {
            'vector': {
                'default_weight': 0.5,
                'parameters': {'limit': 10, 'threshold': 0.7}
            },
            'graph': {
                'default_weight': 0.3,
                'parameters': {'limit': 5, 'depth': 2}
            },
            'keyword': {
                'default_weight': 0.2,
                'parameters': {'limit': 15, 'fuzzy': True}
            }
        }

        # 意图到策略的映射
        self.intent_strategy_mapping = {
            QueryType.FACTUAL: ['vector', 'keyword'],
            QueryType.ANALYTICAL: ['vector', 'graph', 'keyword'],
            QueryType.COMPARATIVE: ['graph', 'vector'],
            QueryType.TEMPORAL: ['vector', 'keyword'],
            QueryType.CAUSAL: ['graph', 'vector'],
            QueryType.AGGREGATE: ['vector', 'keyword']
        }

    async def create_plan(
        self,
        query_id: str,
        original_query: str,
        query_understanding: QueryUnderstanding
    ) -> QueryPlan:
        """
        创建执行计划

        Args:
            query_id: 查询ID
            original_query: 原始查询
            query_understanding: 查询理解结果

        Returns:
            查询计划
        """
        logger.info(f"开始创建执行计划: {query_id}")

        try:
            # 分析查询意图
            intent = query_understanding.intent

            # 生成子查询
            sub_queries = await self._generate_sub_queries(
                original_query,
                query_understanding
            )

            # 选择检索策略
            retrieval_strategies = self._select_retrieval_strategies(
                intent,
                query_understanding
            )

            # 确定迭代参数
            max_iterations = self._determine_iterations(
                query_understanding.query_complexity,
                intent
            )

            confidence_threshold = self._determine_confidence_threshold(
                query_understanding.query_complexity
            )

            # 构建查询计划
            plan = QueryPlan(
                query_id=query_id,
                original_query=original_query,
                query_type=intent.primary_intent,
                sub_queries=sub_queries,
                retrieval_strategies=retrieval_strategies,
                entities=[{
                    'text': e.text,
                    'type': e.type,
                    'confidence': e.confidence
                } for e in query_understanding.entities],
                time_constraints=intent.time_range,
                max_iterations=max_iterations,
                confidence_threshold=confidence_threshold,
                created_at=query_understanding.metadata.get('processed_at')
            )

            logger.info(f"执行计划创建完成: 策略={retrieval_strategies}, 迭代次数={max_iterations}")

            return plan

        except Exception as e:
            logger.error(f"创建执行计划失败: {e}")
            # 返回默认计划
            return QueryPlan(
                query_id=query_id,
                original_query=original_query,
                query_type=QueryType.FACTUAL,
                sub_queries=[original_query],
                retrieval_strategies=['vector', 'keyword'],
                entities=[],
                max_iterations=2,
                confidence_threshold=0.6
            )

    async def _generate_sub_queries(
        self,
        original_query: str,
        query_understanding: QueryUnderstanding
    ) -> List[str]:
        """生成子查询"""
        sub_queries = [original_query]

        # 基于查询复杂度决定是否生成子查询
        if query_understanding.query_complexity == 'simple':
            return sub_queries

        intent = query_understanding.intent

        # 比较查询：为每个比较实体生成子查询
        if intent.comparison_entities:
            for entity in intent.comparison_entities:
                sub_query = f"{entity} {original_query}"
                sub_queries.append(sub_query)

        # 分析查询：根据分析类型生成子查询
        if intent.analysis_type:
            analysis_query = f"{original_query} {intent.analysis_type}"
            sub_queries.append(analysis_query)

        # 时间范围查询：添加时间约束
        if intent.time_range:
            time_constraints = []
            for key, value in intent.time_range.items():
                if key == 'recent':
                    time_constraints.append(f"最近{value}")
                elif key == 'year':
                    time_constraints.append(f"{value}年")
                elif key == 'quarter':
                    time_constraints.append(f"{value}季度")

            for constraint in time_constraints:
                time_query = f"{original_query} {constraint}"
                sub_queries.append(time_query)

        # 基于实体生成子查询
        entities = query_understanding.entities
        if entities:
            company_entities = [e.text for e in entities if e.type == 'COMPANY']
            if company_entities:
                for company in company_entities[:2]:  # 最多2个公司
                    entity_query = f"{company} {original_query}"
                    sub_queries.append(entity_query)

        # 基于扩展查询生成子查询
        if query_understanding.expanded_queries:
            sub_queries.extend(query_understanding.expanded_queries[:2])  # 最多2个扩展查询

        # 去重并限制数量
        sub_queries = list(set(sub_queries))
        return sub_queries[:5]  # 最多5个子查询

    def _select_retrieval_strategies(
        self,
        intent: QueryIntent,
        query_understanding: QueryUnderstanding
    ) -> List[str]:
        """选择检索策略"""
        # 基于意图选择基础策略
        base_strategies = self.intent_strategy_mapping.get(
            intent.primary_intent,
            ['vector', 'keyword']
        )

        # 根据实体存在情况调整
        if query_understanding.entities:
            if 'graph' not in base_strategies:
                base_strategies.append('graph')

        # 根据复杂度调整
        if query_understanding.query_complexity == 'complex':
            if 'graph' not in base_strategies:
                base_strategies.insert(1, 'graph')  # 插入到中间位置

        # 根据能力需求调整
        capabilities = query_understanding.required_capabilities
        if 'comparison' in capabilities and 'graph' not in base_strategies:
            base_strategies.append('graph')

        return base_strategies

    def _determine_iterations(
        self,
        complexity: str,
        intent: QueryIntent
    ) -> int:
        """确定迭代次数"""
        base_iterations = {
            'simple': 1,
            'moderate': 2,
            'complex': 3
        }

        iterations = base_iterations.get(complexity, 2)

        # 根据意图调整
        if intent.primary_intent in [QueryType.ANALYTICAL, QueryType.CAUSAL]:
            iterations += 1

        # 根据比较实体数量调整
        if intent.comparison_entities and len(intent.comparison_entities) > 2:
            iterations += 1

        return min(iterations, 4)  # 最多4次迭代

    def _determine_confidence_threshold(self, complexity: str) -> float:
        """确定置信度阈值"""
        thresholds = {
            'simple': 0.8,
            'moderate': 0.7,
            'complex': 0.6
        }
        return thresholds.get(complexity, 0.7)

    def create_detailed_plan(
        self,
        query_plan: QueryPlan,
        query_understanding: QueryUnderstanding
    ) -> Dict[str, Any]:
        """
        创建详细执行计划

        Args:
            query_plan: 基础查询计划
            query_understanding: 查询理解结果

        Returns:
            详细执行计划
        """
        detailed_plan = {
            'query_id': query_plan.query_id,
            'original_query': query_plan.original_query,
            'execution_phases': []
        }

        # 第一阶段：基础检索
        detailed_plan['execution_phases'].append({
            'phase': 'initial_retrieval',
            'description': '基于原始查询进行初步检索',
            'tasks': self._create_retrieval_tasks(
                query_plan.original_query,
                query_plan.retrieval_strategies,
                priority=1
            )
        })

        # 第二阶段：子查询检索
        if len(query_plan.sub_queries) > 1:
            detailed_plan['execution_phases'].append({
                'phase': 'subquery_retrieval',
                'description': '执行子查询以补充信息',
                'tasks': [
                    self._create_subtask(
                        sub_query,
                        query_plan.retrieval_strategies,
                        priority=2,
                        dependencies=['initial_retrieval']
                    )
                    for sub_query in query_plan.sub_queries[1:]
                ]
            })

        # 第三阶段：结果融合
        detailed_plan['execution_phases'].append({
            'phase': 'result_fusion',
            'description': '融合和排序检索结果',
            'tasks': [
                {
                    'task_id': 'fusion_task',
                    'description': '融合多策略检索结果',
                    'method': 'weighted_fusion',
                    'dependencies': [task['task_id'] for task in detailed_plan['execution_phases'][0]['tasks']]
                }
            ]
        })

        # 第四阶段：迭代优化
        if query_plan.max_iterations > 1:
            detailed_plan['execution_phases'].append({
                'phase': 'iterative_optimization',
                'description': '迭代优化检索结果',
                'max_iterations': query_plan.max_iterations - 1,
                'confidence_threshold': query_plan.confidence_threshold
            })
        )

        # 添加元数据
        detailed_plan['metadata'] = {
            'query_complexity': query_understanding.query_complexity,
            'required_capabilities': query_understanding.required_capabilities,
            'estimated_execution_time': self._estimate_execution_time(detailed_plan),
            'resource_requirements': self._estimate_resources(detailed_plan)
        }

        return detailed_plan

    def _create_retrieval_tasks(
        self,
        query: str,
        strategies: List[str],
        priority: int = 1
    ) -> List[Dict[str, Any]]:
        """创建检索任务"""
        tasks = []
        for strategy in strategies:
            task = {
                'task_id': f"{strategy}_retrieval",
                'description': f"使用{strategy}策略检索: {query}",
                'strategy': strategy,
                'query': query,
                'parameters': self.strategy_configs[strategy]['parameters'],
                'weight': self.strategy_configs[strategy]['default_weight'],
                'priority': priority
            }
            tasks.append(task)
        return tasks

    def _create_subtask(
        self,
        sub_query: str,
        strategies: List[str],
        priority: int,
        dependencies: List[str]
    ) -> Dict[str, Any]:
        """创建子任务"""
        return {
            'task_id': f"subtask_{hash(sub_query) % 10000}",
            'description': f"执行子查询: {sub_query}",
            'query': sub_query,
            'strategies': strategies,
            'priority': priority,
            'dependencies': dependencies
        }

    def _estimate_execution_time(self, detailed_plan: Dict[str, Any]) -> float:
        """估算执行时间（秒）"""
        base_time = 2.0  # 基础时间

        # 根据检索任务数量
        retrieval_tasks = sum(
            len(phase.get('tasks', []))
            for phase in detailed_plan['execution_phases']
            if 'retrieval' in phase.get('phase', '')
        )
        base_time += retrieval_tasks * 0.5

        # 根据迭代次数
        for phase in detailed_plan['execution_phases']:
            if phase.get('phase') == 'iterative_optimization':
                max_iterations = phase.get('max_iterations', 1)
                base_time += max_iterations * 1.5

        return base_time

    def _estimate_resources(self, detailed_plan: Dict[str, Any]) -> Dict[str, Any]:
        """估算资源需求"""
        # 计算并发任务数
        max_concurrent = 0
        for phase in detailed_plan['execution_phases']:
            tasks = len(phase.get('tasks', []))
            max_concurrent = max(max_concurrent, tasks)

        return {
            'max_concurrent_tasks': max_concurrent,
            'estimated_memory_mb': max_concurrent * 100,  # 每个任务约100MB
            'requires_gpu': False,
            'estimated_api_calls': len(detailed_plan['execution_phases']) * 3
        }