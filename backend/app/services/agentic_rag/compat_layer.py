"""
Agentic RAG 兼容层
提供旧API到新API的适配，确保平滑过渡
"""

from app.core.structured_logging import get_structured_logger
from typing import Dict, List, Optional, Any

# 导入新实现

logger = get_structured_logger(__name__)

# ========== 别名定义，提供向后兼容 ==========

# 计划阶段兼容
PlanPhase = AgenticRAGPlanner
rag_plan_phase = AgenticRAGPlanner()

# 执行阶段兼容
ExecutePhase = AgenticRAGExecutor
rag_execute_phase = AgenticRAGExecutor()

# 生成阶段兼容
GenerationPhase = AgenticRAGGenerator
rag_generation_phase = AgenticRAGGenerator()

# ========== 适配器类 ==========

class PlanPhaseAdapter:
    """
    计划阶段适配器
    将旧API调用转换为新API
    """

    def __init__(self):
        self.planner = AgenticRAGPlanner()

    async def process_query(
        self,
        query: str,
        context: Any = None
    ) -> RetrievalPlan:
        """
        旧API: process_query
        新API: create_retrieval_plan
        """
        logger.debug("PlanPhaseAdapter: process_query -> create_retrieval_plan")
        return await self.planner.create_retrieval_plan(query, context)

    async def _preprocess_query(self, query: str, context: Any) -> str:
        """预处理查询（兼容方法）"""
        # 调用新实现
        return await self.planner._extract_keywords(query)

class ExecutePhaseAdapter:
    """
    执行阶段适配器
    将旧API调用转换为新API
    """

    def __init__(self):
        self.executor = AgenticRAGExecutor()

    async def execute_retrieval(
        self,
        plan: RetrievalPlan,
        document_ids: Optional[List[int]] = None,
        task_id: Optional[str] = None
    ) -> FusedResult:
        """
        旧API: execute_retrieval
        新API: execute_plan
        """
        logger.debug("ExecutePhaseAdapter: execute_retrieval -> execute_plan")
        result = await self.executor.execute_plan(plan)

        # 转换返回类型
        # ExecutionResult -> FusedResult (适配)
        if result.fused_results:
            return result.fused_results[0] if result.fused_results else FusedResult(
                content="",
                sources=[],
                overall_score=0.0,
                method_contributions={},
                metadata={}
            )
        else:
            # 空结果
            return FusedResult(
                content="",
                sources=[],
                overall_score=0.0,
                method_contributions={},
                metadata={"error": result.error_message or "No results"}
            )

class GenerationPhaseAdapter:
    """
    生成阶段适配器
    将旧API调用转换为新API
    """

    def __init__(self):
        self.generator = AgenticRAGGenerator()

    async def generate_answer(
        self,
        plan: RetrievalPlan,
        fused_result: Any,
        task_id: Optional[str] = None
    ) -> GenerationResult:
        """
        旧API: generate_answer
        新API: generate_answer (参数略有不同)
        """
        logger.debug("GenerationPhaseAdapter: generate_answer (adapted)")

        # 构建plan_context
        plan_context = {
            "query_analysis": plan.query_analysis if hasattr(plan, 'query_analysis') else None,
            "constraints": plan.constraints if hasattr(plan, 'constraints') else {}
        }

        # 调用新API
        result = await self.generator.generate_answer(
            execution_result=fused_result,  # 注意参数名
            plan_context=plan_context
        )

        return result

# ========== 便捷函数 ==========

async def process_with_legacy_api(
    query: str,
    retrieval_level: str = "fast",
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    使用旧API风格处理查询（内部使用新实现）

    Args:
        query: 查询文本
        retrieval_level: 检索级别
        context: 上下文信息

    Returns:
        处理结果字典
    """
    try:
        # 阶段1：计划
        planner = AgenticRAGPlanner()
        plan = await planner.create_retrieval_plan(query, context)

        # 阶段2：执行
        executor = AgenticRAGExecutor()
        execution_result = await executor.execute_plan(plan)

        # 阶段3：生成
        generator = AgenticRAGGenerator()
        plan_context = {
            "query_analysis": plan.query_analysis,
            "constraints": plan.constraints
        }
        generation_result = await generator.generate_answer(
            execution_result,
            plan_context
        )

        # 组合结果
        return {
            "query": query,
            "retrieval_level": retrieval_level,
            "plan": plan,
            "execution_result": execution_result,
            "generation_result": generation_result,
            "success": True
        }

    except Exception as e:
        logger.error(f"process_with_legacy_api 失败: {e}")
        return {
            "query": query,
            "retrieval_level": retrieval_level,
            "error": str(e),
            "success": False
        }

# ========== 导出兼容接口 ==========

__all__ = [
    # 适配器类
    "PlanPhaseAdapter",
    "ExecutePhaseAdapter",
    "GenerationPhaseAdapter",

    # 旧类别名（向后兼容）
    "PlanPhase",
    "ExecutePhase",
    "GenerationPhase",

    # 便捷函数
    "process_with_legacy_api",

    # 新实现（直接导出）
    "AgenticRAGPlanner",
    "AgenticRAGExecutor",
    "AgenticRAGGenerator",
]
