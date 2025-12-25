"""
Agentic RAG 系统
整合计划、执行、生成三个阶段的完整流程
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import time

from .planner import RetrievalPlan, rag_planner
from .executor import ExecutionResult, rag_executor
from .generator import GenerationResult, rag_generator

logger = logging.getLogger(__name__)


class RetrievalLevel(Enum):
    """检索级别"""
    FAST = "fast"                 # 快速检索：P95 ≤ 3秒
    ENHANCED = "enhanced"         # 增强检索：P95 ≤ 8秒
    DEEP_SEARCH = "deep_search"   # 深度检索：支持异步


class ProcessStatus(Enum):
    """处理状态"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgenticRAGRequest:
    """Agentic RAG 请求"""
    query: str
    retrieval_level: RetrievalLevel = RetrievalLevel.FAST
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    async_mode: bool = False


@dataclass
class AgenticRAGResponse:
    """Agentic RAG 响应"""
    request_id: str
    query: str
    answer: str
    retrieval_level: RetrievalLevel
    sources: List[str]
    quality_score: float
    processing_time: float
    plan: Optional[RetrievalPlan] = None
    execution_result: Optional[ExecutionResult] = None
    generation_result: Optional[GenerationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AsyncTaskInfo:
    """异步任务信息"""
    task_id: str
    status: ProcessStatus
    progress: float = 0.0
    current_stage: str = ""
    result: Optional[AgenticRAGResponse] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class AgenticRAGSystem:
    """Agentic RAG 系统"""

    def __init__(self):
        # 异步任务存储
        self.async_tasks: Dict[str, AsyncTaskInfo] = {}

        # 级别配置
        self.level_configs = {
            RetrievalLevel.FAST: {
                "max_plan_time": 1.0,
                "max_execution_time": 5.0,
                "max_generation_time": 2.0,
                "max_results": 5,
                "quality_threshold": 0.6
            },
            RetrievalLevel.ENHANCED: {
                "max_plan_time": 2.0,
                "max_execution_time": 10.0,
                "max_generation_time": 4.0,
                "max_results": 15,
                "quality_threshold": 0.75
            },
            RetrievalLevel.DEEP_SEARCH: {
                "max_plan_time": 3.0,
                "max_execution_time": 20.0,
                "max_generation_time": 8.0,
                "max_results": 30,
                "quality_threshold": 0.8
            }
        }

    async def process_query(self, request: AgenticRAGRequest) -> AgenticRAGResponse:
        """
        处理查询请求

        Args:
            request: Agentic RAG 请求

        Returns:
            AgenticRAGResponse: 处理响应
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            logger.info(f"开始处理查询 {request_id}: {request.query[:100]}...")

            if request.async_mode:
                # 异步模式
                task_id = await self._create_async_task(request)
                return await self._wait_for_async_task(task_id)
            else:
                # 同步模式
                return await self._process_sync(request_id, request)

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"查询处理失败 {request_id}: {str(e)}")

            return AgenticRAGResponse(
                request_id=request_id,
                query=request.query,
                answer="抱歉，查询处理过程中出现错误，请稍后重试。",
                retrieval_level=request.retrieval_level,
                sources=[],
                quality_score=0.0,
                processing_time=total_time,
                metadata={"error": str(e)}
            )

    async def _process_sync(self, request_id: str, request: AgenticRAGRequest) -> AgenticRAGResponse:
        """同步处理"""
        # 阶段1：计划
        plan = await self._planning_stage(request)

        # 阶段2：执行
        execution_result = await self._execution_stage(plan, request)

        # 阶段3：生成
        generation_result = await self._generation_stage(execution_result, plan)

        # 构建响应
        response = AgenticRAGResponse(
            request_id=request_id,
            query=request.query,
            answer=generation_result.answer,
            retrieval_level=request.retrieval_level,
            sources=generation_result.sources,
            quality_score=generation_result.quality_score,
            processing_time=time.time() - time.time(),
            plan=plan,
            execution_result=execution_result,
            generation_result=generation_result,
            metadata={
                "session_id": request.session_id,
                "user_id": request.user_id,
                "async_mode": False
            }
        )

        logger.info(f"同步处理完成 {request_id}")
        return response

    async def _planning_stage(self, request: AgenticRAGRequest) -> RetrievalPlan:
        """计划阶段"""
        logger.info(f"开始计划阶段: {request.retrieval_level.value}")

        # 配置级别特定的约束
        level_config = self.level_configs[request.retrieval_level]

        # 修改检索器配置
        self._configure_level_settings(request.retrieval_level, level_config)

        # 创建检索计划
        plan = await rag_planner.create_retrieval_plan(request.query, request.context)

        # 调整计划以符合级别要求
        plan = self._adjust_plan_for_level(plan, request.retrieval_level, level_config)

        logger.info(f"计划阶段完成: {plan.plan_id}")
        return plan

    async def _execution_stage(self, plan: RetrievalPlan, request: AgenticRAGRequest) -> ExecutionResult:
        """执行阶段"""
        logger.info(f"开始执行阶段: {plan.plan_id}")

        # 执行检索
        execution_result = await rag_executor.execute_plan(plan)

        # 根据级别调整结果
        execution_result = self._adjust_result_for_level(execution_result, request.retrieval_level)

        logger.info(f"执行阶段完成: {plan.plan_id}")
        return execution_result

    async def _generation_stage(self, execution_result: ExecutionResult, plan: RetrievalPlan) -> GenerationResult:
        """生成阶段"""
        logger.info(f"开始生成阶段: {plan.plan_id}")

        # 准备生成上下文
        plan_context = {
            "query_analysis": plan.query_analysis,
            "constraints": plan.constraints
        }

        # 生成答案
        generation_result = await rag_generator.generate_answer(execution_result, plan_context)

        logger.info(f"生成阶段完成: {plan.plan_id}")
        return generation_result

    def _configure_level_settings(self, level: RetrievalLevel, config: Dict[str, Any]):
        """配置级别特定设置"""
        # 这里可以根据需要调整各种组件的配置
        # 例如：向量检索的top_k、图谱检索的深度等
        pass

    def _adjust_plan_for_level(self, plan: RetrievalPlan, level: RetrievalLevel, config: Dict[str, Any]) -> RetrievalPlan:
        """根据级别调整计划"""
        # 调整最大结果数量
        for strategy in plan.strategies:
            strategy.max_results = min(strategy.max_results, config["max_results"])
            strategy.quality_threshold = max(strategy.quality_threshold, config["quality_threshold"])

        return plan

    def _adjust_result_for_level(self, result: ExecutionResult, level: RetrievalLevel) -> ExecutionResult:
        """根据级别调整结果"""
        config = self.level_configs[level]

        # 限制结果数量
        max_results = config["max_results"]
        if len(result.fused_results) > max_results:
            result.fused_results = result.fused_results[:max_results]

        return result

    async def _create_async_task(self, request: AgenticRAGRequest) -> str:
        """创建异步任务"""
        task_id = str(uuid.uuid4())

        task_info = AsyncTaskInfo(
            task_id=task_id,
            status=ProcessStatus.PENDING,
            current_stage="任务创建"
        )

        self.async_tasks[task_id] = task_info

        # 启动异步处理
        asyncio.create_task(self._process_async(task_id, request))

        logger.info(f"异步任务创建: {task_id}")
        return task_id

    async def _process_async(self, task_id: str, request: AgenticRAGRequest):
        """异步处理任务"""
        task_info = self.async_tasks[task_id]

        try:
            # 更新状态：计划阶段
            task_info.status = ProcessStatus.PLANNING
            task_info.progress = 0.1
            task_info.current_stage = "计划阶段"
            task_info.updated_at = datetime.now()

            plan = await self._planning_stage(request)

            # 更新进度
            task_info.progress = 0.3
            task_info.updated_at = datetime.now()

            # 执行阶段
            task_info.status = ProcessStatus.EXECUTING
            task_info.current_stage = "执行阶段"
            task_info.progress = 0.3
            task_info.updated_at = datetime.now()

            execution_result = await self._execution_stage(plan, request)

            # 更新进度
            task_info.progress = 0.7
            task_info.updated_at = datetime.now()

            # 生成阶段
            task_info.status = ProcessStatus.GENERATING
            task_info.current_stage = "生成阶段"
            task_info.progress = 0.7
            task_info.updated_at = datetime.now()

            generation_result = await self._generation_stage(execution_result, plan)

            # 构建最终结果
            response = AgenticRAGResponse(
                request_id=task_id,
                query=request.query,
                answer=generation_result.answer,
                retrieval_level=request.retrieval_level,
                sources=generation_result.sources,
                quality_score=generation_result.quality_score,
                processing_time=time.time() - time.time(),
                plan=plan,
                execution_result=execution_result,
                generation_result=generation_result
            )

            # 更新任务状态
            task_info.status = ProcessStatus.COMPLETED
            task_info.progress = 1.0
            task_info.current_stage = "处理完成"
            task_info.result = response
            task_info.updated_at = datetime.now()

            logger.info(f"异步任务完成: {task_id}")

        except Exception as e:
            task_info.status = ProcessStatus.FAILED
            task_info.error_message = str(e)
            task_info.current_stage = "处理失败"
            task_info.updated_at = datetime.now()

            logger.error(f"异步任务失败 {task_id}: {str(e)}")

    async def _wait_for_async_task(self, task_id: str, timeout: float = 60.0) -> AgenticRAGResponse:
        """等待异步任务完成"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            task_info = self.async_tasks.get(task_id)

            if not task_info:
                raise ValueError(f"任务不存在: {task_id}")

            if task_info.status == ProcessStatus.COMPLETED:
                return task_info.result
            elif task_info.status == ProcessStatus.FAILED:
                raise RuntimeError(f"任务执行失败: {task_info.error_message}")

            await asyncio.sleep(0.1)

        raise TimeoutError(f"任务执行超时: {task_id}")

    async def get_task_status(self, task_id: str) -> Optional[AsyncTaskInfo]:
        """获取任务状态"""
        return self.async_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task_info = self.async_tasks.get(task_id)

        if task_info and task_info.status in [ProcessStatus.PENDING, ProcessStatus.PLANNING, ProcessStatus.EXECUTING]:
            task_info.status = ProcessStatus.FAILED
            task_info.error_message = "任务已取消"
            task_info.updated_at = datetime.now()
            return True

        return False

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        total_tasks = len(self.async_tasks)
        completed_tasks = sum(1 for t in self.async_tasks.values() if t.status == ProcessStatus.COMPLETED)
        failed_tasks = sum(1 for t in self.async_tasks.values() if t.status == ProcessStatus.FAILED)

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "active_tasks": total_tasks - completed_tasks - failed_tasks
        }


# 全局系统实例
agentic_rag_system = AgenticRAGSystem()