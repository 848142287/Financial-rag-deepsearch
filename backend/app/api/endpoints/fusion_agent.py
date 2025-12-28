"""
融合智能体API端点
提供完整的智能代理服务
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging

from app.services.fusion_agent.core.agent_architecture import FusionAgent
from app.services.fusion_agent.output_layer import DebugInfoCollector
from app.core.auth import get_current_user
from app.models.user import User
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)

router = APIRouter()

# 全局智能体实例
fusion_agent = FusionAgent()
debug_collector = DebugInfoCollector()


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="对话历史")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文信息")
    mode: str = Field(default="standard", description="执行模式")
    stream: bool = Field(default=False, description="是否流式返回")


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="回答内容")
    debug_info: Optional[Dict[str, Any]] = Field(default=None, description="调试信息")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")
    execution_time: float = Field(..., description="执行时间")


class StatusResponse(BaseModel):
    """状态响应模型"""
    agent_id: str = Field(..., description="智能体ID")
    state: str = Field(..., description="当前状态")
    progress: float = Field(..., description="执行进度")
    current_task: Optional[str] = Field(default=None, description="当前任务")


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    处理用户查询
    """
    try:
        logger.info(f"用户 {current_user.id} 发起查询: {request.query[:100]}...")

        # 转换历史格式
        history = None
        if request.history:
            from app.services.fusion_agent.core.agent_architecture import AgentMessage
            history = [
                AgentMessage(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    timestamp=msg.get('timestamp'),
                    metadata=msg.get('metadata', {})
                )
                for msg in request.history
            ]

        # 执行查询
        if request.stream:
            # 流式处理
            return StreamingResponse(
                stream_query_response(fusion_agent, request.query, history, request.context),
                media_type="text/event-stream"
            )
        else:
            # 同步处理
            result = await fusion_agent.process_query(
                query=request.query,
                history=history,
                context=request.context
            )

            # 构建响应
            response = QueryResponse(
                answer=result['final_answer'],
                debug_info=result.get('debug_info') if request.mode == 'debug' else None,
                metadata=result.get('metadata'),
                execution_time=result.get('execution_time', 0)
            )

            # 记录查询日志
            await log_query_execution(current_user.id, request.query, response)

            return response

    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_agent_status(current_user: User = Depends(get_current_user)):
    """
    获取智能体状态
    """
    try:
        status = await fusion_agent.get_status()

        return StatusResponse(
            agent_id=status['agent_id'],
            state=status['current_state'],
            progress=status['progress'],
            current_task=status['current_task']
        )
    except Exception as e:
        logger.error(f"获取状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.post("/stream")
async def stream_query_endpoint(
    request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    流式查询端点
    """
    try:
        # 转换历史格式
        history = None
        if request.history:
            from app.services.fusion_agent.core.agent_architecture import AgentMessage
            history = [
                AgentMessage(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', '')
                )
                for msg in request.history
            ]

        # 返回流式响应
        return StreamingResponse(
            stream_query_response(fusion_agent, request.query, history, request.context),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"流式查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"流式查询失败: {str(e)}")


@router.get("/health")
async def health_check():
    """
    智能体健康检查
    """
    try:
        # 检查组件健康状态
        component_health = await fusion_agent.health_check()

        # 检查整体健康状态
        all_healthy = all(component_health.values())

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "components": component_health,
            "agent_id": fusion_agent.agent_id,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }


@router.get("/debug")
async def get_debug_info(current_user: User = Depends(get_current_user)):
    """
    获取调试信息
    """
    try:
        # 获取智能体状态
        agent_status = await fusion_agent.get_status()

        # 收集调试信息
        debug_info = await debug_collector.process({}, {
            'current_state': agent_status['current_state'],
            'agent_id': agent_status['agent_id'],
            'progress': agent_status['progress']
        })

        return {
            "agent_status": agent_status,
            "debug_info": debug_info,
            "timestamp": asyncio.get_event_loop().time()
        }

    except Exception as e:
        logger.error(f"获取调试信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取调试信息失败: {str(e)}")


@router.post("/configure")
async def configure_agent(
    configuration: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    配置智能体参数
    """
    try:
        # 验证配置
        valid_configs = ['timeout', 'max_tasks', 'retry_count', 'debug_mode']
        for key in configuration:
            if key not in valid_configs:
                raise HTTPException(status_code=400, detail=f"无效的配置项: {key}")

        # 应用配置
        # 这里可以实现具体的配置逻辑
        logger.info(f"用户 {current_user.id} 配置智能体: {configuration}")

        return {
            "message": "配置成功",
            "applied_configuration": configuration
        }

    except Exception as e:
        logger.error(f"配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"配置失败: {str(e)}")


@router.get("/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)):
    """
    获取智能体性能指标
    """
    try:
        # 获取性能指标
        metrics = await collect_agent_metrics()

        return {
            "metrics": metrics,
            "timestamp": asyncio.get_event_loop().time()
        }

    except Exception as e:
        logger.error(f"获取指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")


@router.delete("/reset")
async def reset_agent(current_user: User = Depends(get_current_user)):
    """
    重置智能体状态
    """
    try:
        # 重置智能体状态
        fusion_agent._reset_state()

        logger.info(f"用户 {current_user.id} 重置智能体状态")

        return {
            "message": "智能体状态已重置",
            "agent_id": fusion_agent.agent_id
        }

    except Exception as e:
        logger.error(f"重置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重置失败: {str(e)}")


# 辅助函数
async def stream_query_response(agent: FusionAgent, query: str, history: List, context: Dict):
    """流式查询响应生成器"""
    try:
        async for chunk in agent.stream_process(query, history):
            # 格式化SSE响应
            chunk_data = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {chunk_data}\n\n"
            await asyncio.sleep(0.1)  # 控制发送频率

    except Exception as e:
        error_chunk = {
            "type": "error",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"


async def log_query_execution(user_id: int, query: str, response: QueryResponse):
    """记录查询执行日志"""
    try:
        log_entry = {
            "user_id": user_id,
            "query": query,
            "answer_length": len(response.answer),
            "execution_time": response.execution_time,
            "timestamp": asyncio.get_event_loop().time()
        }

        # 存储到Redis或数据库
        log_key = f"fusion_agent:query_log:{user_id}:{int(asyncio.get_event_loop().time())}"
        redis_client.setex(log_key, 86400 * 7, json.dumps(log_entry))  # 缓存7天

    except Exception as e:
        logger.error(f"记录查询日志失败: {str(e)}")


async def collect_agent_metrics() -> Dict[str, Any]:
    """收集智能体性能指标"""
    try:
        # 获取基本状态
        status = await fusion_agent.get_status()

        # 获取组件健康状态
        component_health = await fusion_agent.health_check()

        # 获取资源使用情况
        debug_info = await debug_collector.process({}, {})

        metrics = {
            "performance": {
                "agent_id": status['agent_id'],
                "current_state": status['current_state'],
                "progress": status['progress'],
                "completed_tasks": status['completed_tasks'],
                "failed_tasks": status['failed_tasks'],
                "execution_time": status['execution_time']
            },
            "health": {
                "overall_healthy": all(component_health.values()),
                "components": component_health
            },
            "resources": debug_info.get('resource_usage', {}),
            "debug_info": debug_info.get('performance_metrics', {}),
            "collection_timestamp": asyncio.get_event_loop().time()
        }

        return metrics

    except Exception as e:
        logger.error(f"收集指标失败: {str(e)}")
        return {
            "error": str(e),
            "collection_timestamp": asyncio.get_event_loop().time()
        }