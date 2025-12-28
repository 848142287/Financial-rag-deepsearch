"""
工作流引擎
基于LangGraph实现智能工作流编排和执行
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

try:
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory
    from langchain.callbacks.base import BaseCallbackHandler
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    # 如果LangChain未安装，提供模拟类
    BaseMessage = object
    HumanMessage = object
    AIMessage = object
    SystemMessage = object
    ConversationBufferMemory = object
    BaseCallbackHandler = object

from app.core.config import settings
from app.core.events import event_publisher
from app.core.events.event_types import EventType

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """节点类型"""
    START = "start"
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    TRANSFORM = "transform"
    END = "end"


@dataclass
class WorkflowNode:
    """工作流节点"""
    id: str
    name: str
    node_type: NodeType
    description: Optional[str] = None
    function: Optional[Callable] = None
    agent: Optional[str] = None
    tool: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowEdge:
    """工作流边"""
    from_node: str
    to_node: str
    condition: Optional[Callable] = None
    description: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowState:
    """工作流状态"""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_node: Optional[str] = None
    messages: List[BaseMessage] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    node_results: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """添加消息"""
        self.messages.append(message)

    def update_data(self, key: str, value: Any) -> None:
        """更新数据"""
        self.data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """获取数据"""
        return self.data.get(key, default)

    def add_node_result(self, node_id: str, result: Any) -> None:
        """添加节点执行结果"""
        self.node_results[node_id] = result

    def get_node_result(self, node_id: str) -> Any:
        """获取节点执行结果"""
        return self.node_results.get(node_id)


class WorkflowCallbackHandler(BaseCallbackHandler):
    """工作流回调处理器"""

    def __init__(self, workflow_id: str, execution_id: str):
        self.workflow_id = workflow_id
        self.execution_id = execution_id

    async def on_node_start(self, node_id: str, state: WorkflowState) -> None:
        """节点开始执行"""
        logger.info(f"工作流 {self.workflow_id} 节点 {node_id} 开始执行")

        # 发布事件
        await event_publisher.workflow_node_started(
            workflow_id=self.workflow_id,
            execution_id=self.execution_id,
            node_id=node_id,
            state=state
        )

    async def on_node_end(self, node_id: str, result: Any, state: WorkflowState) -> None:
        """节点执行结束"""
        logger.info(f"工作流 {self.workflow_id} 节点 {node_id} 执行完成")

        # 发布事件
        await event_publisher.workflow_node_completed(
            workflow_id=self.workflow_id,
            execution_id=self.execution_id,
            node_id=node_id,
            result=result
        )

    async def on_node_error(self, node_id: str, error: Exception, state: WorkflowState) -> None:
        """节点执行错误"""
        logger.error(f"工作流 {self.workflow_id} 节点 {node_id} 执行失败: {error}")

        # 发布事件
        await event_publisher.workflow_node_failed(
            workflow_id=self.workflow_id,
            execution_id=self.execution_id,
            node_id=node_id,
            error=str(error)
        )

    async def on_workflow_end(self, state: WorkflowState) -> None:
        """工作流执行结束"""
        logger.info(f"工作流 {self.workflow_id} �{state.status.value}完成")

        # 发布事件
        if state.status == WorkflowStatus.COMPLETED:
            await event_publisher.workflow_completed(
                workflow_id=self.workflow_id,
                execution_id=self.execution_id,
                result=state.data
            )
        elif state.status == WorkflowStatus.FAILED:
            await event_publisher.workflow_failed(
                workflow_id=self.workflow_id,
                execution_id=self.execution_id,
                error=state.error
            )


class WorkflowEngine:
    """工作流引擎"""

    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowState] = {}
        self.agents: Dict[str, Any] = {}
        self.tools: Dict[str, Any] = {}
        self.callback_handlers: Dict[str, WorkflowCallbackHandler] = {}
        self._lock = asyncio.Lock()
        self._checkpointer = SqliteSaver.from_conn_string(":memory:")

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """注册工作流"""
        self.workflows[workflow.id] = workflow
        logger.info(f"注册工作流: {workflow.name} ({workflow.id})")

    def register_agent(self, name: str, agent: Any) -> None:
        """注册智能体"""
        self.agents[name] = agent
        logger.info(f"注册智能体: {name}")

    def register_tool(self, name: str, tool: Any) -> None:
        """注册工具"""
        self.tools[name] = tool
        logger.info(f"注册工具: {name}")

    async def execute_workflow(self, workflow_id: str,
                             initial_state: Optional[Dict[str, Any]] = None,
                             user_id: Optional[str] = None,
                             session_id: Optional[str] = None) -> str:
        """执行工作流"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流不存在: {workflow_id}")

        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())

        # 创建工作流状态
        state = WorkflowState(
            workflow_id=workflow_id,
            execution_id=execution_id,
            data=initial_state or {},
            metadata={
                'user_id': user_id,
                'session_id': session_id,
                'workflow_name': workflow.name,
                'workflow_version': workflow.version
            }
        )

        # 创建回调处理器
        callback_handler = WorkflowCallbackHandler(workflow_id, execution_id)
        self.callback_handlers[execution_id] = callback_handler

        # 添加到活跃执行列表
        self.active_executions[execution_id] = state

        try:
            # 发布工作流开始事件
            await event_publisher.workflow_started(
                workflow_id=workflow_id,
                execution_id=execution_id,
                initial_state=state
            )

            # 构建LangGraph
            graph = await self._build_graph(workflow, state)

            # 执行工作流
            state.status = WorkflowStatus.RUNNING

            result = await graph.ainvoke(
                state.data,
                config={"configurable": {"thread_id": execution_id}}
            )

            # 更新状态
            state.data.update(result)
            state.status = WorkflowStatus.COMPLETED
            state.end_time = datetime.now()

            await callback_handler.on_workflow_end(state)

            return execution_id

        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            state.status = WorkflowStatus.FAILED
            state.error = str(e)
            state.end_time = datetime.now()

            await callback_handler.on_workflow_end(state)
            raise

        finally:
            # 清理回调处理器
            if execution_id in self.callback_handlers:
                del self.callback_handlers[execution_id]

    async def _build_graph(self, workflow: WorkflowDefinition,
                          initial_state: WorkflowState) -> StateGraph:
        """构建LangGraph"""

        # 创建状态图
        workflow_graph = StateGraph(dict)

        # 添加节点
        for node in workflow.nodes:
            if node.node_type == NodeType.START:
                workflow_graph.add_node(node.id, self._create_start_node(node))
            elif node.node_type == NodeType.AGENT:
                workflow_graph.add_node(node.id, self._create_agent_node(node))
            elif node.node_type == NodeType.TOOL:
                workflow_graph.add_node(node.id, self._create_tool_node(node))
            elif node.node_type == NodeType.CONDITION:
                workflow_graph.add_node(node.id, self._create_condition_node(node))
            elif node.node_type == NodeType.TRANSFORM:
                workflow_graph.add_node(node.id, self._create_transform_node(node))
            elif node.node_type == NodeType.END:
                workflow_graph.add_node(node.id, self._create_end_node(node))

        # 添加边
        for edge in workflow.edges:
            if edge.condition:
                workflow_graph.add_conditional_edges(
                    edge.from_node,
                    edge.condition,
                    {True: edge.to_node, False: END}
                )
            else:
                workflow_graph.add_edge(edge.from_node, edge.to_node)

        # 设置入口点
        start_nodes = [n.id for n in workflow.nodes if n.node_type == NodeType.START]
        if start_nodes:
            workflow_graph.set_entry_point(start_nodes[0])

        return workflow_graph.compile(checkpointer=self._checkpointer)

    def _create_start_node(self, node: WorkflowNode):
        """创建开始节点"""
        async def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"开始节点执行: {node.name}")
            # 初始化状态
            state.update(node.config.get('initial_data', {}))
            return state
        return start_node

    def _create_agent_node(self, node: WorkflowNode):
        """创建智能体节点"""
        async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"智能体节点执行: {node.name}")

            if node.agent not in self.agents:
                raise ValueError(f"智能体不存在: {node.agent}")

            agent = self.agents[node.agent]

            # 准备输入
            input_data = {**state, **node.config}

            # 执行智能体
            try:
                result = await agent.arun(input_data)
                state[f"{node.id}_result"] = result
                return state
            except Exception as e:
                logger.error(f"智能体 {node.agent} 执行失败: {e}")
                state[f"{node.id}_error"] = str(e)
                return state

        return agent_node

    def _create_tool_node(self, node: WorkflowNode):
        """创建工具节点"""
        async def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"工具节点执行: {node.name}")

            if node.tool not in self.tools:
                raise ValueError(f"工具不存在: {node.tool}")

            tool = self.tools[node.tool]

            # 准备输入参数
            tool_input = {**state, **node.config}

            # 执行工具
            try:
                if hasattr(tool, 'arun'):
                    result = await tool.arun(**tool_input)
                else:
                    result = tool.run(**tool_input)

                state[f"{node.id}_result"] = result
                return state
            except Exception as e:
                logger.error(f"工具 {node.tool} 执行失败: {e}")
                state[f"{node.id}_error"] = str(e)
                return state

        return tool_node

    def _create_condition_node(self, node: WorkflowNode):
        """创建条件节点"""
        async def condition_node(state: Dict[str, Any]) -> bool:
            logger.info(f"条件节点执行: {node.name}")

            if not node.function:
                return True

            try:
                result = await node.function(state)
                return bool(result)
            except Exception as e:
                logger.error(f"条件节点执行失败: {e}")
                return False

        return condition_node

    def _create_transform_node(self, node: WorkflowNode):
        """创建转换节点"""
        async def transform_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"转换节点执行: {node.name}")

            if not node.function:
                return state

            try:
                result = await node.function(state)
                if isinstance(result, dict):
                    state.update(result)
                else:
                    state[f"{node.id}_result"] = result

                return state
            except Exception as e:
                logger.error(f"转换节点执行失败: {e}")
                state[f"{node.id}_error"] = str(e)
                return state

        return transform_node

    def _create_end_node(self, node: WorkflowNode):
        """创建结束节点"""
        async def end_node(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"结束节点执行: {node.name}")

            # 添加结束标记
            state['_end_time'] = datetime.now().isoformat()
            state['_status'] = 'completed'

            return state

        return end_node

    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowState]:
        """获取执行状态"""
        return self.active_executions.get(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        if execution_id in self.active_executions:
            state = self.active_executions[execution_id]
            state.status = WorkflowStatus.CANCELLED
            state.end_time = datetime.now()

            # 发布取消事件
            await event_publisher.workflow_cancelled(
                workflow_id=state.workflow_id,
                execution_id=execution_id
            )

            logger.info(f"工作流执行已取消: {execution_id}")
            return True

        return False

    async def pause_execution(self, execution_id: str) -> bool:
        """暂停执行"""
        if execution_id in self.active_executions:
            state = self.active_executions[execution_id]
            state.status = WorkflowStatus.PAUSED

            logger.info(f"工作流执行已暂停: {execution_id}")
            return True

        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """恢复执行"""
        if execution_id in self.active_executions:
            state = self.active_executions[execution_id]
            if state.status == WorkflowStatus.PAUSED:
                state.status = WorkflowStatus.RUNNING

                logger.info(f"工作流执行已恢复: {execution_id}")
                return True

        return False

    def get_workflow_definitions(self) -> List[WorkflowDefinition]:
        """获取所有工作流定义"""
        return list(self.workflows.values())

    def get_active_executions(self) -> List[WorkflowState]:
        """获取活跃的执行"""
        return [state for state in self.active_executions.values()
                if state.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]]

    async def cleanup_completed_executions(self, max_age_hours: int = 24) -> int:
        """清理已完成的执行"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        completed_ids = []
        for execution_id, state in self.active_executions.items():
            if (state.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                state.end_time and state.end_time < cutoff_time):
                completed_ids.append(execution_id)

        for execution_id in completed_ids:
            del self.active_executions[execution_id]

        logger.info(f"清理了 {len(completed_ids)} 个已完成的工作流执行")
        return len(completed_ids)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_executions = len(self.active_executions)
        running_executions = len([s for s in self.active_executions.values() if s.status == WorkflowStatus.RUNNING])
        completed_executions = len([s for s in self.active_executions.values() if s.status == WorkflowStatus.COMPLETED])
        failed_executions = len([s for s in self.active_executions.values() if s.status == WorkflowStatus.FAILED])

        return {
            'total_workflows': len(self.workflows),
            'total_agents': len(self.agents),
            'total_tools': len(self.tools),
            'total_executions': total_executions,
            'running_executions': running_executions,
            'completed_executions': completed_executions,
            'failed_executions': failed_executions
        }


# 全局工作流引擎实例
workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """获取工作流引擎实例"""
    global workflow_engine

    if workflow_engine is None:
        workflow_engine = WorkflowEngine()

    return workflow_engine