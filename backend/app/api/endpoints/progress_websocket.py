"""
WebSocket进度推送路由
提供实时进度更新的WebSocket端点
"""

from app.core.structured_logging import get_structured_logger
import json
from typing import Dict, Set

logger = get_structured_logger(__name__)

router = APIRouter(prefix="/api/progress", tags=["Progress"])

class ProgressWebSocketHandler:
    """WebSocket进度处理器"""

    def __init__(self):
        # 活跃连接: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

        # 客户端订阅: client_id -> Set[task_id]
        self.client_subscriptions: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """连接WebSocket"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_subscriptions[client_id] = set()

        logger.info(f"WebSocket connected: {client_id}")

        # 发送连接确认
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "WebSocket连接成功"
        })

    def disconnect(self, client_id: str):
        """断开WebSocket"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        if client_id in self.client_subscriptions:
            # 从进度管理器中注销连接
            progress_manager = get_progress_manager()
            for task_id in self.client_subscriptions[client_id]:
                progress_manager.unregister_connection(
                    task_id,
                    self.active_connections.get(client_id)
                )

            del self.client_subscriptions[client_id]

        logger.info(f"WebSocket disconnected: {client_id}")

    async def subscribe_task(self, client_id: str, task_id: str):
        """订阅任务进度"""
        if client_id not in self.client_subscriptions:
            logger.warning(f"Client not connected: {client_id}")
            return

        if client_id not in self.active_connections:
            logger.warning(f"WebSocket not found: {client_id}")
            return

        # 添加订阅
        self.client_subscriptions[client_id].add(task_id)

        # 注册到进度管理器
        progress_manager = get_progress_manager()
        websocket = self.active_connections[client_id]
        progress_manager.register_connection(task_id, websocket)

        # 发送当前任务状态
        task = progress_manager.get_task(task_id)
        if task:
            await websocket.send_json({
                "type": "task_status",
                "task_id": task_id,
                "data": task.to_dict()
            })

        logger.info(f"Client {client_id} subscribed to task {task_id}")

    async def unsubscribe_task(self, client_id: str, task_id: str):
        """取消订阅任务"""
        if client_id not in self.client_subscriptions:
            return

        if task_id in self.client_subscriptions[client_id]:
            self.client_subscriptions[client_id].discard(task_id)

            # 从进度管理器注销
            progress_manager = get_progress_manager()
            websocket = self.active_connections.get(client_id)
            if websocket:
                progress_manager.unregister_connection(task_id, websocket)

            logger.info(f"Client {client_id} unsubscribed from task {task_id}")

    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self.active_connections)

# 全局WebSocket处理器
ws_handler = ProgressWebSocketHandler()

@router.websocket("/ws/{client_id}")
async def websocket_progress_endpoint(
    websocket: WebSocket,
    client_id: str
):
    """
    WebSocket进度推送端点

    Args:
        websocket: WebSocket连接
        client_id: 客户端ID

    消息格式:
    客户端 -> 服务器:
    {
        "action": "subscribe|unsubscribe|ping",
        "task_id": "task_id"  // for subscribe/unsubscribe
    }

    服务器 -> 客户端:
    {
        "type": "progress|task_status|connected|error",
        "task_id": "task_id",
        "stage": "stage_name",
        "progress": 0-100,
        "message": "message",
        "data": {...}
    }
    """
    await ws_handler.connect(websocket, client_id)

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)

            action = message.get("action")

            if action == "subscribe":
                # 订阅任务
                task_id = message.get("task_id")
                if task_id:
                    await ws_handler.subscribe_task(client_id, task_id)

            elif action == "unsubscribe":
                # 取消订阅
                task_id = message.get("task_id")
                if task_id:
                    await ws_handler.unsubscribe_task(client_id, task_id)

            elif action == "ping":
                # 心跳
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })

    except WebSocketDisconnect:
        ws_handler.disconnect(client_id)
        logger.info(f"WebSocket disconnected normally: {client_id}")

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        ws_handler.disconnect(client_id)

@router.get("/tasks/{task_id}")
async def get_task_progress(task_id: str):
    """
    获取任务进度(HTTP轮询方式)

    Args:
        task_id: 任务ID

    Returns:
        任务进度信息
    """
    progress_manager = get_progress_manager()
    task = progress_manager.get_task(task_id)

    if not task:
        return {
            "success": False,
            "error": "Task not found"
        }

    return {
        "success": True,
        "data": task.to_dict(),
        "estimated_time_remaining": progress_manager.get_estimated_time_remaining(task_id)
    }

@router.get("/tasks")
async def get_all_tasks():
    """获取所有任务"""
    progress_manager = get_progress_manager()
    tasks = progress_manager.get_all_tasks()

    return {
        "success": True,
        "data": [task.to_dict() for task in tasks],
        "count": len(tasks)
    }

@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    progress_manager = get_progress_manager()
    progress_manager.cancel_task(task_id)

    return {
        "success": True,
        "message": f"Task {task_id} cancelled"
    }

@router.get("/stats")
async def get_progress_stats():
    """获取进度统计"""
    return {
        "active_connections": ws_handler.get_connection_count(),
        "total_tasks": len(get_progress_manager().get_all_tasks())
    }
