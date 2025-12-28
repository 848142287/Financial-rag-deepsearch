"""
WebSocket实时通信服务
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型"""
    # 文档处理
    DOCUMENT_PROCESSING = "document_processing"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PROCESSED = "document_processed"
    DOCUMENT_FAILED = "document_failed"

    # 搜索处理
    SEARCH_PROCESSING = "search_processing"
    SEARCH_COMPLETED = "search_completed"
    SEARCH_FAILED = "search_failed"

    # 融合智能体
    FUSION_SEARCH = "fusion_search"
    FUSION_PROGRESS = "fusion_progress"

    # 迭代搜索
    ITERATIVE_SEARCH = "iterative_search"
    FEEDBACK_RECEIVED = "feedback_received"
    OPTIMIZATION_APPLIED = "optimization_applied"

    # 系统通知
    SYSTEM_NOTIFICATION = "system_notification"
    HEALTH_CHECK = "health_check"
    ERROR = "error"

    # 通用
    PROGRESS_UPDATE = "progress_update"
    TASK_STATUS = "task_status"
    CHAT_MESSAGE = "chat_message"


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        # 活跃连接 {user_id: {connection_id: WebSocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # 连接元数据 {connection_id: metadata}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        # 订阅管理 {user_id: Set[topics]}
        self.subscriptions: Dict[str, Set[str]] = {}
        # 任务连接映射 {task_id: Set[user_ids}}
        self.task_subscribers: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str, connection_id: Optional[str] = None):
        """建立WebSocket连接"""
        await websocket.accept()

        if not connection_id:
            connection_id = str(uuid.uuid4())

        # 添加到活跃连接
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}

        self.active_connections[user_id][connection_id] = websocket

        # 保存连接元数据
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "last_ping": datetime.utcnow().isoformat()
        }

        # 初始化订阅
        if user_id not in self.subscriptions:
            self.subscriptions[user_id] = set()

        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")

        # 发送连接确认
        await self.send_to_connection(connection_id, {
            "type": MessageType.SYSTEM_NOTIFICATION.value,
            "message": "连接成功",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        return connection_id

    def disconnect(self, user_id: str, connection_id: str):
        """断开WebSocket连接"""
        if user_id in self.active_connections:
            if connection_id in self.active_connections[user_id]:
                del self.active_connections[user_id][connection_id]

            # 如果用户没有其他连接，清理用户数据
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                if user_id in self.subscriptions:
                    del self.subscriptions[user_id]

        # 清理元数据
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]

        # 清理任务订阅
        for task_id, subscribers in self.task_subscribers.items():
            if user_id in subscribers:
                subscribers.remove(user_id)

        logger.info(f"WebSocket disconnected: {connection_id} for user {user_id}")

    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """发送消息到特定连接"""
        # 查找连接
        for user_connections in self.active_connections.values():
            if connection_id in user_connections:
                websocket = user_connections[connection_id]
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                    return True
                except Exception as e:
                    logger.error(f"Failed to send message to connection {connection_id}: {e}")
                    return False
        return False

    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """发送消息到用户的所有连接"""
        if user_id not in self.active_connections:
            return False

        disconnected = []
        for connection_id, websocket in self.active_connections[user_id].items():
            try:
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}, connection {connection_id}: {e}")
                disconnected.append(connection_id)

        # 清理断开的连接
        for connection_id in disconnected:
            self.disconnect(user_id, connection_id)

        return len(self.active_connections.get(user_id, {})) > 0

    async def broadcast(self, message: Dict[str, Any], exclude_users: Optional[List[str]] = None):
        """广播消息到所有连接"""
        exclude_set = set(exclude_users or [])
        sent_count = 0

        for user_id, connections in self.active_connections.items():
            if user_id not in exclude_set:
                if await self.send_to_user(user_id, message):
                    sent_count += 1

        return sent_count

    async def send_to_task_subscribers(self, task_id: str, message: Dict[str, Any]):
        """发送消息到任务订阅者"""
        if task_id not in self.task_subscribers:
            return 0

        sent_count = 0
        for user_id in self.task_subscribers[task_id]:
            if await self.send_to_user(user_id, message):
                sent_count += 1

        return sent_count

    def subscribe_to_task(self, user_id: str, task_id: str):
        """订阅任务更新"""
        if task_id not in self.task_subscribers:
            self.task_subscribers[task_id] = set()
        self.task_subscribers[task_id].add(user_id)

        logger.info(f"User {user_id} subscribed to task {task_id}")

    def unsubscribe_from_task(self, user_id: str, task_id: str):
        """取消订阅任务更新"""
        if task_id in self.task_subscribers:
            self.task_subscribers[task_id].discard(user_id)
            if not self.task_subscribers[task_id]:
                del self.task_subscribers[task_id]

        logger.info(f"User {user_id} unsubscribed from task {task_id}")

    def subscribe_to_topic(self, user_id: str, topic: str):
        """订阅主题"""
        if user_id not in self.subscriptions:
            self.subscriptions[user_id] = set()
        self.subscriptions[user_id].add(topic)

    def unsubscribe_from_topic(self, user_id: str, topic: str):
        """取消订阅主题"""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].discard(topic)

    async def send_to_topic(self, topic: str, message: Dict[str, Any]):
        """发送消息到主题订阅者"""
        sent_count = 0
        for user_id, topics in self.subscriptions.items():
            if topic in topics:
                if await self.send_to_user(user_id, message):
                    sent_count += 1
        return sent_count

    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        return {
            "total_connections": total_connections,
            "total_users": len(self.active_connections),
            "active_tasks": len(self.task_subscribers),
            "topics": {
                topic: len(users)
                for topic, users in self._get_topic_subscribers().items()
            }
        }

    def _get_topic_subscribers(self) -> Dict[str, Set[str]]:
        """获取所有主题的订阅者"""
        topic_subscribers = {}
        for user_id, topics in self.subscriptions.items():
            for topic in topics:
                if topic not in topic_subscribers:
                    topic_subscribers[topic] = set()
                topic_subscribers[topic].add(user_id)
        return topic_subscribers

    async def ping_all_connections(self):
        """向所有连接发送心跳"""
        ping_message = {
            "type": MessageType.HEALTH_CHECK.value,
            "message": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }

        return await self.broadcast(ping_message)


class WebSocketService:
    """WebSocket服务"""

    def __init__(self):
        self.manager = ConnectionManager()
        self._running = False
        self._cleanup_task = None

    async def start(self):
        """启动WebSocket服务"""
        self._running = True
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("WebSocket service started")

    async def stop(self):
        """停止WebSocket服务"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("WebSocket service stopped")

    def get_manager(self) -> ConnectionManager:
        """获取连接管理器"""
        return self.manager

    async def send_message(self, message_type: str, data: Dict[str, Any],
                          user_id: Optional[str] = None, task_id: Optional[str] = None,
                          topic: Optional[str] = None):
        """发送消息"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }

        sent_count = 0

        # 发送给特定用户
        if user_id:
            if await self.manager.send_to_user(user_id, message):
                sent_count += 1

        # 发送给任务订阅者
        elif task_id:
            sent_count = await self.manager.send_to_task_subscribers(task_id, message)

        # 发送给主题订阅者
        elif topic:
            sent_count = await self.manager.send_to_topic(topic, message)

        return sent_count

    async def send_progress_update(self, task_id: str, progress: int, status: str, message: str = ""):
        """发送进度更新"""
        return await self.send_message(
            MessageType.PROGRESS_UPDATE.value,
            {
                "task_id": task_id,
                "progress": progress,
                "status": status,
                "message": message
            },
            task_id=task_id
        )

    async def send_error(self, user_id: str, error: str, task_id: Optional[str] = None):
        """发送错误消息"""
        data = {
            "error": error,
            "task_id": task_id
        }

        return await self.send_message(
            MessageType.ERROR.value,
            data,
            user_id=user_id
        )

    async def send_system_notification(self, message: str, level: str = "info"):
        """发送系统通知"""
        return await self.manager.send_message(
            MessageType.SYSTEM_NOTIFICATION.value,
            {
                "message": message,
                "level": level
            }
        )

    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                # 每30秒清理一次
                await asyncio.sleep(30)
                await self._cleanup_dead_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_dead_connections(self):
        """清理死连接"""
        # 这里可以实现死连接检测逻辑
        # 例如：检查最后心跳时间
        pass


# 全局WebSocket服务实例
websocket_service = WebSocketService()

#便捷函数
async def send_document_progress(document_id: str, user_id: str, progress: int, status: str, message: str = ""):
    """发送文档处理进度"""
    await websocket_service.send_message(
        MessageType.DOCUMENT_PROCESSING.value,
        {
            "document_id": document_id,
            "progress": progress,
            "status": status,
            "message": message
        },
        user_id=user_id
    )


async def send_search_progress(task_id: str, user_id: str, progress: int, status: str, message: str = ""):
    """发送搜索进度"""
    await websocket_service.send_message(
        MessageType.SEARCH_PROCESSING.value,
        {
            "task_id": task_id,
            "progress": progress,
            "status": status,
            "message": message
        },
        user_id=user_id
    )


async def send_task_complete(task_id: str, result: Dict[str, Any]):
    """发送任务完成通知"""
    # 通过Redis发布消息（用于跨进程通知）
    redis_client.publish(
        "task_complete",
        json.dumps({
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    )

    # 通过WebSocket发送
    await websocket_service.send_message(
        MessageType.TASK_STATUS.value,
        {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
    )


async def notify_document_processed(document_id: str, user_id: str):
    """通知文档处理完成"""
    await websocket_service.send_message(
        MessageType.DOCUMENT_PROCESSED.value,
        {
            "document_id": document_id
        },
        user_id=user_id
    )


async def notify_search_completed(search_id: str, user_id: str, results: List[Dict[str, Any]]):
    """通知搜索完成"""
    await websocket_service.send_message(
        MessageType.SEARCH_COMPLETED.value,
        {
            "search_id": search_id,
            "results": results
        },
        user_id=user_id
    )