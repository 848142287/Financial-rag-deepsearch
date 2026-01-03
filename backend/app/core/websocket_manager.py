"""
WebSocket连接管理器
处理实时通信和进度推送
"""

import asyncio
import json
from app.core.structured_logging import get_structured_logger
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
import uuid
from enum import Enum

from fastapi import WebSocket
from app.core.redis_client import redis_client

logger = get_structured_logger(__name__)


class MessageType(Enum):
    """消息类型枚举"""
    # 任务相关
    TASK_STARTED = 'task_started'
    TASK_PROGRESS = 'task_progress'
    TASK_COMPLETED = 'task_completed'
    TASK_FAILED = 'task_failed'
    TASK_CANCELLED = 'task_cancelled'

    # 文档处理相关
    DOCUMENT_UPLOADED = 'document_uploaded'
    DOCUMENT_PROCESSING = 'document_processing'
    DOCUMENT_PROCESSED = 'document_processed'
    DOCUMENT_FAILED = 'document_failed'

    # 检索相关
    QUERY_STARTED = 'query_started'
    QUERY_PROGRESS = 'query_progress'
    QUERY_COMPLETED = 'query_completed'

    # 系统相关
    SYSTEM_NOTIFICATION = 'system_notification'
    HEALTH_STATUS = 'health_status'
    METRICS_UPDATE = 'metrics_update'

    # 心跳
    HEARTBEAT = 'heartbeat'
    PONG = 'pong'


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        # 活跃连接
        self.active_connections: Dict[str, WebSocket] = {}

        # 用户连接映射
        self.user_connections: Dict[int, Set[str]] = {}

        # 任务订阅映射
        self.task_subscribers: Dict[str, Set[str]] = {}

        # 心跳管理
        self.last_heartbeat: Dict[str, datetime] = {}

        # 心跳检查任务
        self.heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket, user_id: int, connection_id: Optional[str] = None):
        """建立WebSocket连接"""
        if connection_id is None:
            connection_id = str(uuid.uuid4())

        await websocket.accept()

        # 存储连接
        self.active_connections[connection_id] = websocket

        # 更新用户连接映射
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)

        # 记录心跳时间
        self.last_heartbeat[connection_id] = datetime.now()

        logger.info(f"WebSocket连接建立: 用户={user_id}, 连接ID={connection_id}")

        # 发送连接确认消息
        await self.send_personal_message({
            'type': MessageType.SYSTEM_NOTIFICATION.value,
            'data': {
                'message': '连接已建立',
                'connection_id': connection_id,
                'timestamp': datetime.now().isoformat()
            }
        }, connection_id)

        # 启动心跳检查
        if not self.heartbeat_task or self.heartbeat_task.done():
            self.heartbeat_task = asyncio.create_task(self._heartbeat_check())

        return connection_id

    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id in self.active_connections:
            # 移除连接
            del self.active_connections[connection_id]

            # 从用户连接映射中移除
            for user_id, connections in self.user_connections.items():
                if connection_id in connections:
                    connections.remove(connection_id)
                    if not connections:
                        del self.user_connections[user_id]
                    break

            # 从任务订阅中移除
            for task_id, subscribers in self.task_subscribers.items():
                if connection_id in subscribers:
                    subscribers.remove(connection_id)
                    if not subscribers:
                        del self.task_subscribers[task_id]

            # 清理心跳记录
            if connection_id in self.last_heartbeat:
                del self.last_heartbeat[connection_id]

            logger.info(f"WebSocket连接断开: 连接ID={connection_id}")

    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """发送个人消息"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"发送个人消息失败: {e}")
                await self.disconnect(connection_id)

    async def send_user_message(self, message: Dict[str, Any], user_id: int):
        """发送用户消息"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_personal_message(message, connection_id)

    async def subscribe_task(self, task_id: str, connection_id: str):
        """订阅任务进度"""
        if task_id not in self.task_subscribers:
            self.task_subscribers[task_id] = set()
        self.task_subscribers[task_id].add(connection_id)

        logger.info(f"连接 {connection_id} 订阅任务 {task_id}")

    async def unsubscribe_task(self, task_id: str, connection_id: str):
        """取消订阅任务进度"""
        if task_id in self.task_subscribers:
            self.task_subscribers[task_id].discard(connection_id)
            if not self.task_subscribers[task_id]:
                del self.task_subscribers[task_id]

        logger.info(f"连接 {connection_id} 取消订阅任务 {task_id}")

    async def broadcast_task_update(self, task_id: str, message_type: MessageType, data: Dict[str, Any]):
        """广播任务更新"""
        message = {
            'type': message_type.value,
            'task_id': task_id,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }

        # 发送给订阅该任务的连接
        if task_id in self.task_subscribers:
            for connection_id in self.task_subscribers[task_id]:
                await this.send_personal_message(message, connection_id)

        # 同时发布到Redis，用于多实例同步
        try:
            redis_client.publish('task_updates', json.dumps({
                'task_id': task_id,
                'message_type': message_type.value,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }))
        except Exception as e:
            logger.error(f"发布任务更新到Redis失败: {e}")

    async def send_system_notification(self, message: str, level: str = 'info', target_users: List[int] = None):
        """发送系统通知"""
        notification = {
            'type': MessageType.SYSTEM_NOTIFICATION.value,
            'data': {
                'message': message,
                'level': level,  # info, warning, error
                'timestamp': datetime.now().isoformat()
            }
        }

        if target_users:
            # 发送给特定用户
            for user_id in target_users:
                await self.send_user_message(notification, user_id)
        else:
            # 广播给所有用户
            for connection_id in self.active_connections:
                await self.send_personal_message(notification, connection_id)

    async def handle_heartbeat(self, connection_id: str):
        """处理心跳"""
        self.last_heartbeat[connection_id] = datetime.now()

        # 发送PONG响应
        await this.send_personal_message({
            'type': MessageType.PONG.value,
            'data': {
                'timestamp': datetime.now().isoformat()
            }
        }, connection_id)

    async def _heartbeat_check(self):
        """心跳检查任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                current_time = datetime.now()
                timeout_connections = []

                for connection_id, last_heartbeat_time in this.last_heartbeat.items():
                    # 超过2分钟没有心跳，断开连接
                    if (current_time - last_heartbeat_time).seconds > 120:
                        timeout_connections.append(connection_id)

                # 断开超时连接
                for connection_id in timeout_connections:
                    logger.warning(f"连接 {connection_id} 心跳超时，断开连接")
                    await this.disconnect(connection_id)

            except Exception as e:
                logger.error(f"心跳检查失败: {e}")

    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            'total_connections': len(this.active_connections),
            'total_users': len(this.user_connections),
            'active_tasks': len(this.task_subscribers),
            'connections_per_user': {
                user_id: len(connections)
                for user_id, connections in this.user_connections.items()
            }
        }

    async def force_disconnect_user(self, user_id: int):
        """强制断开用户的所有连接"""
        if user_id in this.user_connections:
            connection_ids = list(this.user_connections[user_id])
            for connection_id in connection_ids:
                await this.disconnect(connection_id)


# 全局连接管理器实例
connection_manager = ConnectionManager()


class RedisMessageHandler:
    """Redis消息处理器"""

    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """启动消息监听"""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._listen_to_redis())

        logger.info("Redis消息监听器已启动")

    async def stop(self):
        """停止消息监听"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info("Redis消息监听器已停止")

    async def _listen_to_redis(self):
        """监听Redis消息"""
        try:
            # 创建Redis发布订阅连接
            pubsub = redis_client.pubsub()
            await pubsub.subscribe('task_updates', 'system_notifications')

            while self.running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        await self._handle_redis_message(message['data'])
                except Exception as e:
                    logger.error(f"处理Redis消息失败: {e}")
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Redis消息监听失败: {e}")

    async def _handle_redis_message(self, data: bytes):
        """处理Redis消息"""
        try:
            message = json.loads(data.decode('utf-8'))

            if 'task_id' in message:
                # 任务更新消息
                task_id = message['task_id']
                message_type = MessageType(message['message_type'])
                message_data = message['data']

                await this.manager.broadcast_task_update(task_id, message_type, message_data)

            elif 'notification' in message:
                # 系统通知消息
                notification = message['notification']
                level = notification.get('level', 'info')
                target_users = notification.get('target_users')

                await this.manager.send_system_notification(
                    notification['message'],
                    level,
                    target_users
                )

        except Exception as e:
            logger.error(f"解析Redis消息失败: {e}")


# Redis消息处理器实例
redis_message_handler = RedisMessageHandler(connection_manager)


# 启动WebSocket服务
async def start_websocket_service():
    """启动WebSocket相关服务"""
    # 启动Redis消息监听
    await redis_message_handler.start()
    logger.info("WebSocket服务启动完成")


# 停止WebSocket服务
async def stop_websocket_service():
    """停止WebSocket相关服务"""
    # 停止Redis消息监听
    await redis_message_handler.stop()
    logger.info("WebSocket服务停止完成")


# 获取WebSocket管理器
def get_connection_manager() -> ConnectionManager:
    """获取连接管理器实例"""
    return connection_manager