"""
事件总线
实现事件的发布、订阅和处理机制
"""

import asyncio
from app.core.structured_logging import get_structured_logger
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from datetime import datetime

from .event_types import Event, EventType
from .state_manager import StateManager

logger = get_structured_logger(__name__)


class EventHandler:
    """事件处理器基类"""

    def __init__(self, name: str):
        self.name = name
        self.subscribed_events: Set[str] = set()
        self.is_active = True

    async def handle(self, event: Event) -> None:
        """处理事件"""
        raise NotImplementedError("子类必须实现handle方法")

    def subscribe(self, event_type: str) -> None:
        """订阅事件类型"""
        self.subscribed_events.add(event_type)

    def unsubscribe(self, event_type: str) -> None:
        """取消订阅事件类型"""
        self.subscribed_events.discard(event_type)

    def can_handle(self, event: Event) -> bool:
        """检查是否能处理事件"""
        return event.event_type in self.subscribed_events and self.is_active


class EventBus:
    """事件总线

    负责事件的发布、订阅和路由
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._state_manager = Optional[StateManager]()
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        self._is_running = False
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "handlers_registered": 0,
            "errors": 0
        }
        self.logger = get_structured_logger(__name__)  # 使用统一日志框架

    def set_state_manager(self, state_manager: StateManager) -> None:
        """设置状态管理器"""
        self._state_manager = state_manager

    async def start(self) -> None:
        """启动事件总线"""
        if self._is_running:
            return

        self._is_running = True
        self.logger.info("事件总线已启动")

        # 发布系统启动事件
        await self.publish_event(
            EventType.SYSTEM_STARTED,
            source="event_bus",
            data={"startup_time": datetime.now().isoformat()}
        )

    async def stop(self) -> None:
        """停止事件总线"""
        if not self._is_running:
            return

        self._is_running = False
        self.logger.info("事件总线已停止")

        # 发布系统关闭事件
        await self.publish_event(
            EventType.SYSTEM_SHUTDOWN,
            source="event_bus",
            data={"shutdown_time": datetime.now().isoformat()}
        )

    def register_handler(self, handler: EventHandler, event_types: Optional[List[str]] = None) -> None:
        """注册事件处理器"""
        if event_types:
            for event_type in event_types:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(handler)
                handler.subscribe(event_type)
        else:
            self._global_handlers.append(handler)

        self._stats["handlers_registered"] += 1
        self.logger.info(f"注册事件处理器: {handler.name}")

    def unregister_handler(self, handler: EventHandler) -> None:
        """注销事件处理器"""
        # 从特定事件类型中移除
        for event_type, handlers in self._handlers.items():
            if handler in handlers:
                handlers.remove(handler)

        # 从全局处理器中移除
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

        self.logger.info(f"注销事件处理器: {handler.name}")

    async def publish_event(self, event_type: EventType, **kwargs) -> None:
        """发布事件"""
        event = Event(event_type=event_type.value, **kwargs)
        await self._publish(event)

    async def publish_event_object(self, event: Event) -> None:
        """发布事件对象"""
        await self._publish(event)

    async def _publish(self, event: Event) -> None:
        """内部发布方法"""
        try:
            # 记录事件历史
            self._add_to_history(event)
            self._stats["events_published"] += 1

            # 更新状态管理器
            if self._state_manager:
                await self._state_manager.handle_event(event)

            # 获取相关的处理器
            relevant_handlers = []

            # 全局处理器
            relevant_handlers.extend([h for h in self._global_handlers if h.can_handle(event)])

            # 特定事件类型处理器
            if event.event_type in self._handlers:
                relevant_handlers.extend([h for h in self._handlers[event.event_type] if h.can_handle(event)])

            # 并行处理事件
            if relevant_handlers:
                tasks = [self._handle_event_safe(handler, event) for handler in relevant_handlers]
                await asyncio.gather(*tasks, return_exceptions=True)

            self.logger.debug(f"事件 {event.event_type} 已发布给 {len(relevant_handlers)} 个处理器")

        except Exception as e:
            self.logger.error(f"发布事件失败: {e}")
            self._stats["errors"] += 1

    async def _handle_event_safe(self, handler: EventHandler, event: Event) -> None:
        """安全地处理事件"""
        try:
            await handler.handle(event)
            self._stats["events_processed"] += 1
        except Exception as e:
            self.logger.error(f"事件处理器 {handler.name} 处理事件失败: {e}")
            self._stats["errors"] += 1

    def _add_to_history(self, event: Event) -> None:
        """添加到事件历史"""
        self._event_history.append(event)

        # 限制历史大小
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]

    def get_event_history(self, limit: Optional[int] = None,
                         event_type: Optional[str] = None,
                         since: Optional[datetime] = None) -> List[Event]:
        """获取事件历史"""
        history = self._event_history

        # 按事件类型过滤
        if event_type:
            history = [e for e in history if e.event_type == event_type]

        # 按时间过滤
        if since:
            history = [e for e in history if e.timestamp >= since]

        # 限制数量
        if limit:
            history = history[-limit:]

        return history

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "handlers_by_type": {k: len(v) for k, v in self._handlers.items()},
            "global_handlers": len(self._global_handlers),
            "history_size": len(self._event_history),
            "is_running": self._is_running
        }

    async def clear_history(self) -> None:
        """清空事件历史"""
        self._event_history.clear()
        self.logger.info("事件历史已清空")

    def get_handler_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        info = {
            "global_handlers": [
                {
                    "name": handler.name,
                    "active": handler.is_active,
                    "subscribed_events": list(handler.subscribed_events)
                }
                for handler in self._global_handlers
            ],
            "specific_handlers": {}
        }

        for event_type, handlers in self._handlers.items():
            info["specific_handlers"][event_type] = [
                {
                    "name": handler.name,
                    "active": handler.is_active
                }
                for handler in handlers
            ]

        return info


class EventPublisher:
    """事件发布器

    提供便捷的事件发布接口
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = get_structured_logger(__name__)  # 使用统一日志框架

    async def system_started(self, component: str, **kwargs) -> None:
        """发布系统启动事件"""
        await self.event_bus.publish_event(
            EventType.SYSTEM_STARTED,
            source=component,
            data=kwargs
        )

    async def system_error(self, component: str, error: Exception, **kwargs) -> None:
        """发布系统错误事件"""
        await self.event_bus.publish_event(
            EventType.ERROR_OCCURRED,
            source=component,
            data={
                "error_message": str(error),
                "error_type": type(error).__name__,
                **kwargs
            }
        )

    async def document_uploaded(self, document_id: str, file_name: str, **kwargs) -> None:
        """发布文档上传事件"""
        await self.event_bus.publish_event(
            EventType.DOCUMENT_UPLOADED,
            source="document_service",
            document_id=document_id,
            file_name=file_name,
            data=kwargs
        )

    async def document_processing_completed(self, document_id: str, **kwargs) -> None:
        """发布文档处理完成事件"""
        await self.event_bus.publish_event(
            EventType.DOCUMENT_PARSING_COMPLETED,
            source="document_service",
            document_id=document_id,
            data=kwargs
        )

    async def search_query_received(self, query: str, user_id: str, **kwargs) -> None:
        """发布搜索查询接收事件"""
        await self.event_bus.publish_event(
            EventType.SEARCH_QUERY_RECEIVED,
            source="search_service",
            query=query,
            user_id=user_id,
            data=kwargs
        )

    async def search_completed(self, query: str, result_count: int, **kwargs) -> None:
        """发布搜索完成事件"""
        await self.event_bus.publish_event(
            EventType.SEARCH_COMPLETED,
            source="search_service",
            query=query,
            result_count=result_count,
            data=kwargs
        )

    async def deep_search_completed(self, query: str, iterations: int, **kwargs) -> None:
        """发布DeepSearch完成事件"""
        await self.event_bus.publish_event(
            EventType.DEEP_SEARCH_COMPLETED,
            source="deep_search_service",
            query=query,
            data={"iterations": iterations, **kwargs}
        )

    async def evaluation_completed(self, evaluation_type: str, score: float, **kwargs) -> None:
        """发布评估完成事件"""
        await self.event_bus.publish_event(
            EventType.EVALUATION_COMPLETED,
            source="evaluation_service",
            data={
                "evaluation_type": evaluation_type,
                "score": score,
                **kwargs
            }
        )


# 全局事件总线实例
event_bus = EventBus()
event_publisher = EventPublisher(event_bus)