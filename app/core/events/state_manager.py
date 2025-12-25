"""
状态管理器
实现集中式状态管理，维护系统全局状态
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from threading import Lock

from .event_types import Event, EventType

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """组件状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class RetrievalLevel(Enum):
    """检索级别"""
    FAST = "fast"
    ENHANCED = "enhanced"
    DEEP_SEARCH = "deep_search"


class SystemLoadLevel(Enum):
    """系统负载级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComponentState:
    """组件状态"""
    component_id: str
    component_type: str
    status: ComponentStatus
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """用户会话状态"""
    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    query_count: int = 0
    active_searches: Set[str] = field(default_factory=set)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchMetrics:
    """搜索指标"""
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    deep_search_usage: int = 0
    multi_strategy_usage: int = 0


@dataclass
class SystemHealth:
    """系统健康状态"""
    overall_status: str = "healthy"
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_connections: int = 0
    queue_size: int = 0
    error_rate: float = 0.0
    uptime: float = 0.0


@dataclass
class SystemState:
    """系统全局状态"""
    # 基础信息
    version: str = "1.0.0"
    start_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # 组件状态
    components: Dict[str, ComponentState] = field(default_factory=dict)

    # 用户会话
    active_sessions: Dict[str, UserSession] = field(default_factory=dict)

    # 搜索指标
    search_metrics: SearchMetrics = field(default_factory=SearchMetrics)

    # 系统健康
    health: SystemHealth = field(default_factory=SystemHealth)

    # 配置状态
    current_config: Dict[str, Any] = field(default_factory=dict)

    # 负载管理
    current_load: SystemLoadLevel = SystemLoadLevel.LOW
    max_concurrent_searches: int = 100
    current_concurrent_searches: int = 0

    # 缓存状态
    cache_stats: Dict[str, Any] = field(default_factory=dict)

    # 事件统计
    event_stats: Dict[str, int] = field(default_factory=lambda: {
        "events_processed": 0,
        "events_failed": 0,
        "handlers_active": 0
    })


class StateManager:
    """状态管理器

    负责维护系统全局状态，提供状态查询和更新接口
    """

    def __init__(self):
        self._state = SystemState()
        self._lock = Lock()
        self._subscribers: Dict[str, Set[callable]] = {}
        self._logger = logging.getLogger(__name__)

    def get_state(self) -> SystemState:
        """获取当前系统状态"""
        with self._lock:
            # 更新最后更新时间
            self._state.last_updated = datetime.now()
            return self._state

    def get_component_state(self, component_id: str) -> Optional[ComponentState]:
        """获取组件状态"""
        with self._lock:
            return self._state.components.get(component_id)

    def update_component_state(self, component_id: str, component_type: str,
                             status: ComponentStatus, **kwargs) -> None:
        """更新组件状态"""
        with self._lock:
            component_state = self._state.components.get(component_id)
            if component_state:
                component_state.status = status
                component_state.last_updated = datetime.now()
                component_state.metadata.update(kwargs)
                if "error_message" in kwargs:
                    component_state.error_message = kwargs["error_message"]
            else:
                self._state.components[component_id] = ComponentState(
                    component_id=component_id,
                    component_type=component_type,
                    status=status,
                    last_updated=datetime.now(),
                    metadata=kwargs
                )

        # 通知订阅者
        self._notify_subscribers("component_updated", component_id)

    def get_user_session(self, user_id: str) -> Optional[UserSession]:
        """获取用户会话"""
        with self._lock:
            for session in self._state.active_sessions.values():
                if session.user_id == user_id:
                    return session
        return None

    def create_user_session(self, user_id: str, session_id: str, **kwargs) -> UserSession:
        """创建用户会话"""
        with self._lock:
            session = UserSession(
                user_id=user_id,
                session_id=session_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                preferences=kwargs
            )
            self._state.active_sessions[session_id] = session
            return session

    def update_user_session(self, session_id: str, **kwargs) -> None:
        """更新用户会话"""
        with self._lock:
            session = self._state.active_sessions.get(session_id)
            if session:
                session.last_activity = datetime.now()
                if "query_count" in kwargs:
                    session.query_count = kwargs["query_count"]
                if "active_searches" in kwargs:
                    session.active_searches.update(kwargs["active_searches"])

    def remove_user_session(self, session_id: str) -> None:
        """移除用户会话"""
        with self._lock:
            self._state.active_sessions.pop(session_id, None)

    def update_search_metrics(self, **kwargs) -> None:
        """更新搜索指标"""
        with self._lock:
            metrics = self._state.search_metrics

            if "total_searches" in kwargs:
                metrics.total_searches += kwargs["total_searches"]
            if "successful_searches" in kwargs:
                metrics.successful_searches += kwargs["successful_searches"]
            if "failed_searches" in kwargs:
                metrics.failed_searches += kwargs["failed_searches"]
            if "avg_response_time" in kwargs:
                # 更新平均响应时间
                current_avg = metrics.avg_response_time
                new_avg = (current_avg + kwargs["avg_response_time"]) / 2
                metrics.avg_response_time = new_avg
            if "cache_hit_rate" in kwargs:
                metrics.cache_hit_rate = kwargs["cache_hit_rate"]
            if "deep_search_usage" in kwargs:
                metrics.deep_search_usage += kwargs["deep_search_usage"]
            if "multi_strategy_usage" in kwargs:
                metrics.multi_strategy_usage += kwargs["multi_strategy_usage"]

    def get_search_metrics(self) -> SearchMetrics:
        """获取搜索指标"""
        with self._lock:
            return self._state.search_metrics

    def update_system_health(self, **kwargs) -> None:
        """更新系统健康状态"""
        with self._lock:
            health = self._state.health

            for key, value in kwargs.items():
                if hasattr(health, key):
                    setattr(health, key, value)

            # 计算整体状态
            if health.error_rate > 0.1 or health.cpu_usage > 0.9:
                health.overall_status = "critical"
            elif health.error_rate > 0.05 or health.cpu_usage > 0.7:
                health.overall_status = "warning"
            else:
                health.overall_status = "healthy"

    def get_system_health(self) -> SystemHealth:
        """获取系统健康状态"""
        with self._lock:
            return self._state.health

    def update_load_level(self, load_level: SystemLoadLevel) -> None:
        """更新系统负载级别"""
        with self._lock:
            self._state.current_load = load_level

    def increment_concurrent_searches(self) -> bool:
        """增加并发搜索数量"""
        with self._lock:
            if self._state.current_concurrent_searches < self._state.max_concurrent_searches:
                self._state.current_concurrent_searches += 1
                return True
            return False

    def decrement_concurrent_searches(self) -> None:
        """减少并发搜索数量"""
        with self._lock:
            if self._state.current_concurrent_searches > 0:
                self._state.current_concurrent_searches -= 1

    def update_cache_stats(self, **kwargs) -> None:
        """更新缓存统计"""
        with self._lock:
            self._state.cache_stats.update(kwargs)

    async def handle_event(self, event: Event) -> None:
        """处理事件"""
        try:
            # 更新事件统计
            with self._lock:
                self._state.event_stats["events_processed"] += 1

            # 根据事件类型更新状态
            if event.event_type == EventType.SYSTEM_STARTED.value:
                self._handle_system_started(event)
            elif event.event_type == EventType.DOCUMENT_UPLOADED.value:
                self._handle_document_uploaded(event)
            elif event.event_type == EventType.SEARCH_QUERY_RECEIVED.value:
                self._handle_search_query_received(event)
            elif event.event_type == EventType.SEARCH_COMPLETED.value:
                self._handle_search_completed(event)
            elif event.event_type == EventType.ERROR_OCCURRED.value:
                self._handle_error_occurred(event)

        except Exception as e:
            self._logger.error(f"处理事件失败: {e}")
            with self._lock:
                self._state.event_stats["events_failed"] += 1

    def _handle_system_started(self, event: Event) -> None:
        """处理系统启动事件"""
        component = event.data.get("component", "unknown")
        self.update_component_state(
            component_id=component,
            component_type="system",
            status=ComponentStatus.READY
        )

    def _handle_document_uploaded(self, event: Event) -> None:
        """处理文档上传事件"""
        self.update_component_state(
            component_id="document_processor",
            component_type="service",
            status=ComponentStatus.BUSY
        )

    def _handle_search_query_received(self, event: Event) -> None:
        """处理搜索查询接收事件"""
        # 增加并发搜索数量
        if not self.increment_concurrent_searches():
            self._logger.warning("并发搜索数量已达上限")

        # 更新用户会话
        if event.user_id:
            session = self.get_user_session(event.user_id)
            if not session and event.session_id:
                session = self.create_user_session(event.user_id, event.session_id)

            if session:
                self.update_user_session(
                    session.session_id,
                    query_count=session.query_count + 1
                )

    def _handle_search_completed(self, event: Event) -> None:
        """处理搜索完成事件"""
        # 减少并发搜索数量
        self.decrement_concurrent_searches()

        # 更新搜索指标
        result_count = event.data.get("result_count", 0)
        execution_time = event.data.get("execution_time", 0)
        cache_hit = event.data.get("cache_hit", False)

        self.update_search_metrics(
            total_searches=1,
            successful_searches=1 if result_count > 0 else 0,
            avg_response_time=execution_time,
            cache_hit_rate=self._state.search_metrics.cache_hit_rate if cache_hit else 0.5
        )

    def _handle_error_occurred(self, event: Event) -> None:
        """处理错误事件"""
        error_component = event.data.get("component", "unknown")

        self.update_component_state(
            component_id=error_component,
            component_type="system",
            status=ComponentStatus.ERROR,
            error_message=event.data.get("error_message")
        )

        # 更新系统健康状态
        current_health = self.get_system_health()
        new_error_rate = current_health.error_rate + 0.01
        self.update_system_health(error_rate=new_error_rate)

    def subscribe(self, event_type: str, callback: callable) -> None:
        """订阅状态变化"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)

    def unsubscribe(self, event_type: str, callback: callable) -> None:
        """取消订阅状态变化"""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)

    def _notify_subscribers(self, event_type: str, *args, **kwargs) -> None:
        """通知订阅者"""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    self._logger.error(f"通知订阅者失败: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        with self._lock:
            state = self._state

            return {
                "system_info": {
                    "version": state.version,
                    "uptime": (datetime.now() - state.start_time).total_seconds(),
                    "last_updated": state.last_updated.isoformat()
                },
                "components": {
                    component_id: {
                        "type": component.component_type,
                        "status": component.status.value,
                        "last_updated": component.last_updated.isoformat(),
                        "error_message": component.error_message
                    }
                    for component_id, component in state.components.items()
                },
                "sessions": {
                    "active_sessions": len(state.active_sessions),
                    "total_queries": sum(s.query_count for s in state.active_sessions.values())
                },
                "search_metrics": {
                    "total_searches": state.search_metrics.total_searches,
                    "success_rate": (state.search_metrics.successful_searches /
                                  max(state.search_metrics.total_searches, 1)),
                    "avg_response_time": state.search_metrics.avg_response_time,
                    "cache_hit_rate": state.search_metrics.cache_hit_rate
                },
                "system_health": {
                    "overall_status": state.health.overall_status,
                    "cpu_usage": state.health.cpu_usage,
                    "memory_usage": state.health.memory_usage,
                    "active_connections": state.health.active_connections,
                    "error_rate": state.health.error_rate
                },
                "load_management": {
                    "current_load": state.current_load.value,
                    "concurrent_searches": state.current_concurrent_searches,
                    "max_concurrent_searches": state.max_concurrent_searches
                },
                "events": state.event_stats
            }

    def export_state(self) -> Dict[str, Any]:
        """导出当前状态"""
        with self._lock:
            return {
                "state": self._state,
                "timestamp": datetime.now().isoformat()
            }

    def import_state(self, state_data: Dict[str, Any]) -> None:
        """导入状态"""
        with self._lock:
            if "state" in state_data:
                self._state = state_data["state"]
            self._logger.info("状态已导入")


# 全局状态管理器实例
state_manager = StateManager()