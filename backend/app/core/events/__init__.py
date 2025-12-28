"""
事件驱动架构模块
实现系统的事件总线、事件处理器和状态管理
"""

from .event_bus import EventBus, Event
from .event_handlers import DocumentEventHandler, SearchEventHandler
from .state_manager import StateManager, SystemState
from .event_types import EventType, SystemEvent, DocumentEvent, SearchEvent

__all__ = [
    'EventBus',
    'Event',
    'EventType',
    'SystemEvent',
    'DocumentEvent',
    'SearchEvent',
    'DocumentEventHandler',
    'SearchEventHandler',
    'StateManager',
    'SystemState'
]