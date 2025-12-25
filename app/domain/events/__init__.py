"""
Domain Events

Defines domain events that capture significant business events in the system.
Domain events enable loose coupling between different parts of the system
and support event-driven architecture patterns.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from uuid import uuid4
import json


@dataclass(frozen=True)
class DomainEvent:
    """Base domain event"""
    type: str
    aggregate_id: Optional[Union[int, str]]
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    version: int = 1
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "type": self.type,
            "aggregate_id": self.aggregate_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainEvent':
        """Create from dictionary"""
        return cls(
            type=data["type"],
            aggregate_id=data["aggregate_id"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_id=data["event_id"],
            version=data.get("version", 1),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            metadata=data.get("metadata", {})
        )


@dataclass(frozen=True)
class DocumentCreatedEvent(DomainEvent):
    """Event fired when a document is created"""
    type: str = "document_created"

    def __post_init__(self):
        if not self.data.get("title") and not self.data.get("file_name"):
            raise ValueError("Document must have title or file_name")


@dataclass(frozen=True)
class DocumentUpdatedEvent(DomainEvent):
    """Event fired when a document is updated"""
    type: str = "document_updated"


@dataclass(frozen=True)
class DocumentDeletedEvent(DomainEvent):
    """Event fired when a document is deleted"""
    type: str = "document_deleted"


@dataclass(frozen=True)
class DocumentProcessingStartedEvent(DomainEvent):
    """Event fired when document processing starts"""
    type: str = "document_processing_started"


@dataclass(frozen=True)
class DocumentProcessingCompletedEvent(DomainEvent):
    """Event fired when document processing completes"""
    type: str = "document_processing_completed"

    def __post_init__(self):
        required_fields = ["chunk_count", "embedding_count"]
        for field in required_fields:
            if field not in self.data:
                raise ValueError(f"DocumentProcessingCompletedEvent requires {field}")


@dataclass(frozen=True)
class DocumentProcessingFailedEvent(DomainEvent):
    """Event fired when document processing fails"""
    type: str = "document_processing_failed"

    def __post_init__(self):
        if not self.data.get("error_message"):
            raise ValueError("DocumentProcessingFailedEvent requires error_message")


@dataclass(frozen=True)
class QueryExecutedEvent(DomainEvent):
    """Event fired when a query is executed"""
    type: str = "query_executed"

    def __post_init__(self):
        required_fields = ["query_text", "result_count"]
        for field in required_fields:
            if field not in self.data:
                raise ValueError(f"QueryExecutedEvent requires {field}")


@dataclass(frozen=True)
class TaskCreatedEvent(DomainEvent):
    """Event fired when a task is created"""
    type: str = "task_created"

    def __post_init__(self):
        if not self.data.get("task_type"):
            raise ValueError("TaskCreatedEvent requires task_type")


@dataclass(frozen=True)
class TaskCompletedEvent(DomainEvent):
    """Event fired when a task is completed"""
    type: str = "task_completed"


@dataclass(frozen=True)
class TaskFailedEvent(DomainEvent):
    """Event fired when a task fails"""
    type: str = "task_failed"

    def __post_init__(self):
        if not self.data.get("error_message"):
            raise ValueError("TaskFailedEvent requires error_message")


@dataclass(frozen=True)
class UserCreatedEvent(DomainEvent):
    """Event fired when a user is created"""
    type: str = "user_created"

    def __post_init__(self):
        if not self.data.get("email"):
            raise ValueError("UserCreatedEvent requires email")


@dataclass(frozen=True)
class KnowledgeGraphUpdatedEvent(DomainEvent):
    """Event fired when knowledge graph is updated"""
    type: str = "knowledge_graph_updated"


@dataclass(frozen=True)
class EmbeddingGeneratedEvent(DomainEvent):
    """Event fired when embeddings are generated"""
    type: str = "embedding_generated"

    def __post_init__(self):
        required_fields = ["model_name", "vector_count"]
        for field in required_fields:
            if field not in self.data:
                raise ValueError(f"EmbeddingGeneratedEvent requires {field}")


@dataclass(frozen=True)
class CacheInvalidatedEvent(DomainEvent):
    """Event fired when cache is invalidated"""
    type: str = "cache_invalidated"


class EventStore:
    """Simple in-memory event store for domain events"""

    def __init__(self):
        self._events: List[DomainEvent] = []
        self._streams: Dict[str, List[DomainEvent]] = {}

    def append(self, event: DomainEvent) -> None:
        """Append event to store"""
        self._events.append(event)

        # Add to aggregate stream
        aggregate_id = str(event.aggregate_id) if event.aggregate_id else "global"
        if aggregate_id not in self._streams:
            self._streams[aggregate_id] = []
        self._streams[aggregate_id].append(event)

    def get_events(self, aggregate_id: Optional[Union[int, str]] = None,
                   event_type: Optional[str] = None,
                   limit: Optional[int] = None) -> List[DomainEvent]:
        """Get events from store"""
        events = self._events

        if aggregate_id:
            aggregate_id = str(aggregate_id)
            if aggregate_id in self._streams:
                events = self._streams[aggregate_id]
            else:
                return []

        if event_type:
            events = [e for e in events if e.type == event_type]

        if limit:
            events = events[-limit:]

        return events

    def get_event_stream(self, aggregate_id: Union[int, str],
                        from_version: Optional[int] = None) -> List[DomainEvent]:
        """Get event stream for aggregate"""
        aggregate_id = str(aggregate_id)
        if aggregate_id not in self._streams:
            return []

        events = self._streams[aggregate_id]

        if from_version:
            events = [e for e in events if e.version >= from_version]

        return events

    def clear(self) -> None:
        """Clear all events"""
        self._events.clear()
        self._streams.clear()


class EventBus:
    """Simple in-memory event bus for publishing events"""

    def __init__(self):
        self._handlers: Dict[str, List[callable]] = {}

    def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe to event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: callable) -> None:
        """Unsubscribe from event type"""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def publish(self, event: DomainEvent) -> None:
        """Publish event to all subscribers"""
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                if hasattr(handler, '__call__'):
                    import asyncio
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Error in event handler for {event.type}: {e}")

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Get number of subscribers"""
        if event_type:
            return len(self._handlers.get(event_type, []))
        return sum(len(handlers) for handlers in self._handlers.values())


# Global instances (in a real application, these would be injected)
_event_store = EventStore()
_event_bus = EventBus()


def get_event_store() -> EventStore:
    """Get global event store instance"""
    return _event_store


def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    return _event_bus


# Decorator for event handlers

def event_handler(event_type: str):
    """Decorator to register event handler"""
    def decorator(handler: callable):
        _event_bus.subscribe(event_type, handler)
        return handler
    return decorator


# Utility functions

async def publish_event(event: DomainEvent) -> None:
    """Publish event to event bus and store"""
    _event_store.append(event)
    await _event_bus.publish(event)


def create_event(event_type: str, aggregate_id: Optional[Union[int, str]] = None,
                data: Optional[Dict[str, Any]] = None,
                correlation_id: Optional[str] = None,
                causation_id: Optional[str] = None) -> DomainEvent:
    """Create domain event"""
    return DomainEvent(
        type=event_type,
        aggregate_id=aggregate_id,
        data=data or {},
        correlation_id=correlation_id,
        causation_id=causation_id
    )


def create_correlation_chain(events: List[DomainEvent]) -> List[DomainEvent]:
    """Create correlation chain for events"""
    if not events:
        return []

    correlation_id = str(uuid4())
    result = []

    for i, event in enumerate(events):
        causation_id = events[i-1].event_id if i > 0 else None

        # Create new event with correlation metadata
        event_data = event.to_dict()
        event_data["correlation_id"] = correlation_id
        event_data["causation_id"] = causation_id

        result.append(DomainEvent.from_dict(event_data))

    return result