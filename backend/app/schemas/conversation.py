"""
对话相关的Pydantic模式
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class MessageBase(BaseModel):
    """消息基础模式"""
    role: str  # 'user', 'assistant', 'system'
    content: str


class MessageCreate(MessageBase):
    """消息创建模式"""
    conversation_id: int
    metadata: Optional[Dict[str, Any]] = None


class Message(MessageBase):
    """消息响应模式"""
    id: int
    conversation_id: int
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    """对话基础模式"""
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    """对话创建模式"""
    # 可以接受空的title，将使用默认值


class Conversation(ConversationBase):
    """对话响应模式"""
    id: int
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []

    class Config:
        from_attributes = True


class ConversationWithMessages(Conversation):
    """包含消息的对话模式"""


class WebSocketMessage(BaseModel):
    """WebSocket消息模式"""
    type: str  # 'message', 'error', 'status'
    data: Dict[str, Any]