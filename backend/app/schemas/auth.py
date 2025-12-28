"""
认证相关的Pydantic模式
"""

from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """用户基础模式"""
    username: str
    email: EmailStr


class UserCreate(UserBase):
    """用户创建模式"""
    password: str


class User(UserBase):
    """用户响应模式"""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """令牌模式"""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """令牌数据模式"""
    username: Optional[str] = None