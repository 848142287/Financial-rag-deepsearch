"""
用户模型
定义用户相关的数据模型
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from enum import Enum


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class UserBase(BaseModel):
    """用户基础模型"""
    username: str
    email: EmailStr
    is_active: bool = True
    role: UserRole = UserRole.USER


class UserCreate(UserBase):
    """创建用户模型"""
    password: str


class UserUpdate(BaseModel):
    """更新用户模型"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None
    password: Optional[str] = None


class UserResponse(UserBase):
    """用户响应模型"""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """用户登录模型"""
    username: str
    password: str


class Token(BaseModel):
    """令牌模型"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """令牌数据模型"""
    username: Optional[str] = None
    user_id: Optional[str] = None


class UserInDB(UserBase):
    """数据库中的用户模型"""
    id: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# 简化的用户类，用于兼容性
class User(UserResponse):
    """用户模型（兼容性）"""
    pass


class UserProfile(BaseModel):
    """用户档案模型"""
    id: str
    username: str
    email: EmailStr
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    document_count: int = 0
    search_count: int = 0

    class Config:
        from_attributes = True