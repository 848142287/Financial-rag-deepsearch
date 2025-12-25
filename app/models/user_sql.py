"""
用户数据库模型
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Enum, JSON, Boolean
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)

    # 用户状态
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)
    role = Column(Enum(UserRole), default=UserRole.USER, index=True)

    # 用户信息
    full_name = Column(String(255))
    avatar_url = Column(String(500))
    bio = Column(Text)
    preferences = Column(JSON)  # 用户偏好设置

    # 统计信息
    document_count = Column(Integer, default=0)
    search_count = Column(Integer, default=0)
    last_login_at = Column(DateTime(timezone=True))

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}', role='{self.role}')>"