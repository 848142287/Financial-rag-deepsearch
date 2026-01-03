"""
认证模块 - 简化版本
提供基本的JWT认证功能
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# JWT配置
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# HTTP Bearer认证
security = HTTPBearer()

class TokenData(BaseModel):
    """Token数据模型"""
    username: Optional[str] = None
    user_id: Optional[str] = None

class User(BaseModel):
    """用户模型"""
    id: str
    username: str
    email: str
    is_active: bool = True

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    """验证令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_data = TokenData(username=username, user_id=user_id)
        return token_data

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """获取当前用户（需要认证）"""
    token_data = verify_token(credentials.credentials)

    # 这里应该从数据库获取用户信息
    # 简化实现，返回一个模拟用户
    user = User(
        id=token_data.user_id or "default_user",
        username=token_data.username or "default_user",
        email=f"{token_data.username}@example.com"
    )

    return user


def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[User]:
    """获取当前用户（可选认证，用于允许匿名访问的端点）"""
    if credentials is None:
        # 没有提供token，返回默认用户
        return User(
            id="anonymous",
            username="anonymous",
            email="anonymous@local",
            is_active=True
        )

    try:
        token_data = verify_token(credentials.credentials)
        user = User(
            id=token_data.user_id or "default_user",
            username=token_data.username or "default_user",
            email=f"{token_data.username}@example.com",
            is_active=True
        )
        return user
    except JWTError:
        # Token无效，返回默认用户
        return User(
            id="anonymous",
            username="anonymous",
            email="anonymous@local",
            is_active=True
        )

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# 可选的权限检查
def require_permissions(required_permissions: list):
    """权限检查装饰器"""
    def permission_checker(current_user: User = Depends(get_current_active_user)):
        # 简化实现 - 实际应该检查用户权限
        # 这里只是示例，假设所有用户都有权限
        return current_user
    return permission_checker