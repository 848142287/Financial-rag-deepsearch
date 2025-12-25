"""
API路由初始化
"""

from .router import api_router
# Temporarily disable v2 endpoints

__all__ = ["api_router"]  # , "knowledge_fusion_router", "fusion_agent_router"