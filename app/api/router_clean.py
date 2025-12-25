"""
API路由总入口 - 清理版本
"""

from fastapi import APIRouter

# 导入各模块路由
from app.api.endpoints import conversations, admin, evaluation, deduplication, documents_enhanced, simple_search, document_retry, optimized_search
from app.api.endpoints import intelligent_search_fixed

# 创建主路由器
api_router = APIRouter()

# 1. 对话管理
api_router.include_router(
    conversations.router,
    prefix="/conversations",
    tags=["对话管理"]
)

# 2. 文档管理
api_router.include_router(
    documents_enhanced.router,
    prefix="/documents-enhanced",
    tags=["增强文档管理"]
)

# 为前端兼容性
api_router.include_router(
    documents_enhanced.router,
    prefix="/documents",
    tags=["文档管理"]
)

# 3. 智能搜索
api_router.include_router(
    intelligent_search_fixed.router,
    prefix="/intelligent-search",
    tags=["智能搜索"]
)

# 4. 简单搜索
api_router.include_router(
    simple_search.router,
    prefix="/simple-search",
    tags=["简单搜索"]
)

# 5. 系统管理
api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["系统管理"]
)

# 6. RAG评估
api_router.include_router(
    evaluation.router,
    prefix="/evaluation",
    tags=["RAG评估"]
)

# 7. 文档去重
api_router.include_router(
    deduplication.router,
    prefix="/deduplication",
    tags=["文档去重"]
)

# 8. 文档重试
api_router.include_router(
    document_retry.router,
    prefix="/document-retry",
    tags=["文档重试"]
)

# 9. 优化搜索
api_router.include_router(
    optimized_search.router,
    prefix="/optimized-search",
    tags=["优化搜索"]
)

# 健康检查端点
@api_router.get("/health-check")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": "2025-12-21",
        "endpoints": {
            "conversations": "/conversations",
            "documents": "/documents",
            "documents-enhanced": "/documents-enhanced",
            "intelligent-search": "/intelligent-search",
            "simple-search": "/simple-search",
            "admin": "/admin",
            "evaluation": "/evaluation",
            "deduplication": "/deduplication"
        }
    }