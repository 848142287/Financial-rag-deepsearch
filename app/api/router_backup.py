"""
API路由总入口
"""

from fastapi import APIRouter

# 导入各模块路由
from app.api.endpoints import conversations, admin, evaluation, sla, progress, deduplication, documents_enhanced, simple_search, document_retry, optimized_search
# Temporarily disable problematic consolidated endpoints
# from app.api.endpoints import consolidated_rag, consolidated_documents
# Temporarily disable problematic endpoints
# from app.api.endpoints import enhanced_evaluation, fusion_agent, feedback_optimizer, async_tasks, enhanced_batch_upload, documents_batch, unified_system
# Temporarily disable v1 endpoints as well

# 创建主路由器
api_router = APIRouter()

# api_router.include_router(
#     documents.router,
#     prefix="/documents",
#     tags=["文档管理"]
# )

api_router.include_router(
    conversations.router,
    prefix="/conversations",
    tags=["对话管理"]
)

# api_router.include_router(
#     intelligent_search_fixed.router,
#     prefix="/search-fixed",
#     tags=["智能搜索"]
# )

# Temporarily disable consolidated endpoints
# api_router.include_router(
#     consolidated_rag.router,
#     tags=["统一RAG检索"]
# )

# api_router.include_router(
#     consolidated_documents.router,
#     tags=["统一文档处理"]
# )

# 保留原有端点作为向后兼容（注释掉）
# api_router.include_router(
#     rag.router,
#     prefix="/rag",
#     tags=["RAG检索(旧版)"]
# )

# api_router.include_router(
#     documents.router,
#     prefix="/documents",
#     tags=["文档管理(旧版)"]
# )

api_router.include_router(
    evaluation.router,
    prefix="/evaluation",
    tags=["RAG评估"]
)

# Temporarily disable problematic endpoints
# api_router.include_router(
#     enhanced_evaluation.router,
#     prefix="/evaluation/enhanced",
#     tags=["增强评估"]
# )

api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["系统管理"]
)

api_router.include_router(
    deduplication.router,
    prefix="/deduplication",
    tags=["文档去重"]
)

api_router.include_router(
    documents_enhanced.router,
    prefix="/documents-enhanced",
    tags=["增强文档管理"]
)

# 为前端兼容性添加 /api/v1/documents 路点
api_router.include_router(
    documents_enhanced.router,
    prefix="/documents",
    tags=["文档管理"]
)

# 添加AI增强的文档管理端点
# api_router.include_router(
#     documents_enhanced_v3.router,
#     prefix="/documents-enhanced-v3",
#     tags=["AI增强文档管理"]
# )

# 为前端兼容性添加 /api/v1/documents-enhanced-v3 路点
# api_router.include_router(
#     documents_enhanced_v3.router,
#     prefix="/documents-v3",
#     tags=["AI增强文档管理v3"]
# )

# 启用修复后的智能搜索API
from app.api.endpoints import intelligent_search_fixed

api_router.include_router(
    intelligent_search_fixed.router,
    prefix="/intelligent-search",
    tags=["智能搜索"]
)

# 启用简单搜索API
api_router.include_router(
    simple_search.router,
    prefix="/simple-search",
    tags=["简单搜索"]
)

# 启用文档重试API
api_router.include_router(
    document_retry.router,
    tags=["文档重试"]
)

# 启用优化搜索API
api_router.include_router(
    optimized_search.router,
    tags=["优化搜索"]
)
#
# api_router.include_router(
#     agentic_rag_api.router,
#     prefix="/agentic-rag",
#     tags=["AgenticRAG检索"]
# )
#
# api_router.include_router(
#     agent_rag.router,
#     prefix="/agent-rag",
#     tags=["AgentRAG检索"]
# )

# api_router.include_router(
#     sla.router,
#     tags=["SLA监控"]
# )

# api_router.include_router(
#     progress.router,
#     tags=["进度查询"]
# )

# api_router.include_router(
#     database_query.router,
#     prefix="/database",
#     tags=["数据库查询"]
# )

# Temporarily disable problematic endpoints
# api_router.include_router(
#     fusion_agent.router,
#     prefix="/fusion-agent",
#     tags=["融合智能体"]
# )

# api_router.include_router(
#     feedback_optimizer.router,
#     prefix="/feedback-optimizer",
#     tags=["反馈优化器"]
# )

# api_router.include_router(
#     async_tasks.router,
#     prefix="/async-tasks",
#     tags=["异步任务"]
# )

# api_router.include_router(
#     documents_batch.router,
#     tags=["批量文档操作"]
# )

# api_router.include_router(
#     enhanced_batch_upload.router,
#     tags=["智能批量文档处理"]
# )

# 统一系统API - 整合所有功能的统一入口
# api_router.include_router(
#     unified_system.router,
#     tags=["统一系统"]
# )