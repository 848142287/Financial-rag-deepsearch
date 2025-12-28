"""
API路由总入口 - 清理版本
"""

from fastapi import APIRouter

# 导入各模块路由
from app.api.endpoints import (
    conversations,
    admin,
    deduplication,
    documents_enhanced,
    simple_search,
    optimized_search,
    enhanced_agentrag_search,
    intelligent_search_fixed,
    system_tools,
    enhanced_search_v2,
)

# 可选导入 - 可能有缺失依赖
try:
    from app.api.endpoints import evaluation
except (ImportError, Exception):
    evaluation = None

try:
    from app.api.endpoints import unified_pdf_extraction
except (ImportError, Exception):
    unified_pdf_extraction = None

try:
    from app.api.endpoints import enhanced_document_analysis
except (ImportError, Exception):
    enhanced_document_analysis = None

try:
    from app.api.endpoints import ocr_service
except (ImportError, Exception):
    ocr_service = None

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

# 3.5. 增强版搜索 v2 (NEW!)
api_router.include_router(
    enhanced_search_v2.router,
    prefix="/enhanced-search",
    tags=["增强版搜索v2"]
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
if evaluation:
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

# 8. 文档重试 (暂时注释 - 有循环导入问题)
# if document_retry:
#     api_router.include_router(
#         document_retry.router,
#         prefix="/document-retry",
#         tags=["文档重试"]
#     )

# 9. 优化搜索
api_router.include_router(
    optimized_search.router,
    prefix="/optimized-search",
    tags=["优化搜索"]
)

# 10. 增强版AgentRAG搜索
api_router.include_router(
    enhanced_agentrag_search.router,
    prefix="/enhanced-agentrag-search",
    tags=["增强版AgentRAG搜索"]
)

# 11. 系统工具
api_router.include_router(
    system_tools.router,
    tags=["系统工具"]
)

# 12. 统一 PDF 提取服务 (NEW! - 集成 Multimodal_RAG 功能)
if unified_pdf_extraction:
    api_router.include_router(
        unified_pdf_extraction.router,
        prefix="/pdf-extraction",
        tags=["统一PDF提取"]
    )

# 13. 增强文档分析服务 (NEW! - 集成 03_DataAnalysis_main 功能)
if enhanced_document_analysis:
    api_router.include_router(
        enhanced_document_analysis.router,
        prefix="/enhanced-document-analysis",
        tags=["增强文档分析"]
    )

# 14. OCR 服务 (NEW! - 集成 DeepSeek-OCR 功能)
if ocr_service:
    api_router.include_router(
        ocr_service.router,
        prefix="/ocr-service",
        tags=["OCR服务"]
    )

# 15. RAG问答服务 (NEW! - In-Context Learning + DeepSeek)
from app.api.endpoints import rag_qa
api_router.include_router(
    rag_qa.router,
    tags=["RAG问答"]
)

# 16. 监控服务 (NEW! - 系统监控和自动触发)
from app.api.endpoints import monitoring
api_router.include_router(
    monitoring.router,
    tags=["系统监控"]
)

# 健康检查端点
@api_router.get("/health-check")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": "2025-12-25",
        "endpoints": {
            "conversations": "/conversations",
            "documents": "/documents",
            "documents-enhanced": "/documents-enhanced",
            "intelligent-search": "/intelligent-search",
            "simple-search": "/simple-search",
            "admin": "/admin",
            "evaluation": "/evaluation",
            "deduplication": "/deduplication",
            "pdf-extraction": "/pdf-extraction",
            "enhanced-document-analysis": "/enhanced-document-analysis",
            "ocr-service": "/ocr-service"
        }
    }


@api_router.get("/health")
async def health():
    """健康检查端点（别名）"""
    return {"status": "healthy"}