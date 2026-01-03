"""
API路由总入口 - 清理版本

⚠️ 已废弃的端点标记：
- intelligent_search_fixed: 已移除，请使用 /api/v1/search
- enhanced_search_v2: 已移除，请使用 /api/v1/search
- simple_search: 已删除，请使用 /api/v1/search
- enhanced_agentrag_search: 被 /api/v1/search?strategy=agentic 替代
"""

from fastapi import APIRouter
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# 导入各模块路由
from app.api.endpoints import (
    conversations,
    deduplication,
    optimized_search,
    system_tools,
    document_retry,
    documents,  # 新增：基础文档管理API
)

# 新的统一搜索API
from app.api.endpoints.v1 import unified_search, document_pipeline

# 可选导入 - 可能有缺失依赖
try:
    from app.api.endpoints import admin
except (ImportError, Exception):
    admin = None

try:
    from app.api.endpoints import documents_enhanced
except (ImportError, Exception):
    documents_enhanced = None

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

# ============================================================================
# 0. 统一搜索API (推荐使用)
# ============================================================================
api_router.include_router(
    unified_search.router,
    prefix="/search",
    tags=["统一搜索 API"]
)

# ============================================================================
# 文档处理流水线 API (完整版)
# ============================================================================
api_router.include_router(
    document_pipeline.router,
    prefix="/v1/pipeline",
    tags=["文档处理流水线"]
)

# ============================================================================
# 已废弃的搜索端点 (将被移除，请使用 /api/v1/search)
# ============================================================================
import warnings

# 为废弃端点添加警告
def deprecation_warning():
    """显示废弃警告"""
    warnings.warn(
        "此端点已废弃，请使用 /api/v1/search。旧端点将在未来版本中移除。",
        DeprecationWarning,
        stacklevel=2
    )

# 1. 对话管理
api_router.include_router(
    conversations.router,
    prefix="/conversations",
    tags=["对话管理"]
)

# 2. 文档管理（基础版 - 始终启用）
api_router.include_router(
    documents.router,
    tags=["文档管理"]
)

# 2.1 增强文档管理（如果可用）
if documents_enhanced:
    api_router.include_router(
        documents_enhanced.router,
        prefix="/documents-enhanced",
        tags=["增强文档管理"]
    )

# 5. 系统管理
if admin:
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

# 10. 系统工具
api_router.include_router(
    system_tools.router,
    tags=["系统工具"]
)

# 11. 统一 PDF 提取服务 (NEW! - 集成 Multimodal_RAG 功能)
if unified_pdf_extraction:
    api_router.include_router(
        unified_pdf_extraction.router,
        prefix="/pdf-extraction",
        tags=["统一PDF提取"]
    )

# 12. 增强文档分析服务 (NEW! - 集成 03_DataAnalysis_main 功能)
if enhanced_document_analysis:
    api_router.include_router(
        enhanced_document_analysis.router,
        prefix="/enhanced-document-analysis",
        tags=["增强文档分析"]
    )

# 13. OCR 服务 (NEW! - 集成 DeepSeek-OCR 功能)
if ocr_service:
    api_router.include_router(
        ocr_service.router,
        prefix="/ocr-service",
        tags=["OCR服务"]
    )

# 14. RAG问答服务 (NEW! - In-Context Learning + DeepSeek)
try:
    from app.api.endpoints import rag_qa
    api_router.include_router(
        rag_qa.router,
        tags=["RAG问答"]
    )
except (ImportError, Exception) as e:
    logger.warning(f"RAG问答服务加载失败: {e}")

# 14.1. RAG问答服务 (GLM-4.7 专用)
try:
    from app.api.endpoints.v1 import rag_glm_qa
    api_router.include_router(
        rag_glm_qa.router,
        tags=["RAG问答 (GLM-4.7)"]
    )
except (ImportError, Exception) as e:
    logger.warning(f"RAG问答服务(GLM)加载失败: {e}")

# 15. 监控服务 (NEW! - 系统监控和自动触发)
try:
    from app.api.endpoints import monitoring
    api_router.include_router(
        monitoring.router,
        prefix="/monitoring",
        tags=["系统监控"]
    )
except (ImportError, Exception) as e:
    logger.warning(f"监控服务加载失败: {e}")

# 15.1 备份恢复服务 (NEW! - 数据备份和恢复)
try:
    from app.api.endpoints import backup_restore
    api_router.include_router(
        backup_restore.router,
        prefix="/backup-restore",
        tags=["备份恢复"]
    )
except (ImportError, Exception) as e:
    logger.warning(f"备份恢复服务加载失败: {e}")

# 16. 增强监控服务 (NEW! - 备份恢复、批量评估等)
try:
    from app.api.endpoints import enhanced_monitoring
    api_router.include_router(
        enhanced_monitoring.router,
        prefix="/enhanced-monitoring",
        tags=["增强监控"]
    )
except (ImportError, Exception) as e:
    pass

# 17. 增强反馈优化 (NEW! - 问题分析、意图识别)
try:
    from app.api.endpoints import feedback_optimizer
    api_router.include_router(
        feedback_optimizer.router,
        prefix="/feedback",
        tags=["反馈优化"]
    )
except (ImportError, Exception) as e:
    pass

# 18. 数据去重服务 (NEW! - Milvus向量和Neo4j去重)
try:
    from app.api.endpoints import data_deduplication
    api_router.include_router(
        data_deduplication.router,
        prefix="/dedup",
        tags=["数据去重"]
    )
except (ImportError, Exception) as e:
    logger.warning(f"数据去重服务加载失败: {e}")

# 19. 质量评估监控服务 (NEW! - 向量、融合、图谱质量)
try:
    from app.api.endpoints import quality_monitoring
    api_router.include_router(
        quality_monitoring.router,
        prefix="/quality",
        tags=["质量评估监控"]
    )
except (ImportError, Exception) as e:
    logger.warning(f"质量评估监控服务加载失败: {e}")

# 19. RAGAS检索评估流水线 (NEW! - 检索质量评估)
try:
    from app.api.endpoints import ragas_retrieval_eval
    api_router.include_router(
        ragas_retrieval_eval.router,
        prefix="/ragas-retrieval-eval",
        tags=["RAGAS检索评估"]
    )
except (ImportError, Exception) as e:
    logger.warning(f"RAGAS检索评估服务加载失败: {e}")

# 健康检查端点
@api_router.get("/health-check")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": "2025-12-29",
        "endpoints": {
            "unified_search": "/api/v1/search",
            "conversations": "/conversations",
            "documents": "/documents",
            "documents-enhanced": "/documents-enhanced",
            "document-retry": "/document-retry",
            "optimized_search": "/optimized-search",
            "admin": "/admin",
            "evaluation": "/evaluation",
            "deduplication": "/deduplication",
            "pdf-extraction": "/pdf-extraction",
            "enhanced-document-analysis": "/enhanced-document-analysis",
            "ocr-service": "/ocr-service"
        },
        "deprecated_endpoints": {
            "intelligent-search": "已移除，请使用 /api/v1/search",
            "enhanced-search": "已移除，请使用 /api/v1/search",
            "simple-search": "已删除，请使用 /api/v1/search",
            "enhanced-agentrag-search": "已废弃，请使用 /api/v1/search?strategy=agentic"
        }
    }


@api_router.get("/health")
async def health():
    """健康检查端点（别名）"""
    return {"status": "healthy"}