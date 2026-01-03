"""
金融RAG系统 - 主应用入口
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.router import api_router
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title="金融RAG深度检索系统",
    description="基于多级检索和AgentRAG的金融研报智能问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS - 添加到所有响应（包括错误响应）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 注册API路由
app.include_router(api_router, prefix="/api/v1")

# 全局异常处理器 - 确保所有响应都有CORS头
def get_cors_headers():
    """获取CORS响应头"""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": 500,
                "message": str(exc) if logger.isEnabledFor(40) else "服务器内部错误",
                "type": type(exc).__name__
            },
            "path": str(request.url.path)
        },
        headers=get_cors_headers()
    )

from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理器 - 确保CORS头"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "HTTPException"
            },
            "path": str(request.url.path)
        },
        headers=get_cors_headers()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """验证异常处理器 - 确保CORS头"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": 422,
                "message": "请求验证失败",
                "type": "ValidationError",
                "details": exc.errors()
            },
            "path": str(request.url.path)
        },
        headers=get_cors_headers()
    )

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("金融RAG系统启动中...")

    # 预热BGE模型（可选，通过环境变量控制）
    import os
    warmup_enabled = os.getenv("BGE_MODEL_WARMUP", "true").lower() == "true"

    if warmup_enabled:
        try:
            logger.info("开始预热BGE模型...")
            from app.services.embeddings.unified_embedding_service import get_embedding_service
            from app.services.bge_reranker_local import get_bge_reranker_service, BGERerankerConfig
            from app.core.config import settings

            # 预热嵌入模型
            embedding_service = get_embedding_service()
            embedding_service.warmup()

            # 预热重排序模型
            reranker_config = BGERerankerConfig(
                model_path=settings.bge_reranker_model_path,
                device=settings.bge_reranker_device
            )
            reranker_service = get_bge_reranker_service(reranker_config)
            reranker_service.warmup()

            logger.info("BGE模型预热完成")
        except Exception as e:
            logger.warning(f"BGE模型预热失败（将继续运行）: {e}")
    else:
        logger.info("BGE模型预热已禁用（通过BGE_MODEL_WARMUP=false）")

    logger.info("金融RAG系统启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("金融RAG系统关闭中...")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "金融RAG深度检索系统API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health-check"
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}
