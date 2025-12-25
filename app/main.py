"""
金融RAG系统 - 主应用入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
import logging

logger = logging.getLogger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title="金融RAG深度检索系统",
    description="基于多级检索和AgentRAG的金融研报智能问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("金融RAG系统启动中...")
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
