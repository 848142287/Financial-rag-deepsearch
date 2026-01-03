"""
RAG问答API端点

提供基于In-Context Learning的问答功能
展示：来源、信任度、检索路径
"""

from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.core.structured_logging import get_structured_logger

from app.core.database import get_db
from app.core.errors.unified_errors import handle_errors, ErrorCategory
from app.services.rag_question_answering import get_rag_qa_service

logger = get_structured_logger(__name__)

router = APIRouter(prefix="/api/v1/qa", tags=["Question Answering"])

# 请求模型
class QuestionRequest(BaseModel):
    """问题请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    top_k: int = Field(10, description="检索的文档块数量", ge=1, le=50)
    min_confidence: float = Field(0.7, description="最小置信度阈值", ge=0.0, le=1.0)

class BatchQuestionRequest(BaseModel):
    """批量问题请求"""
    questions: List[str] = Field(..., description="问题列表", min_items=1, max_items=5)
    top_k: int = Field(10, description="每个问题检索的文档块数量", ge=1, le=50)
    min_confidence: float = Field(0.7, description="最小置信度阈值", ge=0.0, le=1.0)

# 响应模型
class SourceInfo(BaseModel):
    """来源信息"""
    chunk_id: int
    document_id: int
    document_title: str
    document_filename: str
    content: str
    confidence: float
    metadata: Dict[str, Any]

class TrustExplanation(BaseModel):
    """信任度解释"""
    evidence_strength: str = Field(..., description="证据强度: 强/中/弱")
    evidence_coverage: str = Field(..., description="证据覆盖度: 完整/部分/不足")
    answer_certainty: str = Field(..., description="答案确定性: 高/中/低")
    average_confidence: float = Field(..., description="平均置信度")
    source_count: int = Field(..., description="来源数量")
    document_count: int = Field(..., description="文档数量")
    reasoning: str = Field(..., description="推理说明")

class RetrievalPathStep(BaseModel):
    """检索路径步骤"""
    action: str = Field(..., description="动作名称")
    status: str = Field(..., description="状态: success/failed")
    details: Dict[str, Any] = Field(default_factory=dict, description="其他详细信息")

class QuestionAnswerResponse(BaseModel):
    """问答响应"""
    question: str
    answer: str
    sources: List[SourceInfo]
    trust_explanation: TrustExplanation
    retrieval_path: Dict[str, Dict[str, Any]]
    execution_time: float
    timestamp: str

@router.post("/ask", response_model=QuestionAnswerResponse, summary="提问接口")
@handle_errors(
    default_return={
        "question": "",
        "answer": "问答处理失败，请稍后重试",
        "sources": [],
        "trust_explanation": None,
        "retrieval_path": {},
        "execution_time": 0.0,
        "timestamp": ""
    },
    error_category=ErrorCategory.RETRIEVAL
)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """
    提问接口

    功能：
    1. 对问题进行embedding
    2. 检索相关文档块（置信度>min_confidence）
    3. 构建In-Context Learning提示词
    4. 调用DeepSeek生成答案
    5. 返回答案、来源、信任度、检索路径

    参数：
    - question: 用户问题
    - top_k: 检索的文档块数量（默认10）
    - min_confidence: 最小置信度阈值（默认0.7）
    """
    logger.info(f"收到问题: {request.question}")

    # 获取服务实例
    qa_service = get_rag_qa_service()

    # 处理问题
    result = await qa_service.answer_question(
        question=request.question,
        top_k=request.top_k,
        min_confidence=request.min_confidence,
        db=db
    )

    logger.info(f"问题回答完成，耗时: {result['execution_time']}秒")
    return result

@router.post("/ask-batch", summary="批量提问接口")
@handle_errors(
    default_return={
        "total": 0,
        "successful": 0,
        "failed": 0,
        "results": []
    },
    error_category=ErrorCategory.RETRIEVAL
)
async def ask_batch_questions(
    request: BatchQuestionRequest,
    db: Session = Depends(get_db)
):
    """
    批量提问接口

    支持一次提交多个问题，并行处理
    """
    logger.info(f"收到批量问题: {len(request.questions)}个")

    # 获取服务实例
    qa_service = get_rag_qa_service()

    # 并行处理所有问题
    import asyncio
    tasks = [
        qa_service.answer_question(
            question=q,
            top_k=request.top_k,
            min_confidence=request.min_confidence,
            db=db
        )
        for q in request.questions
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"问题{i+1}处理失败: {result}")
            responses.append({
                "question": request.questions[i],
                "error": str(result)
            })
        else:
            responses.append(result)

    return {
        "total": len(request.questions),
        "successful": sum([1 for r in responses if "error" not in r]),
        "failed": sum([1 for r in responses if "error" in r]),
        "results": responses
    }

@router.get("/health", summary="健康检查")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "RAG Question Answering",
        "features": [
            "Vector Search",
            "In-Context Learning",
            "Source Citation",
            "Trust Explanation",
            "Retrieval Path Visualization"
        ]
    }

@router.get("/example", summary="获取示例问题")
async def get_example_questions():
    """
    获取示例问题列表

    返回一些示例问题，方便用户测试
    """
    examples = [
        {
            "question": "什么是ChatGPT？它有哪些应用场景？",
            "description": "测试基本概念理解"
        },
        {
            "question": "钠离子电池的优势是什么？与锂电池相比如何？",
            "description": "测试技术对比分析"
        },
        {
            "question": "华为的AI盘古大模型有哪些特点？",
            "description": "测试产品特性了解"
        },
        {
            "question": "2023年中国金融业人才管理有哪些趋势？",
            "description": "测试行业趋势分析"
        },
        {
            "question": "GPU行业在AI算力方面的发展前景如何？",
            "description": "测试行业前景预测"
        }
    ]

    return {
        "examples": examples,
        "usage": "POST /api/v1/qa/ask with {\"question\": \"your question\"}"
    }
