"""
AgentRAG API接口
提供智能问答、深度搜索等功能
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import json
import logging

from ...core.database import get_db
from ...schemas.rag import QueryRequest, QueryResponse
from ...schemas.agent_rag import (
    AgentRAGQueryRequest,
    AgentRAGQueryResponse,
    BatchProcessRequest,
    BatchProcessResponse
)
from ...services.agent_rag.agent_engine import agent_rag_engine
from ...services.agent_rag.deepsearch import deepsearch_optimizer
from ...services.knowledge_base.pipeline import knowledge_base_pipeline
from ...services.document_parser.table_merger import cross_page_table_merger
from ...core.dependencies import get_current_user
from ...models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-rag", tags=["AgentRAG"])


@router.post("/query", response_model=AgentRAGQueryResponse)
async def agent_rag_query(
    request: AgentRAGQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Any = Depends(get_db)
):
    """
    AgentRAG智能查询
    """
    try:
        logger.info(f"AgentRAG查询请求: {request.query[:50]}...")

        # 执行AgentRAG
        result = await agent_rag_engine.execute(
            query=request.query,
            context=request.context,
            options=request.options
        )

        # 构建响应
        response = AgentRAGQueryResponse(
            query=request.query,
            answer=result.final_answer,
            confidence_score=result.confidence_score,
            sources=[
                {
                    "document_id": doc.document_id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "score": doc.score,
                    "source": doc.source
                }
                for doc in result.iterations[-1].retrieved_docs if result.iterations
            ][:5],
            ragas_metrics=result.ragas_metrics,
            execution_trace=result.execution_trace,
            metadata=result.metadata
        )

        # 异步记录查询历史
        if current_user:
            background_tasks.add_task(
                _save_query_history,
                current_user.id,
                request.query,
                result.final_answer,
                result.confidence_score
            )

        return response

    except Exception as e:
        logger.error(f"AgentRAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@router.post("/deep-search")
async def deep_search(
    request: AgentRAGQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    DeepSearch深度搜索
    """
    try:
        logger.info(f"DeepSearch请求: {request.query[:50]}...")

        result = await deepsearch_optimizer.deep_search(
            query=request.query,
            max_iterations=request.options.get('max_iterations', 3) if request.options else 3
        )

        return {
            "query": result.query,
            "answer": result.final_answer,
            "confidence_score": result.confidence_score,
            "optimization_actions": result.optimization_actions,
            "search_trace": result.search_trace,
            "ragas_metrics": result.ragas_metrics,
            "iterations_count": len(result.all_iterations)
        }

    except Exception as e:
        logger.error(f"DeepSearch失败: {e}")
        raise HTTPException(status_code=500, detail=f"深度搜索失败: {str(e)}")


@router.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process_documents(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    批量处理文档
    """
    try:
        logger.info(f"批量文档处理请求: {len(request.file_paths)} 个文件")

        # 验证文件数量
        if len(request.file_paths) > 100:
            raise HTTPException(status_code=400, detail="单次最多处理100个文件")

        # 启动后台批量处理
        task_id = await _start_batch_processing(
            request.file_paths,
            request.config,
            current_user.id
        )

        return BatchProcessResponse(
            task_id=task_id,
            status="started",
            total_files=len(request.file_paths),
            message="批量处理已启动"
        )

    except Exception as e:
        logger.error(f"批量处理启动失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")


@router.get("/batch-status/{task_id}")
async def get_batch_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取批量处理状态
    """
    try:
        status = await _get_batch_processing_status(task_id, current_user.id)
        return status
    except Exception as e:
        logger.error(f"获取批量处理状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取状态失败")


@router.delete("/batch-cancel/{task_id}")
async def cancel_batch_processing(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    取消批量处理
    """
    try:
        success = await _cancel_batch_processing(task_id, current_user.id)
        return {"success": success, "message": "取消成功" if success else "取消失败"}
    except Exception as e:
        logger.error(f"取消批量处理失败: {e}")
        raise HTTPException(status_code=500, detail="取消失败")


@router.post("/merge-tables")
async def merge_cross_page_tables(
    tables_data: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user)
):
    """
    合并跨页表格
    """
    try:
        logger.info(f"跨页表格合并请求: {len(tables_data)} 个表格")

        # 执行表格合并
        merge_results = await cross_page_table_merger.merge_cross_page_tables(tables_data)

        # 验证合并结果
        validation_results = []
        for result in merge_results:
            validation = cross_page_table_merger.validate_merged_table(result)
            validation_results.append(validation)

        return {
            "merged_tables": [
                {
                    "table_id": result.merged_table_id,
                    "original_tables": result.original_tables,
                    "confidence": result.confidence_score,
                    "html_content": result.merged_html,
                    "text_content": result.merged_text
                }
                for result in merge_results
            ],
            "validation_results": validation_results
        }

    except Exception as e:
        logger.error(f"跨页表格合并失败: {e}")
        raise HTTPException(status_code=500, detail=f"表格合并失败: {str(e)}")


@router.get("/query-history")
async def get_query_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Any = Depends(get_db)
):
    """
    获取查询历史
    """
    try:
        history = await _get_user_query_history(current_user.id, limit, offset, db)
        return {"history": history, "total": len(history)}
    except Exception as e:
        logger.error(f"获取查询历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取历史失败")


@router.post("/evaluate")
async def evaluate_response(
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    ground_truth: Optional[str] = None
):
    """
    评估回答质量
    """
    try:
        from ...services.agent_rag.ragas_evaluator import ragas_evaluator

        metrics = await ragas_evaluator.evaluate(
            query=query,
            answer=answer,
            retrieved_docs=retrieved_docs,
            ground_truth=ground_truth
        )

        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise HTTPException(status_code=500, detail="评估失败")


# 辅助函数
async def _save_query_history(user_id: str, query: str, answer: str, confidence: float):
    """保存查询历史"""
    try:
        # 实现查询历史保存逻辑
        logger.info(f"保存查询历史: 用户{user_id}, 置信度{confidence:.3f}")
    except Exception as e:
        logger.error(f"保存查询历史失败: {e}")


async def _start_batch_processing(file_paths: List[str], config: Dict, user_id: str) -> str:
    """启动批量处理"""
    import uuid
    task_id = str(uuid.uuid4())

    # 这里可以启动Celery任务或其他后台任务
    logger.info(f"启动批量处理任务: {task_id}")

    return task_id


async def _get_batch_processing_status(task_id: str, user_id: str) -> Dict:
    """获取批量处理状态"""
    # 实现状态查询逻辑
    return {
        "task_id": task_id,
        "status": "processing",
        "progress": 50,
        "completed": 5,
        "total": 10,
        "errors": []
    }


async def _cancel_batch_processing(task_id: str, user_id: str) -> bool:
    """取消批量处理"""
    # 实现取消逻辑
    logger.info(f"取消批量处理任务: {task_id}")
    return True


async def _get_user_query_history(user_id: str, limit: int, offset: int, db) -> List[Dict]:
    """获取用户查询历史"""
    # 实现历史查询逻辑
    return [
        {
            "id": "1",
            "query": "示例查询",
            "answer": "示例答案",
            "confidence": 0.85,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    ]