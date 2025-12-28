"""
异步任务API端点
提供任务管理、进度查询、WebSocket连接等功能
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
import uuid
from datetime import datetime

from app.core.celery_config import celery_app, get_task_statistics
from app.core.websocket_manager import connection_manager, MessageType, get_connection_manager
from app.core.auth import get_current_user
from app.models.user import User
from app.tasks.document_processing import process_document, batch_process_documents
from app.tasks.retrieval import hybrid_search, batch_search
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)

router = APIRouter()

# 请求模型
class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小")
    file_type: str = Field(..., description="文件类型")
    processing_options: Optional[Dict[str, Any]] = Field(None, description="处理选项")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询")
    max_results: Optional[int] = Field(10, description="最大结果数")
    similarity_threshold: Optional[float] = Field(0.7, description="相似度阈值")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")


class BatchSearchRequest(BaseModel):
    """批量搜索请求"""
    queries: List[Dict[str, str]] = Field(..., description="查询列表")
    search_options: Optional[Dict[str, Any]] = Field(None, description="搜索选项")


# API端点
@router.post("/documents/upload")
async def upload_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    上传文档并启动处理任务
    """
    try:
        # 生成文档ID
        document_id = str(uuid.uuid4())

        # 生成MinIO存储路径
        file_path = f"uploads/{current_user.id}/{document_id}/{request.file_name}"

        # 创建文档记录
        document_data = {
            'id': document_id,
            'user_id': current_user.id,
            'file_name': request.file_name,
            'file_size': request.file_size,
            'file_type': request.file_type,
            'file_path': file_path,
            'upload_time': datetime.now(),
            'status': 'uploading'
        }

        # 保存到数据库
        # mysql_client.save_document(document_data)

        # 启动文档处理任务
        task = process_document.delay(
            document_id=document_id,
            file_path=file_path,
            user_id=current_user.id,
            processing_options=request.processing_options
        )

        # 更新任务ID
        # mysql_client.update_document_task_id(document_id, task.id)

        logger.info(f"文档上传成功: {document_id}, 任务ID: {task.id}")

        return {
            "success": True,
            "data": {
                "document_id": document_id,
                "task_id": task.id,
                "status": "uploaded",
                "message": "文档上传成功，开始处理"
            }
        }

    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.post("/search")
async def create_search_task(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    创建搜索任务
    """
    try:
        # 生成查询ID
        query_id = str(uuid.uuid4())

        # 启动搜索任务
        task = hybrid_search.delay(
            query_id=query_id,
            query=request.query,
            user_id=current_user.id,
            search_options={
                'max_results': request.max_results,
                'similarity_threshold': request.similarity_threshold,
                'filters': request.filters or {}
            }
        )

        logger.info(f"搜索任务创建成功: {query_id}, 任务ID: {task.id}")

        return {
            "success": True,
            "data": {
                "query_id": query_id,
                "task_id": task.id,
                "query": request.query,
                "status": "started",
                "message": "搜索任务已启动"
            }
        }

    except Exception as e:
        logger.error(f"创建搜索任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建搜索任务失败: {str(e)}")


@router.post("/search/batch")
async def create_batch_search_task(
    request: BatchSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    创建批量搜索任务
    """
    try:
        if len(request.queries) > 50:
            raise HTTPException(status_code=400, detail="批量搜索最多支持50个查询")

        # 为每个查询生成ID
        queries_with_ids = []
        for query_data in request.queries:
            query_id = str(uuid.uuid4())
            queries_with_ids.append({
                'id': query_id,
                'query': query_data.get('query', '')
            })

        # 启动批量搜索任务
        task = batch_search.delay(
            queries=queries_with_ids,
            user_id=current_user.id,
            search_options=request.search_options
        )

        logger.info(f"批量搜索任务创建成功: {len(queries_with_ids)} 个查询, 任务ID: {task.id}")

        return {
            "success": True,
            "data": {
                "batch_id": task.id,
                "query_count": len(queries_with_ids),
                "queries": queries_with_ids,
                "status": "started",
                "message": "批量搜索任务已启动"
            }
        }

    except Exception as e:
        logger.error(f"创建批量搜索任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建批量搜索任务失败: {str(e)}")


@router.get("/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取任务状态
    """
    try:
        # 从Celery获取任务状态
        result = celery_app.AsyncResult(task_id)

        if result.state == 'PENDING':
            status = {
                'task_id': task_id,
                'status': 'pending',
                'message': '任务等待中'
            }
        elif result.state == 'PROGRESS':
            status = {
                'task_id': task_id,
                'status': 'processing',
                'progress': result.info.get('current', 0),
                'total': result.info.get('total', 100),
                'message': result.info.get('status', '处理中')
            }
        elif result.state == 'SUCCESS':
            status = {
                'task_id': task_id,
                'status': 'completed',
                'result': result.result,
                'message': '任务完成'
            }
        elif result.state == 'FAILURE':
            status = {
                'task_id': task_id,
                'status': 'failed',
                'error': str(result.info),
                'message': '任务失败'
            }
        else:
            status = {
                'task_id': task_id,
                'status': result.state.lower(),
                'message': f'任务状态: {result.state}'
            }

        return {
            "success": True,
            "data": status
        }

    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    取消任务
    """
    try:
        # 撤销Celery任务
        celery_app.control.revoke(task_id, terminate=True)

        logger.info(f"任务取消成功: {task_id}")

        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "cancelled",
                "message": "任务已取消"
            }
        }

    except Exception as e:
        logger.error(f"取消任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.get("/tasks/statistics")
async def get_task_statistics_endpoint(
    current_user: User = Depends(get_current_user)
):
    """
    获取任务统计信息
    """
    try:
        stats = get_task_statistics()

        return {
            "success": True,
            "data": stats
        }

    except Exception as e:
        logger.error(f"获取任务统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务统计失败: {str(e)}")


@router.get("/system/health")
async def get_system_health():
    """
    获取系统健康状态
    """
    try:
        # 获取Celery工作状态
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        stats = inspect.stats()

        # 获取Redis状态
        try:
            redis_info = redis_client.info()
            redis_status = "healthy"
        except:
            redis_status = "unhealthy"

        health_data = {
            "celery_workers": {
                "active": len(stats) if stats else 0,
                "total_tasks": sum(worker.get('total', 0) for worker in stats.values()) if stats else 0
            },
            "active_tasks": len(active_tasks) if active_tasks else 0,
            "redis_status": redis_status,
            "timestamp": datetime.now().isoformat()
        }

        return {
            "success": True,
            "data": health_data
        }

    except Exception as e:
        logger.error(f"获取系统健康状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统健康状态失败: {str(e)}")


# WebSocket端点
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """
    WebSocket连接端点
    """
    manager = get_connection_manager()
    connection_id = None

    try:
        # 建立连接
        connection_id = await manager.connect(websocket, user_id)

        logger.info(f"WebSocket连接建立: 用户={user_id}, 连接ID={connection_id}")

        # 处理消息循环
        while True:
            try:
                # 接收消息
                data = await websocket.receive_text()
                message = json.loads(data)

                # 处理不同类型的消息
                await handle_websocket_message(manager, websocket, user_id, connection_id, message)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket消息处理失败: {e}")
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        # 清理连接
        if connection_id:
            await manager.disconnect(connection_id)
            logger.info(f"WebSocket连接断开: 用户={user_id}, 连接ID={connection_id}")


async def handle_websocket_message(manager, websocket: WebSocket, user_id: int,
                                 connection_id: str, message: Dict[str, Any]):
    """
    处理WebSocket消息
    """
    message_type = message.get('type')

    if message_type == 'subscribe_task':
        # 订阅任务进度
        task_id = message.get('task_id')
        if task_id:
            await manager.subscribe_task(task_id, connection_id)

    elif message_type == 'unsubscribe_task':
        # 取消订阅任务
        task_id = message.get('task_id')
        if task_id:
            await manager.unsubscribe_task(task_id, connection_id)

    elif message_type == 'heartbeat':
        # 处理心跳
        await manager.handle_heartbeat(connection_id)

    elif message_type == 'get_connection_stats':
        # 获取连接统计
        stats = await manager.get_connection_stats()
        await manager.send_personal_message({
            'type': 'connection_stats',
            'data': stats
        }, connection_id)

    else:
        # 未知消息类型
        await manager.send_personal_message({
            'type': 'error',
            'data': {
                'message': f'未知消息类型: {message_type}'
            }
        }, connection_id)


@router.get("/ws/test", response_class=HTMLResponse)
async def websocket_test_page():
    """
    WebSocket测试页面
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket测试</title>
    </head>
    <body>
        <h1>WebSocket连接测试</h1>
        <div>
            <input type="number" id="userId" placeholder="用户ID" value="1">
            <button onclick="connect()">连接</button>
            <button onclick="disconnect()">断开</button>
        </div>
        <div>
            <input type="text" id="taskId" placeholder="任务ID">
            <button onclick="subscribeTask()">订阅任务</button>
        </div>
        <div>
            <button onclick="sendHeartbeat()">发送心跳</button>
        </div>
        <div>
            <h3>消息日志:</h3>
            <div id="messages" style="border: 1px solid #ccc; height: 300px; overflow-y: scroll; padding: 10px;"></div>
        </div>

        <script>
            let ws = null;

            function connect() {
                const userId = document.getElementById('userId').value;
                ws = new WebSocket(`ws://localhost:8000/api/async-tasks/ws/${userId}`);

                ws.onopen = function() {
                    addMessage('连接已建立');
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    addMessage('收到消息: ' + JSON.stringify(data, null, 2));
                };

                ws.onclose = function() {
                    addMessage('连接已关闭');
                };

                ws.onerror = function(error) {
                    addMessage('连接错误: ' + error);
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }

            function subscribeTask() {
                const taskId = document.getElementById('taskId').value;
                if (ws && taskId) {
                    ws.send(JSON.stringify({
                        type: 'subscribe_task',
                        task_id: taskId
                    }));
                    addMessage('已订阅任务: ' + taskId);
                }
            }

            function sendHeartbeat() {
                if (ws) {
                    ws.send(JSON.stringify({
                        type: 'heartbeat'
                    }));
                    addMessage('已发送心跳');
                }
            }

            function addMessage(message) {
                const messages = document.getElementById('messages');
                const div = document.createElement('div');
                div.textContent = new Date().toLocaleTimeString() + ': ' + message;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


# 任务管理器API
@router.get("/tasks/active")
async def get_active_tasks(current_user: User = Depends(get_current_user)):
    """
    获取活跃任务列表
    """
    try:
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()

        user_tasks = []
        if active_tasks:
            for worker_name, tasks in active_tasks.items():
                for task in tasks:
                    # 这里应该根据实际任务结构过滤用户任务
                    # 简化实现，返回所有活跃任务
                    user_tasks.append({
                        'id': task.get('id'),
                        'name': task.get('name'),
                        'args': task.get('args', []),
                        'worker': worker_name
                    })

        return {
            "success": True,
            "data": {
                "active_tasks": user_tasks,
                "count": len(user_tasks)
            }
        }

    except Exception as e:
        logger.error(f"获取活跃任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取活跃任务失败: {str(e)}")


@router.get("/tasks/history")
async def get_task_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """
    获取任务历史
    """
    try:
        # 从Redis获取任务历史
        history_keys = redis_client.keys("task_status:*")
        history_keys.sort(reverse=True)

        tasks = []
        for key in history_keys[offset:offset+limit]:
            try:
                task_data = redis_client.get(key)
                if task_data:
                    task_info = json.loads(task_data.decode('utf-8'))
                    tasks.append(task_info)
            except:
                continue

        return {
            "success": True,
            "data": {
                "tasks": tasks,
                "total": len(history_keys),
                "limit": limit,
                "offset": offset
            }
        }

    except Exception as e:
        logger.error(f"获取任务历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务历史失败: {str(e)}")


@router.post("/tasks/cleanup")
async def cleanup_expired_tasks(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    清理过期任务
    """
    try:
        # 启动清理任务
        from app.tasks.maintenance import cleanup_expired_results
        task = cleanup_expired_results.delay()

        return {
            "success": True,
            "data": {
                "task_id": task.id,
                "message": "清理任务已启动"
            }
        }

    except Exception as e:
        logger.error(f"启动清理任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动清理任务失败: {str(e)}")


@router.get("/system/metrics")
async def get_system_metrics(current_user: User = Depends(get_current_user)):
    """
    获取系统指标
    """
    try:
        # 启动指标收集任务
        from app.tasks.evaluation import collect_performance_metrics
        task = collect_performance_metrics.delay()

        # 等待短时间获取结果
        await asyncio.sleep(1)

        result = task.result if task.ready() else None

        return {
            "success": True,
            "data": {
                "task_id": task.id,
                "metrics": result,
                "message": "指标收集已启动"
            }
        }

    except Exception as e:
        logger.error(f"获取系统指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统指标失败: {str(e)}")