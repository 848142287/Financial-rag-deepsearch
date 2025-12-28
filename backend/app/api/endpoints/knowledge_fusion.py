"""
知识融合API接口
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ...core.database import get_db
from ...core.dependencies import get_current_user
from ...models.user import User
from ...services.knowledge_repair.content_cleaner import ContentCleaner
from ...services.knowledge_repair.cross_page_processor import CrossPageProcessor
from ...services.knowledge_repair.quality_detector import QualityDetector
from ...services.parsers.multimodal_parser import MultimodalParser
from ...services.semantic_fusion.semantic_fusion_engine import SemanticFusionEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-fusion", tags=["知识融合"])

# 全局变量存储处理状态（生产环境应使用Redis等）
fusion_tasks = {}


@router.post("/start")
async def start_fusion(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Any = Depends(get_db)
):
    """
    开始知识融合
    """
    try:
        # 创建任务ID
        task_id = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user.id}"

        # 初始化任务状态
        fusion_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "开始处理文件...",
            "result": None,
            "error": None
        }

        # 添加后台任务
        background_tasks.add_task(
            process_fusion_task,
            task_id,
            files,
            current_user.id
        )

        return JSONResponse({
            "task_id": task_id,
            "message": "知识融合任务已启动",
            "status": "processing"
        })

    except Exception as e:
        logger.error(f"启动知识融合失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.get("/status/{task_id}")
async def get_fusion_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取融合任务状态
    """
    if task_id not in fusion_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = fusion_tasks[task_id]

    return JSONResponse({
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "error": task.get("error"),
        "result": task.get("result") if task["status"] == "completed" else None
    })


@router.get("/result/{task_id}")
async def get_fusion_result(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取融合结果
    """
    if task_id not in fusion_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = fusion_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    return JSONResponse({
        "task_id": task_id,
        "result": task["result"],
        "statistics": task["result"].get("statistics", {}),
        "metadata": task["result"].get("metadata", {})
    })


async def process_fusion_task(
    task_id: str,
    files: List[UploadFile],
    user_id: str
):
    """
    处理知识融合任务（后台任务）
    """
    try:
        task = fusion_tasks[task_id]

        # 初始化服务
        config = {
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "chunk_size": 512,
            "overlap": 50
        }

        multimodal_parser = MultimodalParser(config)
        content_cleaner = ContentCleaner(config)
        cross_page_processor = CrossPageProcessor(config)
        quality_detector = QualityDetector(config)
        semantic_fusion_engine = SemanticFusionEngine(config)

        # 1. 解析文件
        task["message"] = "正在解析文件..."
        task["progress"] = 10

        parsed_contents = []
        for i, file in enumerate(files):
            # 保存文件
            file_content = await file.read()

            # 解析内容（这里简化处理，实际应该保存到临时文件）
            content = {
                "id": f"file_{i}",
                "name": file.filename,
                "type": file.content_type,
                "size": len(file_content),
                "text": file_content.decode('utf-8', errors='ignore'),  # 简化处理
                "multimodal_contents": []
            }

            # 使用多模态解析器
            try:
                parse_result = await multimodal_parser.parse(content["text"])
                content["multimodal_contents"] = parse_result.metadata.get("multimodal_contents", [])
                content["entities"] = parse_result.metadata.get("entities", [])
                content["relations"] = parse_result.metadata.get("relations", [])
            except Exception as e:
                logger.warning(f"解析文件 {file.filename} 失败: {str(e)}")

            parsed_contents.append(content)

            # 更新进度
            task["progress"] = 10 + (i + 1) * 20 / len(files)

        # 2. 内容清洗
        task["message"] = "正在清洗内容..."
        task["progress"] = 40

        cleaned_contents = []
        for content in parsed_contents:
            # 清洗多模态内容
            if content["multimodal_contents"]:
                cleaned_mm = await content_cleaner.clean_multimodal_content(content["multimodal_contents"])
                content["multimodal_contents"] = cleaned_mm

            # 清洗文本内容
            if content["text"]:
                cleaning_result = await content_cleaner.clean_content(content["text"])
                content["text"] = cleaning_result.cleaned_content
                content["quality_metadata"] = {
                    "quality_score": cleaning_result.quality_score,
                    "issues_found": cleaning_result.issues_found
                }

            cleaned_contents.append(content)

        # 3. 跨页处理
        task["message"] = "正在处理跨页内容..."
        task["progress"] = 60

        if len(cleaned_contents) > 1:
            # 模拟跨页内容处理
            for i, content in enumerate(cleaned_contents):
                content["page_number"] = i + 1

            references, merged_contents = await cross_page_processor.process_cross_page_content(cleaned_contents)

            # 添加合并后的内容
            for merged in merged_contents:
                cleaned_contents.append({
                    "id": merged.merged_id,
                    "name": f"合并内容_{merged.merged_id}",
                    "type": "merged",
                    "text": merged.merged_content,
                    "multimodal_contents": [],
                    "entities": [],
                    "relations": [],
                    "merge_metadata": merged.metadata
                })

        # 4. 质量检测
        task["message"] = "正在进行质量检测..."
        task["progress"] = 70

        quality_reports = []
        for content in cleaned_contents:
            quality_report = await quality_detector.detect_quality(
                content["text"],
                content.get("type", "text"),
                {"source": content["name"]}
            )
            quality_reports.append(quality_report)

            # 添加质量信息到内容
            content["quality_report"] = {
                "overall_score": quality_report.overall_score,
                "quality_level": quality_report.quality_level.value,
                "issues_count": len(quality_report.issues)
            }

        # 5. 语义融合
        task["message"] = "正在进行语义融合..."
        task["progress"] = 80

        fusion_result = await semantic_fusion_engine.fuse_knowledge(cleaned_contents)

        # 6. 完成任务
        task["message"] = "知识融合完成"
        task["progress"] = 100
        task["status"] = "completed"
        task["result"] = {
            "unified_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "content_type": chunk.content_type.value,
                    "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "metadata": chunk.metadata or {},
                    "entities": chunk.entities or [],
                    "relations": chunk.relations or [],
                    "confidence": chunk.confidence
                }
                for chunk in fusion_result.unified_chunks
            ],
            "statistics": fusion_result.statistics,
            "metadata": fusion_result.metadata,
            "quality_summary": {
                "avg_quality_score": sum(r.overall_score for r in quality_reports) / len(quality_reports),
                "total_issues": sum(len(r.issues) for r in quality_reports),
                "quality_distribution": {
                    level: sum(1 for r in quality_reports if r.quality_level.value == level)
                    for level in ["excellent", "good", "fair", "poor"]
                }
            }
        }

        logger.info(f"任务 {task_id} 完成")

    except Exception as e:
        logger.error(f"处理任务 {task_id} 失败: {str(e)}")
        task["status"] = "failed"
        task["error"] = str(e)
        task["message"] = f"处理失败: {str(e)}"


@router.get("/models")
async def get_fusion_models(
    current_user: User = Depends(get_current_user)
):
    """
    获取可用的融合模型
    """
    models = {
        "parsers": {
            "multimodal": {
                "name": "多模态解析器",
                "supported_formats": [
                    "pdf", "docx", "xlsx", "txt", "md",
                    "png", "jpg", "jpeg", "tiff"
                ],
                "features": [
                    "文本解析", "图片分析", "表格提取", "公式识别"
                ]
            }
        },
        "embeddings": {
            "sentence_transformers": {
                "name": "Sentence Transformers",
                "models": [
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "sentence-transformers/all-MiniLM-L6-v2"
                ],
                "dimensions": 384
            }
        },
        "quality_detectors": {
            "completeness": "完整性检测",
            "accuracy": "准确性检测",
            "consistency": "一致性检测",
            "clarity": "清晰度检测"
        }
    }

    return JSONResponse(models)


@router.post("/customize")
async def customize_fusion_config(
    config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    自定义融合配置
    """
    try:
        # 验证配置
        required_fields = ["chunk_size", "overlap", "embedding_model"]
        for field in required_fields:
            if field not in config:
                raise HTTPException(status_code=400, detail=f"缺少必需字段: {field}")

        # 保存配置（这里简化处理，实际应该保存到数据库）
        user_config = {
            "user_id": current_user.id,
            "config": config,
            "created_at": datetime.now()
        }

        return JSONResponse({
            "message": "配置保存成功",
            "config": user_config
        })

    except Exception as e:
        logger.error(f"保存配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")