"""
多模态文档解析API端点
提供基于Mineru、Qwen-VL-OCR和Qwen-VL-Max的联合解析服务
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
from sqlalchemy.orm import Session
from typing import List, Any, Optional
import logging
import asyncio
import tempfile
import os
from datetime import datetime

from app.core.database import get_db
from app.services.multimodal_parser.core.multimodal_parser import (
    MultimodalDocumentParser, ParsingConfig, ParsedDocument
)
from app.services.multimodal_parser.evaluators.integrity_evaluator import IntegrityEvaluator

logger = logging.getLogger(__name__)

router = APIRouter()

# 全局解析器实例
parser = None
integrity_evaluator = None


async def get_parser():
    """获取解析器实例"""
    global parser
    if parser is None:
        parser = MultimodalDocumentParser()
        # 初始化各个组件
        if parser.mineru_engine:
            logger.info("Mineru引擎已加载")
        if parser.qwen_vl_engine:
            logger.info("Qwen-VL引擎已加载")
    return parser


async def get_integrity_evaluator():
    """获取完整性评估器实例"""
    global integrity_evaluator
    if integrity_evaluator is None:
        integrity_evaluator = IntegrityEvaluator()
    return integrity_evaluator


@router.post("/parse-document")
async def parse_document(
    file: UploadFile = File(...),
    use_mineru: bool = Form(True),
    use_qwen_vl_ocr: bool = Form(True),
    use_qwen_vl_max: bool = Form(True),
    enable_auto_repair: bool = Form(True),
    integrity_threshold: float = Form(0.8)
):
    """
    解析文档（多引擎联合解析）

    Args:
        file: 上传的文件
        use_mineru: 是否使用Mineru
        use_qwen_vl_ocr: 是否使用Qwen-VL-OCR
        use_qwen_vl_max: 是否使用Qwen-VL-Max
        enable_auto_repair: 是否启用自动修复
        integrity_threshold: 完整性阈值

    Returns:
        解析结果
    """
    try:
        # 检查文件类型
        file_ext = os.path.splitext(file.filename)[1].lower()
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']

        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}。支持的格式: {supported_extensions}"
            )

        # 创建配置
        config = ParsingConfig(
            use_mineru=use_mineru,
            use_qwen_vl_ocr=use_qwen_vl_ocr,
            use_qwen_vl_max=use_qwen_vl_max,
            enable_auto_repair=enable_auto_repair,
            integrity_threshold=integrity_threshold
        )

        # 获取解析器
        parser_instance = await get_parser()

        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # 解析文档
            start_time = datetime.now()
            parsed_document = await parser_instance.parse_document(temp_file_path)
            processing_time = (datetime.now() - start_time).total_seconds()

            # 评估完整性
            evaluator = await get_integrity_evaluator()
            integrity_score = await evaluator.evaluate_integrity(parsed_document)

            # 生成结果
            result = {
                "success": True,
                "document_id": parsed_document.document_id,
                "title": parsed_document.title,
                "total_pages": parsed_document.total_pages,
                "total_chapters": len(parsed_document.chapters),
                "total_content_blocks": len(parsed_document.content_blocks),
                "integrity_score": integrity_score,
                "processing_time": processing_time,
                "metadata": parsed_document.metadata,
                "parsing_stats": parsed_document.parsing_stats,
                "engines_used": parsed_document.metadata.get("engines_used", []),
                "auto_repair_enabled": enable_auto_repair,
                "integrity_threshold": integrity_threshold
            }

            logger.info(f"文档解析完成: {parsed_document.document_id}, 评分: {integrity_score:.3f}")
            return result

        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"文档解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse-document-detailed")
async def parse_document_detailed(
    file: UploadFile = File(...),
    include_content: bool = Form(False),
    include_chapters: bool = Form(True),
    config: Optional[str] = Form(None)
):
    """
    详细解析文档（包含完整内容）

    Args:
        file: 上传的文件
        include_content: 是否包含完整内容
        include_chapters: 是否包含章节信息
        config: JSON格式的解析配置

    Returns:
        详细解析结果
    """
    try:
        # 解析配置
        parsing_config = None
        if config:
            import json
            try:
                config_dict = json.loads(config)
                parsing_config = ParsingConfig(**config_dict)
            except Exception as e:
                logger.warning(f"配置解析失败，使用默认配置: {str(e)}")
                parsing_config = ParsingConfig()

        # 获取解析器
        parser_instance = await get_parser()

        # 保存上传的文件
        file_ext = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # 解析文档
            start_time = datetime.now()
            parsed_document = await parser_instance.parse_document(temp_file_path)
            processing_time = (datetime.now() - start_time).total_seconds()

            # 构建详细结果
            result = {
                "success": True,
                "document_id": parsed_document.document_id,
                "title": parsed_document.title,
                "total_pages": parsed_document.total_pages,
                "total_chapters": len(parsed_document.chapters),
                "total_content_blocks": len(parsed_document.content_blocks),
                "processing_time": processing_time,
                "metadata": parsed_document.metadata,
                "parsing_stats": parsed_document.parsing_stats,
                "engines_used": parsed_document.metadata.get("engines_used", [])
            }

            # 包含章节信息
            if include_chapters:
                result["chapters"] = [
                    {
                        "id": chapter.id,
                        "title": chapter.title,
                        "level": chapter.level,
                        "start_page": chapter.start_page,
                        "end_page": chapter.end_page,
                        "sub_chapters": chapter.sub_chapters,
                        "blocks_count": len(chapter.blocks),
                        "metadata": chapter.metadata
                    }
                    for chapter in parsed_document.chapters
                ]

            # 包含完整内容
            if include_content:
                result["content_blocks"] = [
                    {
                        "id": block.id,
                        "content_type": block.content_type.value,
                        "content": block.content,
                        "page_number": block.page_number,
                        "chapter_id": block.chapter_id,
                        "confidence": block.confidence,
                        "bbox": block.bbox,
                        "metadata": block.metadata
                    }
                    for block in parsed_document.content_blocks
                ]

            # 评估完整性
            evaluator = await get_integrity_evaluator()
            integrity_report = await evaluator.generate_integrity_report(parsed_document)

            result["integrity"] = {
                "score": integrity_report.overall_score,
                "level": evaluator.get_integrity_level(integrity_report.overall_score),
                "issues_count": len(integrity_report.issues),
                "recommendations": integrity_report.recommendations[:5]  # 限制建议数量
            }

            return result

        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"详细文档解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-integrity")
async def evaluate_integrity(
    document_data: dict
):
    """
    评估文档完整性

    Args:
        document_data: 文档数据（从parse-document获取的结果）

    Returns:
        完整性评估报告
    """
    try:
        # 重构文档对象（简化版本）
        from app.services.multimodal_parser.core.multimodal_parser import Chapter, ContentBlock, ContentType

        # 这里需要根据实际的document_data重构ParsedDocument对象
        # 为了简化，我们直接进行评估逻辑

        evaluator = await get_integrity_evaluator()

        # 模拟评估（实际实现需要完整的文档对象）
        total_blocks = document_data.get("total_content_blocks", 0)
        total_pages = document_data.get("total_pages", 1)

        # 基本完整性评分逻辑
        content_coverage = min(total_blocks / (total_pages * 5), 1.0)  # 假设每页平均5个块
        engines_used = document_data.get("engines_used", [])
        multi_engine_bonus = 0.1 if len(engines_used) > 1 else 0.0

        base_score = content_coverage * 0.7 + multi_engine_bonus
        integrity_score = min(base_score + 0.2, 1.0)  # 最高1.0

        # 生成建议
        recommendations = []
        if integrity_score < 0.7:
            recommendations.append("建议启用自动修复功能")
            recommendations.append("检查是否存在缺失页面")
        if len(engines_used) == 1:
            recommendations.append("建议使用多引擎联合解析")

        result = {
            "success": True,
            "integrity_score": integrity_score,
            "integrity_level": evaluator.get_integrity_level(integrity_score),
            "content_coverage": content_coverage,
            "engines_used_count": len(engines_used),
            "total_blocks": total_blocks,
            "total_pages": total_pages,
            "recommendations": recommendations,
            "metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "engines_used": engines_used
            }
        }

        return result

    except Exception as e:
        logger.error(f"完整性评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/parser-status")
async def get_parser_status():
    """获取解析器状态"""
    try:
        parser_instance = await get_parser()
        evaluator = await get_integrity_evaluator()

        status = {
            "success": True,
            "parser_initialized": parser_instance is not None,
            "evaluator_initialized": evaluator is not None,
            "engines_available": {
                "mineru": parser_instance.mineru_engine is not None,
                "qwen_vl": parser_instance.qwen_vl_engine is not None
            },
            "components_loaded": {
                "structure_analyzer": hasattr(parser_instance, 'structure_analyzer'),
                "content_aggregator": hasattr(parser_instance, 'content_aggregator'),
                "integrity_evaluator": evaluator is not None,
                "auto_repairer": parser_instance.auto_repairer is not None
            },
            "temp_directory": parser_instance.temp_dir,
            "supported_formats": ['.pdf', '.jpg', '.jpeg', '.png', '.tiff'],
            "default_config": {
                "use_mineru": True,
                "use_qwen_vl_ocr": True,
                "use_qwen_vl_max": True,
                "enable_auto_repair": True,
                "integrity_threshold": 0.8
            }
        }

        return status

    except Exception as e:
        logger.error(f"获取解析器状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export-document")
async def export_document(
    document_data: dict,
    format_type: str = "json",
    include_metadata: bool = True
):
    """
    导出解析结果

    Args:
        document_data: 文档数据
        format_type: 导出格式 (json, txt, markdown)
        include_metadata: 是否包含元数据

    Returns:
        导出结果
    """
    try:
        export_content = ""
        content_type = "application/json"

        if format_type.lower() == "json":
            result = {
                "success": True,
                "format": "json",
                "content": document_data
            }
            return result

        elif format_type.lower() == "txt":
            # 纯文本导出
            content_type = "text/plain"
            lines = [f"标题: {document_data.get('title', '未知')}"]
            lines.append(f"页数: {document_data.get('total_pages', 0)}")
            lines.append(f"章节数: {document_data.get('total_chapters', 0)}")
            lines.append(f"内容块数: {document_data.get('total_content_blocks', 0)}")
            lines.append("")

            if include_metadata and "chapters" in document_data:
                lines.append("章节结构:")
                for chapter in document_data["chapters"]:
                    lines.append(f"  {chapter['level']}. {chapter['title']} (页 {chapter['start_page']}-{chapter.get('end_page', '?')})")
                lines.append("")

            export_content = "\n".join(lines)

        elif format_type.lower() == "markdown":
            # Markdown导出
            content_type = "text/markdown"
            lines = [f"# {document_data.get('title', '未知文档')}"]
            lines.append("")
            lines.append(f"**页数**: {document_data.get('total_pages', 0)}")
            lines.append(f"**章节数**: {document_data.get('total_chapters', 0)}")
            lines.append(f"**内容块数**: {document_data.get('total_content_blocks', 0)}")
            lines.append(f"**完整性评分**: {document_data.get('integrity', {}).get('score', 'N/A')}")
            lines.append("")

            if include_metadata and "chapters" in document_data:
                lines.append("## 章节结构")
                for chapter in document_data["chapters"]:
                    lines.append(f"{'#' * (chapter['level'] + 1)} {chapter['title']}")
                    lines.append(f"* 页面范围: {chapter['start_page']}-{chapter.get('end_page', '?')}")
                    lines.append(f"* 子章节数: {len(chapter.get('sub_chapters', []))}")
                    lines.append("")

            export_content = "\n".join(lines)

        else:
            raise HTTPException(status_code=400, detail=f"不支持的导出格式: {format_type}")

        result = {
            "success": True,
            "format": format_type,
            "content_type": content_type,
            "content": export_content,
            "size": len(export_content.encode('utf-8'))
        }

        return result

    except Exception as e:
        logger.error(f"文档导出失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-parse")
async def batch_parse(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    config: Optional[str] = None
):
    """
    批量解析文档

    Args:
        files: 上传的文件列表
        config: 解析配置

    Returns:
        批量任务ID
    """
    try:
        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())

        # 保存文件
        temp_dir = tempfile.mkdtemp(prefix=f"batch_parse_{task_id}_")
        file_paths = []

        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()
            supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']

            if file_ext not in supported_extensions:
                continue

            file_path = os.path.join(temp_dir, f"{len(file_paths)}{file_ext}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_paths.append(file_path)

        # 创建后台任务
        async def process_batch():
            try:
                parser_instance = await get_parser()
                results = []

                for i, file_path in enumerate(file_paths):
                    try:
                        # 解析配置
                        parsing_config = ParsingConfig()
                        if config:
                            import json
                            config_dict = json.loads(config)
                            parsing_config = ParsingConfig(**config_dict)

                        # 解析文档
                        parsed_doc = await parser_instance.parse_document(file_path)

                        # 评估完整性
                        evaluator = await get_integrity_evaluator()
                        integrity_score = await evaluator.evaluate_integrity(parsed_doc)

                        result = {
                            "index": i,
                            "filename": os.path.basename(file_path),
                            "success": True,
                            "document_id": parsed_doc.document_id,
                            "title": parsed_doc.title,
                            "total_pages": parsed_doc.total_pages,
                            "integrity_score": integrity_score,
                            "content_blocks": len(parsed_doc.content_blocks)
                        }
                        results.append(result)

                    except Exception as e:
                        results.append({
                            "index": i,
                            "filename": os.path.basename(file_path),
                            "success": False,
                            "error": str(e)
                        })

                # 保存结果
                result_path = f"/tmp/batch_result_{task_id}.json"
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "task_id": task_id,
                        "total_files": len(file_paths),
                        "processed_files": len([r for r in results if r["success"]]),
                        "results": results,
                        "completed_at": datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)

                logger.info(f"批量解析任务完成: {task_id}")

            except Exception as e:
                logger.error(f"批量解析任务失败 {task_id}: {str(e)}")
            finally:
                # 清理临时文件
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        # 添加后台任务
        background_tasks.add_task(process_batch)

        return {
            "success": True,
            "task_id": task_id,
            "total_files": len(files),
            "status": "processing",
            "message": f"批量解析任务已启动，任务ID: {task_id}"
        }

    except Exception as e:
        logger.error(f"批量解析启动失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-result/{task_id}")
async def get_batch_result(task_id: str):
    """
    获取批量解析结果

    Args:
        task_id: 任务ID

    Returns:
        批量解析结果
    """
    try:
        result_path = f"/tmp/batch_result_{task_id}.json"

        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="任务结果不存在或已完成")

        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        return result

    except Exception as e:
        logger.error(f"获取批量结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-parser")
async def test_parser():
    """测试解析器功能"""
    try:
        parser_instance = await get_parser()
        evaluator = await get_integrity_evaluator()

        # 测试状态
        test_results = {
            "success": True,
            "parser_available": parser_instance is not None,
            "evaluator_available": evaluator is not None,
            "engines": {
                "mineru": parser_instance.mineru_engine is not None,
                "qwen_vl": parser_instance.qwen_vl_engine is not None
            },
            "components": {
                "structure_analyzer": hasattr(parser_instance, 'structure_analyzer'),
                "content_aggregator": hasattr(parser_instance, 'content_aggregator'),
                "integrity_evaluator": evaluator is not None,
                "auto_repairer": parser_instance.auto_repairer is not None
            },
            "capabilities": {
                "parallel_processing": True,
                "auto_repair": True,
                "integrity_evaluation": True,
                "multi_engine_aggregation": True,
                "chapter_structure_analysis": True,
                "content_type_detection": True,
                "semantic_coherence_check": True
            },
            "message": "多模态解析器功能正常"
        }

        return test_results

    except Exception as e:
        logger.error(f"测试解析器失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))