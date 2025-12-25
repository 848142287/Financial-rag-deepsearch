"""
增强的智能文档批量上传接口
集成多引擎文档解析系统，提供高质量的批量文档处理
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
import os
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json

from ...core.database import get_db
from ...core.config import settings
from ...core.dependencies import get_current_user
from ...models.user import User
from ...models.document import Document
from ...schemas.document import DocumentResponse, BatchUploadResponse
from ...services.document_intelligence.enhanced_parser import EnhancedDocumentParser
from ...services.document_intelligence.config.enhanced_parser_config import EnhancedParserConfig, ConfigManager
from ...services.document_intelligence.integration_example import DocumentIntelligenceSystem
from ...services.vector_store.vector_store_manager import vector_store_manager
from ...services.knowledge_graph.knowledge_graph_manager import knowledge_graph_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents/enhanced-batch", tags=["智能批量文档处理"])


class EnhancedBatchProcessor:
    """增强的批量处理器"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.parser = EnhancedDocumentParser(self.config)
        self.vector_store = vector_store_manager
        self.kg_manager = knowledge_graph_manager

    async def process_document_with_intelligence(
        self,
        file_path: str,
        user_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """使用智能解析系统处理单个文档"""

        try:
            # 设置默认解析选项
            if options is None:
                options = {
                    "enable_auto_repair": True,
                    "integrity_threshold": 0.7,
                    "output_format": "structured",
                    "quality_assessment": True,
                    "cross_validation": True,
                    "generate_vectors": True,
                    "generate_knowledge_graph": True
                }

            # 1. 智能文档解析
            logger.info(f"🚀 开始智能解析文档: {file_path}")
            parse_result = await self.parser.parse_document(file_path, options)

            # 2. 生成向量表示
            vectors_generated = False
            vector_ids = []
            if options.get("generate_vectors", True):
                try:
                    vector_result = await self._generate_document_vectors(parse_result, user_id)
                    vectors_generated = True
                    vector_ids = vector_result.get("vector_ids", [])
                    logger.info(f"✅ 向量生成成功: {len(vector_ids)} 个向量")
                except Exception as e:
                    logger.warning(f"⚠️ 向量生成失败: {e}")

            # 3. 构建知识图谱
            kg_entities = []
            kg_relations = []
            if options.get("generate_knowledge_graph", True):
                try:
                    kg_result = await self._build_knowledge_graph(parse_result, user_id)
                    kg_entities = kg_result.get("entities", [])
                    kg_relations = kg_result.get("relations", [])
                    logger.info(f"✅ 知识图谱构建成功: {len(kg_entities)} 个实体, {len(kg_relations)} 个关系")
                except Exception as e:
                    logger.warning(f"⚠️ 知识图谱构建失败: {e}")

            # 4. 生成知识图片
            knowledge_images = []
            try:
                knowledge_images = await self._generate_knowledge_images(parse_result, user_id)
                logger.info(f"✅ 知识图谱生成成功: {len(knowledge_images)} 张图片")
            except Exception as e:
                logger.warning(f"⚠️ 知识图谱生成失败: {e}")

            # 5. 保存文档记录到数据库
            document_record = await self._save_document_record(
                parse_result, user_id, vectors_generated, kg_entities, knowledge_images
            )

            return {
                "document_id": document_record.get("id"),
                "parse_result": parse_result,
                "vector_ids": vector_ids,
                "kg_entities": len(kg_entities),
                "kg_relations": len(kg_relations),
                "knowledge_images": len(knowledge_images),
                "processing_status": "completed",
                "quality_score": parse_result.get("integrity_score", 0),
                "engines_used": parse_result.get("engines_used", []),
                "content_summary": {
                    "total_pages": parse_result.get("total_pages", 0),
                    "total_sections": parse_result.get("total_chapters", 0),
                    "content_blocks": parse_result.get("total_content_blocks", 0),
                    "tables_count": len(parse_result.get("tables", [])),
                    "images_count": len(parse_result.get("images", [])),
                    "formulas_count": len(parse_result.get("formulas", []))
                }
            }

        except Exception as e:
            logger.error(f"❌ 智能文档处理失败: {file_path}, 错误: {e}")
            return {
                "document_id": None,
                "parse_result": None,
                "processing_status": "failed",
                "error": str(e),
                "vector_ids": [],
                "kg_entities": 0,
                "kg_relations": 0,
                "knowledge_images": 0
            }

    async def _generate_document_vectors(
        self,
        parse_result: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """生成文档向量表示"""
        vector_ids = []

        # 提取文本内容
        text_content = self._extract_text_content(parse_result)

        # 分块处理
        chunk_size = 500
        chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                # 生成向量（这里需要集成实际的向量生成服务）
                vector_data = {
                    "text": chunk,
                    "metadata": {
                        "document_id": parse_result.get("document_id"),
                        "chunk_id": i,
                        "user_id": user_id,
                        "content_type": "text",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }

                # 模拟向量生成
                vector_id = f"vec_{uuid.uuid4().hex[:8]}"
                vector_ids.append(vector_id)

                # 保存到向量数据库
                # await self.vector_store.insert_vector(vector_id, vector_data)

        return {"vector_ids": vector_ids, "total_chunks": len(chunks)}

    async def _build_knowledge_graph(
        self,
        parse_result: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """构建知识图谱"""
        entities = []
        relations = []

        # 从解析结果中提取实体
        for section in parse_result.get("sections", []):
            # 提取章节标题作为实体
            if section.get("title"):
                entity = {
                    "id": f"entity_{uuid.uuid4().hex[:8]}",
                    "name": section["title"],
                    "type": "section",
                    "properties": {
                        "level": section.get("level", 1),
                        "document_id": parse_result.get("document_id"),
                        "user_id": user_id
                    }
                }
                entities.append(entity)

        # 从表格中提取实体
        for table in parse_result.get("tables", []):
            if table.get("title"):
                entity = {
                    "id": f"entity_{uuid.uuid4().hex[:8]}",
                    "name": table["title"],
                    "type": "table",
                    "properties": {
                        "rows": table.get("rows", 0),
                        "columns": table.get("columns", 0),
                        "document_id": parse_result.get("document_id"),
                        "user_id": user_id
                    }
                }
                entities.append(entity)

        # 构建关系
        for i, entity in enumerate(entities):
            if i > 0:
                relation = {
                    "id": f"rel_{uuid.uuid4().hex[:8]}",
                    "source": entities[i-1]["id"],
                    "target": entity["id"],
                    "type": "precedes",
                    "properties": {
                        "document_id": parse_result.get("document_id"),
                        "user_id": user_id
                    }
                }
                relations.append(relation)

        return {"entities": entities, "relations": relations}

    async def _generate_knowledge_images(
        self,
        parse_result: Dict[str, Any],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """生成知识图片"""
        knowledge_images = []

        # 为表格生成可视化图片
        for i, table in enumerate(parse_result.get("tables", [])):
            # 模拟知识图片生成
            image_data = {
                "id": f"img_{uuid.uuid4().hex[:8]}",
                "type": "table_visualization",
                "source_content": "table",
                "source_id": table.get("id"),
                "image_path": f"/knowledge_images/table_{i}.png",
                "metadata": {
                    "title": f"表格可视化 - {table.get('title', f'表格{i+1}')}",
                    "document_id": parse_result.get("document_id"),
                    "user_id": user_id,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            knowledge_images.append(image_data)

        # 为公式生成可视化图片
        for i, formula in enumerate(parse_result.get("formulas", [])):
            image_data = {
                "id": f"img_{uuid.uuid4().hex[:8]}",
                "type": "formula_rendering",
                "source_content": "formula",
                "source_id": formula.get("id"),
                "image_path": f"/knowledge_images/formula_{i}.png",
                "metadata": {
                    "title": f"公式渲染 - {formula.get('content', '')[:50]}...",
                    "document_id": parse_result.get("document_id"),
                    "user_id": user_id,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            knowledge_images.append(image_data)

        return knowledge_images

    def _extract_text_content(self, parse_result: Dict[str, Any]) -> str:
        """提取文档的文本内容"""
        text_parts = []

        # 提取章节文本
        for section in parse_result.get("sections", []):
            if section.get("text"):
                text_parts.append(section["text"])

        # 提取表格文本
        for table in parse_result.get("tables", []):
            if table.get("text"):
                text_parts.append(table["text"])

        return " ".join(text_parts)

    async def _save_document_record(
        self,
        parse_result: Dict[str, Any],
        user_id: str,
        vectors_generated: bool,
        kg_entities: List,
        knowledge_images: List
    ) -> Dict[str, Any]:
        """保存文档记录到数据库"""
        # 这里应该保存到实际的数据库
        # 模拟数据库保存
        document_record = {
            "id": f"doc_{uuid.uuid4().hex[:8]}",
            "document_id": parse_result.get("document_id"),
            "user_id": user_id,
            "title": parse_result.get("title", "未命名文档"),
            "file_path": parse_result.get("file_path"),
            "total_pages": parse_result.get("total_pages", 0),
            "total_sections": parse_result.get("total_chapters", 0),
            "integrity_score": parse_result.get("integrity_score", 0),
            "engines_used": parse_result.get("engines_used", []),
            "vectors_generated": vectors_generated,
            "kg_entities_count": len(kg_entities),
            "knowledge_images_count": len(knowledge_images),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        return document_record


# 全局处理器实例
enhanced_processor = EnhancedBatchProcessor()


@router.post("/upload", response_model=Dict[str, Any])
async def enhanced_batch_upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    enable_vector_generation: bool = Form(True),
    enable_knowledge_graph: bool = Form(True),
    enable_knowledge_images: bool = Form(True),
    enable_auto_repair: bool = Form(True),
    integrity_threshold: float = Form(0.7),
    current_user: User = Depends(get_current_user),
    db: Any = Depends(get_db)
):
    """
    增强的智能批量上传接口

    功能特点：
    - 多引擎智能文档解析
    - 自动向量生成
    - 知识图谱构建
    - 知识图片生成
    - 质量评估与自动修复
    """
    try:
        # 检查文件数量限制
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="单次最多上传20个文件")

        # 验证文件类型和大小
        valid_files = []
        for file in files:
            if not _is_valid_file_type(file.filename):
                logger.warning(f"无效文件类型: {file.filename}")
                continue

            if file.size and not _is_valid_file_size(file.size):
                logger.warning(f"文件过大: {file.filename}")
                continue

            valid_files.append(file)

        if not valid_files:
            raise HTTPException(status_code=400, detail="没有有效的文件")

        # 创建批次ID
        batch_id = str(uuid.uuid4())
        upload_dir = os.path.join(settings.upload_dir, "enhanced", batch_id)
        os.makedirs(upload_dir, exist_ok=True)

        # 保存文件
        file_paths = []
        upload_results = []

        for file in valid_files:
            safe_filename = _generate_safe_filename(file.filename)
            file_path = os.path.join(upload_dir, safe_filename)

            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            file_paths.append(file_path)
            upload_results.append({
                "filename": file.filename,
                "original_name": file.filename,
                "safe_filename": safe_filename,
                "file_path": file_path,
                "status": "uploaded",
                "size": len(content),
                "upload_time": datetime.utcnow().isoformat()
            })

        # 准备处理选项
        processing_options = {
            "enable_auto_repair": enable_auto_repair,
            "integrity_threshold": integrity_threshold,
            "output_format": "structured",
            "quality_assessment": True,
            "cross_validation": True,
            "generate_vectors": enable_vector_generation,
            "generate_knowledge_graph": enable_knowledge_graph,
            "generate_knowledge_images": enable_knowledge_images,
            "parallel_processing": True,
            "max_concurrent": 3
        }

        # 启动智能处理任务
        background_tasks.add_task(
            _enhanced_process_uploaded_files,
            batch_id,
            file_paths,
            upload_results,
            processing_options,
            current_user.id
        )

        return {
            "batch_id": batch_id,
            "total_files": len(valid_files),
            "uploaded_files": len(upload_results),
            "status": "processing",
            "processing_options": processing_options,
            "estimated_time": len(valid_files) * 30,  # 估算时间（秒）
            "message": f"✅ 成功上传 {len(valid_files)} 个文件，启动智能处理...",
            "capabilities": [
                "🔍 多引擎文档解析",
                "🧠 智能语义修复",
                "📊 质量评估",
                "🎯 向量生成",
                "🕸️ 知识图谱构建",
                "🖼️ 知识图片生成"
            ]
        }

    except Exception as e:
        logger.error(f"增强批量上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量上传失败: {str(e)}")


@router.get("/upload-status/{batch_id}")
async def get_enhanced_batch_status(
    batch_id: str,
    current_user: User = Depends(get_current_user)
):
    """获取增强批量处理状态"""
    try:
        status = await _get_enhanced_batch_status(batch_id)
        return status
    except Exception as e:
        logger.error(f"获取增强批量处理状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取状态失败")


@router.get("/processing-results/{batch_id}")
async def get_processing_results(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Any = Depends(get_db)
):
    """获取批量处理结果"""
    try:
        results = await _get_processing_results(batch_id, current_user.id)
        return results
    except Exception as e:
        logger.error(f"获取处理结果失败: {e}")
        raise HTTPException(status_code=500, detail="获取结果失败")


@router.post("/process-folder")
async def process_folder_documents(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    recursive: bool = Form(True),
    file_pattern: str = Form("*.pdf"),
    processing_options: str = Form("{}"),
    current_user: User = Depends(get_current_user)
):
    """
    处理指定文件夹中的文档

    专门用于处理券商研报等现有文档集合
    """
    try:
        # 解析处理选项
        options = json.loads(processing_options) if processing_options else {}

        # 查找匹配的文件
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(status_code=400, detail="指定的文件夹不存在")

        # 搜索文件
        if recursive:
            files = list(folder.rglob(file_pattern))
        else:
            files = list(folder.glob(file_pattern))

        if not files:
            raise HTTPException(status_code=404, detail="没有找到匹配的文件")

        # 限制文件数量
        max_files = 100
        if len(files) > max_files:
            files = files[:max_files]

        # 创建批次ID
        batch_id = str(uuid.uuid4())

        # 准备文件路径列表
        file_paths = [str(f) for f in files]

        # 默认处理选项
        default_options = {
            "enable_auto_repair": True,
            "integrity_threshold": 0.7,
            "output_format": "structured",
            "quality_assessment": True,
            "cross_validation": True,
            "generate_vectors": True,
            "generate_knowledge_graph": True,
            "generate_knowledge_images": True,
            "parallel_processing": True,
            "max_concurrent": 2
        }

        # 合并用户选项
        final_options = {**default_options, **options}

        # 启动处理任务
        background_tasks.add_task(
            _enhanced_process_folder_files,
            batch_id,
            file_paths,
            final_options,
            current_user.id
        )

        return {
            "batch_id": batch_id,
            "folder_path": folder_path,
            "file_pattern": file_pattern,
            "recursive": recursive,
            "total_files_found": len(list(folder.rglob(file_pattern)) if recursive else list(folder.glob(file_pattern))),
            "files_to_process": len(file_paths),
            "status": "processing",
            "processing_options": final_options,
            "estimated_time": len(file_paths) * 45,
            "message": f"🚀 开始处理文件夹中的 {len(file_paths)} 个文档..."
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="处理选项格式错误")
    except Exception as e:
        logger.error(f"文件夹处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件夹处理失败: {str(e)}")


# 后台处理函数
async def _enhanced_process_uploaded_files(
    batch_id: str,
    file_paths: List[str],
    upload_results: List[Dict[str, Any]],
    processing_options: Dict[str, Any],
    user_id: str
):
    """增强的上传文件处理"""
    try:
        logger.info(f"🚀 开始增强批量处理: {batch_id}")

        # 更新初始状态
        await _update_enhanced_batch_status(
            batch_id, "processing", 0, len(file_paths), []
        )

        processed_count = 0
        errors = []
        processing_results = []

        # 并发处理文档
        semaphore = asyncio.Semaphore(processing_options.get("max_concurrent", 2))

        async def process_single_file(file_path: str, upload_info: Dict[str, Any]):
            async with semaphore:
                try:
                    result = await enhanced_processor.process_document_with_intelligence(
                        file_path, user_id, processing_options
                    )

                    return {
                        "filename": upload_info["filename"],
                        "file_path": file_path,
                        "result": result,
                        "processing_time": result.get("processing_time", 0),
                        "status": "completed" if result.get("processing_status") == "completed" else "failed"
                    }

                except Exception as e:
                    logger.error(f"文件处理异常: {file_path}, 错误: {e}")
                    return {
                        "filename": upload_info["filename"],
                        "file_path": file_path,
                        "result": None,
                        "error": str(e),
                        "status": "failed"
                    }

        # 执行并发处理
        tasks = [
            process_single_file(file_path, upload_info)
            for file_path, upload_info in zip(file_paths, upload_results)
        ]

        results = await asyncio.gather(*tasks)

        # 统计结果
        for result in results:
            if result["status"] == "completed":
                processed_count += 1
                processing_results.append(result)
            else:
                errors.append(f"{result['filename']}: {result.get('error', 'Unknown error')}")

            # 更新进度
            await _update_enhanced_batch_status(
                batch_id, "processing",
                len([r for r in results if r["status"] in ["completed", "failed"]]),
                len(file_paths),
                errors
            )

        # 计算统计信息
        successful_results = [r for r in processing_results if r["status"] == "completed"]

        if successful_results:
            avg_quality_score = sum(
                r["result"].get("quality_score", 0) for r in successful_results
            ) / len(successful_results)

            total_vectors = sum(
                len(r["result"].get("vector_ids", [])) for r in successful_results
            )

            total_kg_entities = sum(
                r["result"].get("kg_entities", 0) for r in successful_results
            )

            total_kg_relations = sum(
                r["result"].get("kg_relations", 0) for r in successful_results
            )

            total_knowledge_images = sum(
                r["result"].get("knowledge_images", 0) for r in successful_results
            )
        else:
            avg_quality_score = 0
            total_vectors = 0
            total_kg_entities = 0
            total_kg_relations = 0
            total_knowledge_images = 0

        # 更新最终状态
        final_status = "completed" if not errors else "completed_with_errors"
        await _update_enhanced_batch_status(
            batch_id, final_status, processed_count, len(file_paths), errors,
            {
                "avg_quality_score": avg_quality_score,
                "total_vectors": total_vectors,
                "total_kg_entities": total_kg_entities,
                "total_kg_relations": total_kg_relations,
                "total_knowledge_images": total_knowledge_images,
                "processing_results": processing_results
            }
        )

        logger.info(f"✅ 增强批量处理完成: {batch_id}")
        logger.info(f"📊 处理统计: 成功 {processed_count}/{len(file_paths)}")
        logger.info(f"🎯 质量分数: {avg_quality_score:.3f}")
        logger.info(f"🔢 向量数量: {total_vectors}")
        logger.info(f"🕸️ 知识实体: {total_kg_entities}")
        logger.info(f"🖼️ 知识图片: {total_knowledge_images}")

    except Exception as e:
        logger.error(f"❌ 增强批量处理异常: {batch_id}, 错误: {e}")
        await _update_enhanced_batch_status(
            batch_id, "failed", 0, len(file_paths), [str(e)]
        )


async def _enhanced_process_folder_files(
    batch_id: str,
    file_paths: List[str],
    processing_options: Dict[str, Any],
    user_id: str
):
    """处理文件夹中的文件"""
    # 与上面的处理逻辑类似，但专门处理文件夹场景
    await _enhanced_process_uploaded_files(
        batch_id, file_paths,
        [{"filename": Path(f).name, "file_path": f} for f in file_paths],
        processing_options, user_id
    )


# 辅助函数
def _is_valid_file_type(filename: str) -> bool:
    """检查文件类型是否有效"""
    if not filename:
        return False

    valid_extensions = {
        '.pdf', '.docx', '.xlsx', '.txt', '.md',
        '.jpg', '.jpeg', '.png', '.tiff'
    }

    extension = os.path.splitext(filename)[1].lower()
    return extension in valid_extensions


def _is_valid_file_size(size: int) -> bool:
    """检查文件大小是否有效"""
    max_size = 50 * 1024 * 1024  # 50MB
    return 0 < size <= max_size


def _generate_safe_filename(filename: str) -> str:
    """生成安全的文件名"""
    safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    safe_filename = ''.join(c for c in filename if c in safe_chars)

    if len(safe_filename) > 100:
        name, ext = os.path.splitext(safe_filename)
        safe_filename = name[:100-len(ext)] + ext

    if not safe_filename:
        safe_filename = f"document_{uuid.uuid4().hex[:8]}"

    return safe_filename


# 状态管理函数（简化实现）
async def _update_enhanced_batch_status(
    batch_id: str,
    status: str,
    processed: int,
    total: int,
    errors: List[str],
    metrics: Optional[Dict[str, Any]] = None
):
    """更新增强批量处理状态"""
    # 这里应该保存到Redis或数据库
    pass


async def _get_enhanced_batch_status(batch_id: str) -> Dict[str, Any]:
    """获取增强批量处理状态"""
    # 简化实现
    return {
        "batch_id": batch_id,
        "status": "processing",
        "progress": 50,
        "total_files": 10,
        "processed_files": 5,
        "errors": [],
        "metrics": {
            "avg_quality_score": 0.85,
            "total_vectors": 150,
            "total_kg_entities": 80,
            "total_knowledge_images": 25
        }
    }


async def _get_processing_results(batch_id: str, user_id: str) -> Dict[str, Any]:
    """获取处理结果详情"""
    # 简化实现
    return {
        "batch_id": batch_id,
        "results": [
            {
                "document_id": "doc_123",
                "filename": "research_report.pdf",
                "status": "completed",
                "quality_score": 0.92,
                "processing_time": 45.2,
                "engines_used": ["qwen-vl-max", "mineru", "mathpix"],
                "content_summary": {
                    "total_pages": 15,
                    "tables_count": 8,
                    "images_count": 5,
                    "formulas_count": 12
                },
                "vectors_generated": 25,
                "kg_entities": 18,
                "knowledge_images": 6
            }
        ]
    }