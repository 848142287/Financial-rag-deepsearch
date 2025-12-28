"""
统一文档处理编排器
协调所有文档处理组件，提供统一的文档处理入口
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.synchronization import DocumentSync, SyncStatus, SyncPriority

# 导入各种处理服务
from app.services.consolidated_document_service import ConsolidatedDocumentService as DocumentParser
from app.services.qwen_embedding_service import QwenEmbeddingService
from app.services.qwen_service import QwenService
from app.services.result_storage_service import ResultStorageService, StorageType
from app.services.metadata_sync_service import MetadataSyncService, StorageSystem, SyncTask
from app.services.error_handler import (
    ErrorHandler, RetryConfig, RetryStrategy, safe_execute,
    handle_error_with_fallback
)
from app.services.enhanced_pdf_processor import (
    EnhancedPDFProcessor, process_pdf_document, ProcessingResult
)
from app.services.specialized_parsers import (
    analyze_financial_chart, analyze_financial_formula
)
from app.services.financial_ocr_enhancer import (
    enhance_financial_document_ocr, DocumentType
)

# 导入外部存储服务
try:
    from app.services.milvus_service import MilvusService
    from app.services.neo4j_service import Neo4jService
    from app.services.minio_service import MinioService
    EXTERNAL_SERVICES_AVAILABLE = True
except ImportError as e:
    EXTERNAL_SERVICES_AVAILABLE = False
    logging.warning(f"External services not available: {e}")

logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """处理阶段"""
    INIT = "init"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStrategy(str, Enum):
    """处理策略"""
    MINERU_PRIORITY = "mineru_priority"  # 优先使用Mineru
    ENHANCED_PDF = "enhanced_pdf"        # 使用增强PDF处理器
    FALLBACK_ONLY = "fallback_only"      # 仅使用回退处理器
    AUTO = "auto"                        # 自动选择最佳策略


@dataclass
class ProcessingConfig:
    """处理配置"""
    strategy: ProcessingStrategy = ProcessingStrategy.AUTO
    use_ocr: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    enable_vision_analysis: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "qwen_primary"
    enable_multistorage: bool = True
    enable_incremental: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    document_id: int
    stage: ProcessingStage
    total_chunks: int = 0
    processed_chunks: int = 0
    parsing_metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    storage_metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)


class DocumentOrchestrator:
    """统一文档处理编排器"""

    def __init__(self):
        self.document_parser = DocumentParser()
        self.enhanced_pdf_processor = None
        self.qwen_embedding_service = None
        self.qwen_service = None
        self.result_storage_service = None
        self.metadata_sync_service = None
        self.error_handler = ErrorHandler()

        # 外部服务
        self.milvus_service = None
        self.neo4j_service = None
        self.minio_service = None

        # 处理统计
        self.processing_stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "average_processing_time": 0.0
        }

        # 配置重试策略
        self.retry_config = RetryConfig(
            max_retries=3,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=30.0
        )

    async def initialize(self):
        """初始化所有服务"""
        try:
            logger.info("Initializing document orchestrator services...")

            # 初始化增强PDF处理器
            self.enhanced_pdf_processor = EnhancedPDFProcessor()
            await self.enhanced_pdf_processor.initialize()

            # 初始化Qwen嵌入服务
            self.qwen_embedding_service = QwenEmbeddingService()

            # 初始化Qwen多模态服务
            self.qwen_service = QwenService()

            # 初始化结果存储服务
            self.result_storage_service = ResultStorageService()
            await self.result_storage_service.initialize()

            # 初始化元数据同步服务
            self.metadata_sync_service = MetadataSyncService()
            await self.metadata_sync_service.initialize()

            # 初始化外部服务（如果可用）
            if EXTERNAL_SERVICES_AVAILABLE:
                try:
                    self.milvus_service = MilvusService()
                    await self.milvus_service.health_check()
                except Exception as e:
                    logger.warning(f"Milvus service initialization failed: {e}")

                try:
                    self.neo4j_service = Neo4jService()
                    await self.neo4j_service.health_check()
                except Exception as e:
                    logger.warning(f"Neo4j service initialization failed: {e}")

                try:
                    self.minio_service = MinioService()
                    await self.minio_service.health_check()
                except Exception as e:
                    logger.warning(f"MinIO service initialization failed: {e}")

            logger.info("Document orchestrator initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize document orchestrator: {e}")
            raise

    async def process_document(
        self,
        document_id: int,
        config: Optional[ProcessingConfig] = None,
        db: Optional[Session] = None
    ) -> ProcessingResult:
        """
        处理文档的统一入口

        Args:
            document_id: 文档ID
            config: 处理配置
            db: 数据库会话

        Returns:
            ProcessingResult: 处理结果
        """
        start_time = datetime.utcnow()

        if config is None:
            config = ProcessingConfig()

        result = ProcessingResult(
            success=False,
            document_id=document_id,
            stage=ProcessingStage.INIT
        )

        try:
            # 获取数据库会话
            if db is None:
                db_gen = get_db()
                db = next(db_gen)

            # 获取文档
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                result.error_message = f"Document {document_id} not found"
                result.stage = ProcessingStage.FAILED
                return result

            logger.info(f"Starting document processing: {document.title} (ID: {document_id})")

            # 更新文档状态
            document.status = DocumentStatus.PROCESSING
            db.commit()

            # 检查增量处理
            if config.enable_incremental:
                should_skip = await self._check_incremental_processing(document, db)
                if should_skip:
                    result.success = True
                    result.stage = ProcessingStage.COMPLETED
                    result.warnings.append("Document unchanged, skipped processing")

                    document.status = DocumentStatus.COMPLETED
                    document.processed_at = datetime.utcnow()
                    db.commit()

                    return result

            # 阶段1: 文档解析
            logger.info(f"Stage 1: Parsing document {document_id}")
            result.stage = ProcessingStage.PARSING
            parse_result = await self._parse_document(document, config)

            if not parse_result["success"]:
                result.error_message = parse_result["error"]
                result.stage = ProcessingStage.FAILED
                await self._handle_processing_failure(document, result, db)
                return result

            result.parsing_metadata = parse_result["metadata"]
            content_data = parse_result["content"]

            # 阶段2: 文本分块
            logger.info(f"Stage 2: Chunking document {document_id}")
            result.stage = ProcessingStage.CHUNKING
            chunks = await self._create_chunks(document, content_data, config, db)

            if not chunks:
                result.error_message = "Failed to create document chunks"
                result.stage = ProcessingStage.FAILED
                await self._handle_processing_failure(document, result, db)
                return result

            result.total_chunks = len(chunks)

            # 阶段3: 嵌入生成
            logger.info(f"Stage 3: Generating embeddings for {len(chunks)} chunks")
            result.stage = ProcessingStage.EMBEDDING
            embedding_result = await self._generate_embeddings(chunks, config)

            if not embedding_result["success"]:
                result.error_message = embedding_result["error"]
                result.stage = ProcessingStage.FAILED
                await self._handle_processing_failure(document, result, db)
                return result

            result.embedding_metadata = embedding_result["metadata"]
            result.processed_chunks = len(embedding_result["embeddings"])

            # 阶段4: 存储处理
            logger.info(f"Stage 4: Storing processing results")
            result.stage = ProcessingStage.STORAGE
            storage_result = await self._store_results(
                document, chunks, embedding_result["embeddings"], config
            )

            result.storage_metadata = storage_result

            # 阶段5: 索引建立
            logger.info(f"Stage 5: Building indexes")
            result.stage = ProcessingStage.INDEXING
            await self._build_indexes(document, chunks, embedding_result["embeddings"])

            # 处理完成
            result.success = True
            result.stage = ProcessingStage.COMPLETED
            result.processing_time = (datetime.utcnow() - start_time).total_seconds()

            # 更新文档状态
            document.status = DocumentStatus.COMPLETED
            document.processed_at = datetime.utcnow()
            document.doc_metadata = {
                **(document.doc_metadata or {}),
                "orchestrator_metadata": {
                    "processing_time": result.processing_time,
                    "total_chunks": result.total_chunks,
                    "strategy": config.strategy.value,
                    "parsing_metadata": result.parsing_metadata,
                    "embedding_metadata": result.embedding_metadata,
                    "storage_metadata": result.storage_metadata
                }
            }
            db.commit()

            # 更新统计信息
            self._update_stats(result)

            logger.info(f"Document processing completed successfully: {document.title} "
                       f"({result.processing_time:.2f}s, {result.total_chunks} chunks)")

            return result

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            result.error_message = str(e)
            result.stage = ProcessingStage.FAILED
            result.processing_time = (datetime.utcnow() - start_time).total_seconds()

            # 处理失败
            if 'document' in locals():
                await self._handle_processing_failure(document, result, db)

            return result

    async def _parse_document(
        self, document: Document, config: ProcessingConfig
    ) -> Dict[str, Any]:
        """解析文档"""

        async def _enhanced_pdf_parsing() -> Dict[str, Any]:
            """使用增强PDF处理器解析"""
            file_path = document.file_path

            # 首先检查MinIO服务是否可用
            if self.minio_service is None:
                raise Exception("MinIO service not available")

            if not file_path:
                raise FileNotFoundError("Document file path is empty")

            # 从MinIO下载文件到临时位置
            try:
                file_data = await self.minio_service.download_file(file_path)
                if not file_data:
                    raise FileNotFoundError(f"Document file not found in MinIO: {file_path}")

                # 创建临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name

                # 更新文档的实际文件大小
                if document.file_size != len(file_data):
                    document.file_size = len(file_data)
                    # 注意：这里需要在数据库会话中提交，但为了不破坏当前逻辑，先记录日志
                    logger.info(f"Document {document.id} file size updated: {document.file_size} -> {len(file_data)} bytes")

            except Exception as e:
                raise Exception(f"Failed to download file from MinIO: {e}")

            try:
                # 判断文档类型
                document_type = await self._detect_document_type(document)

                # 使用增强PDF处理器 - 使用临时文件路径
                pdf_config = {
                    "use_ocr": config.use_ocr,
                    "extract_images": config.extract_images,
                    "extract_tables": config.extract_tables,
                    "extract_formulas": config.enable_vision_analysis,
                    "extract_charts": config.enable_vision_analysis,
                    "document_type": document_type
                }

                processing_result = await self.enhanced_pdf_processor.process_pdf_document(
                    temp_file_path, pdf_config
                )

                if not processing_result.success:
                    raise Exception(f"Enhanced PDF processing failed: {processing_result.errors}")

                # 转换为统一格式
                return self._convert_processing_result(processing_result)

            finally:
                # 清理临时文件
                try:
                    import os
                    os.unlink(temp_file_path)
                    logger.info(f"Temporary file cleaned up: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}")

        async def _traditional_parsing() -> Dict[str, Any]:
            """传统解析方法"""
            # 首先从MinIO下载文件
            if self.minio_service is None:
                raise Exception("MinIO service not available for traditional parsing")

            file_path = document.file_path
            if not file_path:
                raise FileNotFoundError("Document file path is empty")

            try:
                file_data = await self.minio_service.download_file(file_path)
                if not file_data:
                    raise FileNotFoundError(f"Document file not found in MinIO: {file_path}")

                # 创建临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name

                try:
                    return await self.document_parser.parse_document(
                        temp_file_path,  # 使用临时文件路径
                        use_ocr=config.use_ocr,
                        extract_images=config.extract_images,
                        extract_tables=config.extract_tables,
                        use_enhanced_pdf=False
                    )
                finally:
                    # 清理临时文件
                    try:
                        import os
                        os.unlink(temp_file_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}")

            except Exception as e:
                raise Exception(f"Failed to download file from MinIO for traditional parsing: {e}")

        async def _fallback_parsing() -> Dict[str, Any]:
            """终极回退解析"""
            try:
                # 从MinIO下载文件进行简单文本提取
                if self.minio_service is None:
                    raise Exception("MinIO service not available for fallback parsing")

                file_path = document.file_path
                if not file_path:
                    raise FileNotFoundError("Document file path is empty")

                file_data = await self.minio_service.download_file(file_path)
                if not file_data:
                    raise FileNotFoundError(f"Document file not found in MinIO: {file_path}")

                # 对于PDF，尝试简单文本提取
                if file_data.startswith(b'%PDF'):
                    try:
                        import PyPDF2
                        import io
                        pdf_stream = io.BytesIO(file_data)
                        pdf_reader = PyPDF2.PdfReader(pdf_stream)

                        text_content = ""
                        for page in pdf_reader.pages[:3]:  # 只读取前3页
                            page_text = page.extract_text()
                            if page_text.strip():
                                text_content += page_text + "\n"

                        if text_content.strip():
                            return {
                                "success": True,
                                "content": {
                                    "text_content": [text_content],
                                    "images": [],
                                    "tables": [],
                                    "metadata": {"parser": "PDFFallback"}
                                },
                                "metadata": {"parser": "PDFFallback"}
                            }
                    except Exception as pdf_error:
                        logger.warning(f"PDF fallback parsing failed: {pdf_error}")

                # 如果PDF解析失败或不是PDF，返回基本信息
                return {
                    "success": True,
                    "content": {
                        "text_content": [f"Document: {document.title}\nFile: {document.file_name}\nSize: {len(file_data)} bytes"],
                        "images": [],
                        "tables": [],
                        "metadata": {"parser": "Minimal"}
                    },
                    "metadata": {"parser": "Minimal"}
                }

            except Exception as e:
                # 如果连MinIO访问都失败，返回最基本的信息
                logger.error(f"Fallback parsing completely failed: {e}")
                return {
                    "success": True,
                    "content": {
                        "text_content": [f"Document: {document.title}"],
                        "images": [],
                        "tables": [],
                        "metadata": {"parser": "ErrorFallback"}
                    },
                    "metadata": {"parser": "ErrorFallback"}
                }

        try:
            # 使用回退机制的解析流程
            if config.strategy == ProcessingStrategy.ENHANCED_PDF:
                return await handle_error_with_fallback(
                    Exception("Enhanced PDF parsing attempt"),
                    {"operation": "document_parsing", "document_id": document.id, "strategy": "enhanced"},
                    _enhanced_pdf_parsing,
                    _traditional_parsing
                )
            elif config.strategy == ProcessingStrategy.MINERU_PRIORITY:
                return await handle_error_with_fallback(
                    Exception("Mineru parsing attempt"),
                    {"operation": "document_parsing", "document_id": document.id, "strategy": "mineru"},
                    _traditional_parsing,  # 这里应该调用Mineru，但暂时用传统方法
                    _enhanced_pdf_parsing
                )
            else:  # AUTO or other strategies
                return await handle_error_with_fallback(
                    Exception("Auto parsing attempt"),
                    {"operation": "document_parsing", "document_id": document.id, "strategy": "auto"},
                    _enhanced_pdf_parsing,
                    _traditional_parsing
                )

        except Exception as e:
            # 最后的错误处理
            await self.error_handler.handle_error(
                e,
                {"operation": "document_parsing", "document_id": document.id}
            )
            return {"success": False, "error": str(e)}

    async def _detect_document_type(self, document: Document) -> DocumentType:
        """检测文档类型"""
        try:
            # 基于文件名和内容判断
            filename_lower = document.filename.lower()

            if any(keyword in filename_lower for keyword in ['财务报告', '年报', '季报', '财报']):
                return DocumentType.FINANCIAL_REPORT
            elif any(keyword in filename_lower for keyword in ['发票', 'invoice', '票据']):
                return DocumentType.INVOICE
            elif any(keyword in filename_lower for keyword in ['合同', 'contract', '协议']):
                return DocumentType.CONTRACT
            elif any(keyword in filename_lower for keyword in ['银行', '对账单', 'statement']):
                return DocumentType.BANK_STATEMENT
            elif any(keyword in filename_lower for keyword in ['税务', '税单', 'tax']):
                return DocumentType.TAX_DOCUMENT
            elif any(keyword in filename_lower for keyword in ['保险', 'insurance']):
                return DocumentType.INSURANCE
            elif any(keyword in filename_lower for keyword in ['投资', 'investment', '基金']):
                return DocumentType.INVESTMENT
            else:
                return DocumentType.UNKNOWN

        except Exception as e:
            logger.error(f"Error detecting document type: {e}")
            return DocumentType.UNKNOWN

    def _convert_processing_result(self, processing_result: ProcessingResult) -> Dict[str, Any]:
        """转换处理结果为统一格式"""
        try:
            # 提取文本内容
            text_content = []
            images = []
            tables = []
            charts = []
            formulas = []

            for content_block in processing_result.contents:
                if content_block.content_type in [ContentType.TEXT, ContentType.HEADER]:
                    text_content.append({
                        "page": content_block.page_number,
                        "content": content_block.text,
                        "confidence": content_block.confidence,
                        "bbox": content_block.bbox,
                        "metadata": content_block.metadata
                    })
                elif content_block.content_type == ContentType.IMAGE:
                    images.append({
                        "page": content_block.page_number,
                        "description": content_block.text,
                        "path": content_block.image_path,
                        "confidence": content_block.confidence,
                        "metadata": content_block.metadata
                    })
                elif content_block.content_type == ContentType.TABLE:
                    tables.append({
                        "page": content_block.page_number,
                        "text": content_block.text,
                        "data": content_block.table_data,
                        "confidence": content_block.confidence,
                        "bbox": content_block.bbox
                    })
                elif content_block.content_type == ContentType.CHART:
                    charts.append({
                        "page": content_block.page_number,
                        "description": content_block.text,
                        "path": content_block.image_path,
                        "confidence": content_block.confidence,
                        "metadata": content_block.metadata
                    })
                elif content_block.content_type == ContentType.FORMULA:
                    formulas.append({
                        "page": content_block.page_number,
                        "formula": content_block.text,
                        "confidence": content_block.confidence,
                        "metadata": content_block.metadata
                    })

            return {
                "success": True,
                "content": {
                    "text_content": text_content,
                    "images": images,
                    "tables": tables,
                    "charts": charts,
                    "formulas": formulas,
                    "metadata": processing_result.metadata
                },
                "metadata": {
                    "parser": "EnhancedPDF",
                    "pages_processed": processing_result.pages_processed,
                    "total_pages": processing_result.total_pages,
                    "processing_time": processing_result.processing_time,
                    "errors": processing_result.errors
                }
            }

        except Exception as e:
            logger.error(f"Error converting processing result: {e}")
            return {
                "success": False,
                "error": f"Result conversion failed: {str(e)}"
            }

    async def _create_chunks(
        self,
        document: Document,
        content_data: Dict[str, Any],
        config: ProcessingConfig,
        db: Session
    ) -> List[DocumentChunk]:
        """创建文档分块"""
        try:
            # 删除现有分块
            db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document.id
            ).delete()

            # 提取文本内容
            text_content = []
            if "text_content" in content_data:
                for text_item in content_data["text_content"]:
                    if isinstance(text_item, dict):
                        text_content.append(text_item.get("content", ""))
                    else:
                        text_content.append(str(text_item))

            if not text_content:
                # 尝试从其他地方提取文本
                if "content" in content_data:
                    text_content = [content_data["content"]]
                else:
                    raise ValueError("No text content found in parsed document")

            # 使用文档解析器的分块功能
            chunks_data = self.document_parser.chunk_text(
                text_content,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )

            # 创建数据库分块记录
            chunks = []
            for i, chunk_data in enumerate(chunks_data):
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_data["content"],
                    chunk_metadata={
                        **chunk_data["metadata"],
                        "document_metadata": content_data.get("metadata", {}),
                        "images": content_data.get("images", []),
                        "tables": content_data.get("tables", [])
                    }
                )
                chunks.append(chunk)

            # 保存到数据库
            db.add_all(chunks)
            db.commit()

            logger.info(f"Created {len(chunks)} document chunks")
            return chunks

        except Exception as e:
            logger.error(f"Chunk creation failed: {e}")
            raise

    async def _generate_embeddings(
        self,
        chunks: List[DocumentChunk],
        config: ProcessingConfig
    ) -> Dict[str, Any]:
        """生成嵌入向量"""
        try:
            # 准备文本
            texts = [chunk.content for chunk in chunks]

            # 生成嵌入
            embeddings = await self.embedding_service.embed_batch(texts)

            # 更新分块记录
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_id = hashlib.md5(
                    f"{chunk.document_id}_{chunk.chunk_index}".encode()
                ).hexdigest()

            return {
                "success": True,
                "embeddings": embeddings,
                "metadata": {
                    "model": config.embedding_model,
                    "dimension": len(embeddings[0]) if embeddings else 0,
                    "count": len(embeddings)
                }
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _store_results(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        config: ProcessingConfig
    ) -> Dict[str, Any]:
        """存储处理结果"""
        try:
            storage_results = {}

            if config.enable_multistorage:
                # 存储解析结果
                parse_result_data = {
                    "document_id": document.id,
                    "chunks": [
                        {
                            "chunk_id": chunk.id,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "metadata": chunk.chunk_metadata
                        }
                        for chunk in chunks
                    ],
                    "embeddings": embeddings,
                    "metadata": {
                        "document_id": document.id,
                        "title": document.title,
                        "filename": document.filename,
                        "created_at": datetime.utcnow().isoformat()
                    }
                }

                # 存储到多个存储系统
                await self.result_storage_service.store_parse_result(
                    str(document.id),
                    parse_result_data,
                    StorageType.MONGODB
                )

                # 存储文档摘要
                summary = {
                    "document_id": document.id,
                    "title": document.title,
                    "total_chunks": len(chunks),
                    "has_images": any(
                        chunk.chunk_metadata.get("images") for chunk in chunks
                    ),
                    "has_tables": any(
                        chunk.chunk_metadata.get("tables") for chunk in chunks
                    )
                }

                await self.result_storage_service.store_document_summary(
                    str(document.id),
                    summary
                )

                storage_results["multistorage"] = "completed"

            return storage_results

        except Exception as e:
            logger.error(f"Result storage failed: {e}")
            raise

    async def _build_indexes(
        self,
        document: Document,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ):
        """建立索引"""
        try:
            # 创建同步记录
            sync_record = DocumentSync(
                document_id=document.id,
                document_version=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                content_hash=hashlib.sha256(
                    "".join([chunk.content for chunk in chunks]).encode()
                ).hexdigest(),
                total_chunks=len(chunks),
                sync_status=SyncStatus.COMPLETED
            )

            # 提交同步记录
            db_gen = get_db()
            db = next(db_gen)
            db.add(sync_record)
            db.commit()

            # 如果外部服务可用，触发同步
            if self.metadata_sync_service and EXTERNAL_SERVICES_AVAILABLE:
                # 向量库同步
                if self.milvus_service:
                    sync_task = SyncTask(
                        task_id=f"vector_sync_{document.id}_{int(datetime.utcnow().timestamp())}",
                        document_id=str(document.id),
                        source_system=StorageSystem.MYSQL,
                        target_system=StorageSystem.MILVUS,
                        operation="create",
                        data={"chunks_count": len(chunks)}
                    )
                    await self.metadata_sync_service.submit_sync_task(sync_task)

                # 图谱库同步
                if self.neo4j_service:
                    sync_task = SyncTask(
                        task_id=f"graph_sync_{document.id}_{int(datetime.utcnow().timestamp())}",
                        document_id=str(document.id),
                        source_system=StorageSystem.MYSQL,
                        target_system=StorageSystem.NEO4J,
                        operation="create",
                        data={"chunks_count": len(chunks)}
                    )
                    await self.metadata_sync_service.submit_sync_task(sync_task)

            logger.info(f"Indexes built successfully for document {document.id}")

        except Exception as e:
            logger.error(f"Index building failed: {e}")
            # 不抛出异常，因为索引失败不影响主要处理流程

    async def _check_incremental_processing(
        self, document: Document, db: Session
    ) -> bool:
        """检查是否需要增量处理"""
        try:
            # 检查是否有已完成的同步记录
            latest_sync = db.query(DocumentSync).filter(
                and_(
                    DocumentSync.document_id == document.id,
                    DocumentSync.sync_status == SyncStatus.COMPLETED
                )
            ).order_by(DocumentSync.created_at.desc()).first()

            if not latest_sync:
                return False

            # 检查文件是否有变化
            try:
                with open(document.file_path, 'rb') as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()

                if content_hash == latest_sync.content_hash:
                    logger.info(f"Document {document.id} unchanged, skipping processing")
                    return True

            except Exception:
                # 文件读取失败，需要重新处理
                return False

            return False

        except Exception as e:
            logger.error(f"Incremental processing check failed: {e}")
            return False

    async def _handle_processing_failure(
        self, document: Document, result: ProcessingResult, db: Session
    ):
        """处理处理失败的情况"""
        try:
            # 更新文档状态
            document.status = DocumentStatus.FAILED
            db.commit()

            # 记录失败原因
            logger.error(f"Document processing failed for {document.id}: {result.error_message}")

            # 可以在这里添加告警逻辑

        except Exception as e:
            logger.error(f"Failed to handle processing failure: {e}")

    def _update_stats(self, result: ProcessingResult):
        """更新处理统计信息"""
        self.processing_stats["total_documents"] += 1

        if result.success:
            self.processing_stats["successful_documents"] += 1
        else:
            self.processing_stats["failed_documents"] += 1

        # 更新平均处理时间
        total_time = (
            self.processing_stats["average_processing_time"] *
            (self.processing_stats["total_documents"] - 1) +
            result.processing_time
        )
        self.processing_stats["average_processing_time"] = (
            total_time / self.processing_stats["total_documents"]
        )

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        try:
            # 获取服务健康状态
            health_status = {}

            if self.qwen_embedding_service:
                health_status["qwen_embedding_service"] = await self.qwen_embedding_service.health_check()
            if self.qwen_service:
                health_status["qwen_service"] = await self.qwen_service.health_check()

            if self.result_storage_service:
                health_status["storage_service"] = (
                    await self.result_storage_service.get_storage_statistics()
                )

            if self.metadata_sync_service:
                health_status["sync_service"] = {"status": "active"}

            return {
                "processing_stats": self.processing_stats,
                "health_status": health_status,
                "external_services": {
                    "milvus_available": self.milvus_service is not None,
                    "neo4j_available": self.neo4j_service is not None,
                    "minio_available": self.minio_service is not None
                }
            }

        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {"error": str(e)}


# 全局编排器实例
document_orchestrator = DocumentOrchestrator()


async def get_document_orchestrator() -> DocumentOrchestrator:
    """获取文档编排器实例"""
    if not document_orchestrator.qwen_embedding_service:
        await document_orchestrator.initialize()
    return document_orchestrator


# 便捷函数
async def process_document(
    document_id: int,
    config: Optional[ProcessingConfig] = None
) -> ProcessingResult:
    """便捷的文档处理函数"""
    orchestrator = await get_document_orchestrator()
    return await orchestrator.process_document(document_id, config)