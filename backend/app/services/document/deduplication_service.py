"""
文档去重服务
基于文件hash和内容相似度进行文档去重
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session

import numpy as np

from app.core.logger_utils import get_module_logger
from app.models.document import Document

logger = get_module_logger(__name__)

@dataclass
class DeduplicationResult:
    """去重结果"""
    is_duplicate: bool
    duplicate_of: Optional[int] = None  # 重复文档的ID
    similarity: Optional[float] = None  # 相似度（0-1）
    method: Optional[str] = None  # 去重方法: hash, content, embedding
    reason: Optional[str] = None

@dataclass
class DocumentFingerprint:
    """文档指纹"""
    document_id: int
    file_hash: str  # 文件内容的MD5/SHA256
    size: int  # 文件大小
    title_hash: str  # 标题的hash
    content_hash: Optional[str] = None  # 文本内容的hash
    embedding: Optional[np.ndarray] = None  # 内容嵌入向量

class DocumentDeduplicationService:
    """
    文档去重服务

    支持三种去重方法：
    1. 文件hash去重 - 完全相同的文件
    2. 内容hash去重 - 文本内容相同
    3. 相似度去重 - 基于嵌入向量的余弦相似度
    """

    def __init__(
        self,
        enable_hash_dedup: bool = True,
        enable_content_dedup: bool = True,
        enable_similarity_dedup: bool = True,
        similarity_threshold: float = 0.95,
        hash_algorithm: str = "sha256"
    ):
        self.enable_hash_dedup = enable_hash_dedup
        self.enable_content_dedup = enable_content_dedup
        self.enable_similarity_dedup = enable_similarity_dedup
        self.similarity_threshold = similarity_threshold
        self.hash_algorithm = hash_algorithm

        # 缓存文档指纹
        self._fingerprint_cache: Dict[int, DocumentFingerprint] = {}

        logger.info(
            f"DocumentDeduplicationService initialized with "
            f"hash={enable_hash_dedup}, content={enable_content_dedup}, "
            f"similarity={enable_similarity_dedup} (threshold={similarity_threshold})"
        )

    def compute_file_hash(self, file_path: str) -> str:
        """
        计算文件的hash值

        Args:
            file_path: 文件路径

        Returns:
            文件的hash值
        """
        hash_func = hashlib.new(self.hash_algorithm)

        try:
            with open(file_path, 'rb') as f:
                # 分块读取大文件
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute file hash for {file_path}: {e}")
            return ""

    def compute_content_hash(self, content: str) -> str:
        """
        计算文本内容的hash值

        Args:
            content: 文本内容

        Returns:
            内容的hash值
        """
        hash_func = hashlib.new(self.hash_algorithm)
        hash_func.update(content.encode('utf-8'))
        return hash_func.hexdigest()

    async def get_document_fingerprint(
        self,
        document: Document,
        file_path: Optional[str] = None,
        content: Optional[str] = None
    ) -> DocumentFingerprint:
        """
        生成文档指纹

        Args:
            document: 文档对象
            file_path: 文件路径（可选）
            content: 文档内容（可选）

        Returns:
            文档指纹
        """
        # 检查缓存
        if document.id in self._fingerprint_cache:
            return self._fingerprint_cache[document.id]

        # 计算文件hash
        file_path = file_path or document.file_path
        file_hash = self.compute_file_hash(file_path) if file_path else ""

        # 计算标题hash
        title_hash = self.compute_content_hash(document.title or "")

        # 计算内容hash
        content_hash = self.compute_content_hash(content) if content else None

        # 获取文件大小
        size = document.file_size if hasattr(document, 'file_size') else 0

        fingerprint = DocumentFingerprint(
            document_id=document.id,
            file_hash=file_hash,
            size=size,
            title_hash=title_hash,
            content_hash=content_hash
        )

        # 缓存指纹
        self._fingerprint_cache[document.id] = fingerprint

        return fingerprint

    async def check_duplicate_by_hash(
        self,
        fingerprint: DocumentFingerprint,
        db: Session
    ) -> Optional[DeduplicationResult]:
        """
        基于文件hash检查重复

        Args:
            fingerprint: 文档指纹
            db: 数据库会话

        Returns:
            去重结果
        """
        if not self.enable_hash_dedup or not fingerprint.file_hash:
            return None

        try:
            # 查询相同hash的文档
            existing = db.query(Document).filter(
                and_(
                    Document.id != fingerprint.document_id,
                    # 假设有一个file_hash字段存储hash值
                    # 如果没有，可以通过metadata存储
                )
            ).first()

            # 如果没有file_hash字段，尝试从metadata中查找
            if not existing:
                all_docs = db.query(Document).filter(
                    Document.id != fingerprint.document_id
                ).all()

                for doc in all_docs:
                    if doc.doc_metadata:
                        stored_hash = doc.doc_metadata.get('file_hash')
                        if stored_hash == fingerprint.file_hash:
                            existing = doc
                            break

            if existing:
                return DeduplicationResult(
                    is_duplicate=True,
                    duplicate_of=existing.id,
                    method="hash",
                    reason=f"File hash matches document {existing.id}"
                )

        except Exception as e:
            logger.error(f"Hash deduplication check failed: {e}")

        return None

    async def check_duplicate_by_content(
        self,
        fingerprint: DocumentFingerprint,
        db: Session
    ) -> Optional[DeduplicationResult]:
        """
        基于内容hash检查重复

        Args:
            fingerprint: 文档指纹
            db: 数据库会话

        Returns:
            去重结果
        """
        if not self.enable_content_dedup or not fingerprint.content_hash:
            return None

        try:
            # 从metadata中查找相同内容hash
            all_docs = db.query(Document).filter(
                Document.id != fingerprint.document_id
            ).all()

            for doc in all_docs:
                if doc.doc_metadata:
                    stored_content_hash = doc.doc_metadata.get('content_hash')
                    if stored_content_hash == fingerprint.content_hash:
                        return DeduplicationResult(
                            is_duplicate=True,
                            duplicate_of=doc.id,
                            method="content",
                            reason=f"Content hash matches document {doc.id}"
                        )

        except Exception as e:
            logger.error(f"Content deduplication check failed: {e}")

        return None

    async def check_duplicate_by_similarity(
        self,
        fingerprint: DocumentFingerprint,
        db: Session,
        embedding_service=None
    ) -> Optional[DeduplicationResult]:
        """
        基于相似度检查重复

        Args:
            fingerprint: 文档指纹
            db: 数据库会话
            embedding_service: 嵌入服务（可选）

        Returns:
            去重结果
        """
        if not self.enable_similarity_dedup:
            return None

        if not embedding_service:
            try:
                from app.services.model_service_factory import get_embedding_service
                embedding_service = get_embedding_service()
            except Exception as e:
                logger.warning(f"Cannot get embedding service for similarity check: {e}")
                return None

        try:
            # 这里需要获取所有文档的嵌入进行比较
            # 为提高性能，可以先按标题、大小等条件缩小范围

            all_docs = db.query(Document).filter(
                Document.id != fingerprint.document_id
            ).all()

            for doc in all_docs:
                # 快速过滤：大小差异太大直接跳过
                if hasattr(doc, 'file_size') and fingerprint.size > 0:
                    size_diff = abs(doc.file_size - fingerprint.size) / max(fingerprint.size, 1)
                    if size_diff > 0.1:  # 大小差异超过10%
                        continue

                # 计算标题相似度
                title_similarity = await self._compute_text_similarity(
                    fingerprint.title_hash,
                    self.compute_content_hash(doc.title or "")
                )

                if title_similarity >= self.similarity_threshold:
                    return DeduplicationResult(
                        is_duplicate=True,
                        duplicate_of=doc.id,
                        similarity=title_similarity,
                        method="similarity",
                        reason=f"Title similarity {title_similarity:.2%} exceeds threshold"
                    )

        except Exception as e:
            logger.error(f"Similarity deduplication check failed: {e}")

        return None

    async def _compute_text_similarity(self, hash1: str, hash2: str) -> float:
        """
        计算两个文本的相似度（简化版本）

        Args:
            hash1: 第一个文本的hash
            hash2: 第二个文本的hash

        Returns:
            相似度（0-1）
        """
        # 简化版本：直接比较hash
        # 实际应该使用嵌入向量计算余弦相似度
        if hash1 == hash2:
            return 1.0
        else:
            return 0.0

    async def check_duplicate(
        self,
        document: Document,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        db: Optional[Session] = None,
        embedding_service=None
    ) -> DeduplicationResult:
        """
        综合检查文档是否重复

        Args:
            document: 文档对象
            file_path: 文件路径
            content: 文档内容
            db: 数据库会话
            embedding_service: 嵌入服务

        Returns:
            去重结果
        """
        if db is None:
            from app.core.database import get_db
            db_gen = get_db()
            db = next(db_gen)

        try:
            # 生成文档指纹
            fingerprint = await self.get_document_fingerprint(document, file_path, content)

            # 1. 检查文件hash
            result = await self.check_duplicate_by_hash(fingerprint, db)
            if result:
                logger.info(f"Document {document.id} is duplicate by hash of {result.duplicate_of}")
                return result

            # 2. 检查内容hash
            result = await self.check_duplicate_by_content(fingerprint, db)
            if result:
                logger.info(f"Document {document.id} is duplicate by content of {result.duplicate_of}")
                return result

            # 3. 检查相似度
            result = await self.check_duplicate_by_similarity(fingerprint, db, embedding_service)
            if result:
                logger.info(f"Document {document.id} is duplicate by similarity ({result.similarity:.2%}) of {result.duplicate_of}")
                return result

            # 不是重复文档
            return DeduplicationResult(
                is_duplicate=False,
                reason="No duplicate found"
            )

        except Exception as e:
            logger.error(f"Deduplication check failed for document {document.id}: {e}")
            return DeduplicationResult(
                is_duplicate=False,
                reason=f"Check failed: {str(e)}"
            )

    async def store_document_hash(
        self,
        document: Document,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        db: Optional[Session] = None
    ):
        """
        存储文档hash到metadata

        Args:
            document: 文档对象
            file_path: 文件路径
            content: 文档内容
            db: 数据库会话
        """
        try:
            fingerprint = await self.get_document_fingerprint(document, file_path, content)

            # 更新metadata
            if not document.doc_metadata:
                document.doc_metadata = {}

            document.doc_metadata['file_hash'] = fingerprint.file_hash
            if fingerprint.content_hash:
                document.doc_metadata['content_hash'] = fingerprint.content_hash
            document.doc_metadata['deduplication_checked_at'] = datetime.now().isoformat()

            if db:
                db.commit()

            logger.debug(f"Stored hash for document {document.id}")

        except Exception as e:
            logger.error(f"Failed to store document hash: {e}")

    def clear_cache(self):
        """清除指纹缓存"""
        self._fingerprint_cache.clear()
        logger.info("Document fingerprint cache cleared")

# 全局服务实例
_deduplication_service = None

def get_deduplication_service() -> DocumentDeduplicationService:
    """获取文档去重服务实例"""
    global _deduplication_service
    if _deduplication_service is None:
        _deduplication_service = DocumentDeduplicationService()
    return _deduplication_service
