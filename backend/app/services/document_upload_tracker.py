"""
文档上传追踪服务
从 swxy/backend 移植，记录每个会话的文档上传历史
"""

from app.core.structured_logging import get_structured_logger
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime

logger = get_structured_logger(__name__)

class DocumentUploadTracker:
    """文档上传追踪器"""

    @staticmethod
    def create_upload_record(
        db: Session,
        session_id: str,
        document_name: str,
        document_type: str,
        file_size: Optional[int] = None
    ):
        """
        创建文档上传记录

        Args:
            db: 数据库会话
            session_id: 会话ID
            document_name: 文档名称
            document_type: 文档类型（pdf/docx/txt）
            file_size: 文件大小（字节）

        Returns:
            创建的记录对象
        """
        try:
            # 动态导入模型以避免循环导入
            from app.models.conversation import DocumentUpload

            upload_record = DocumentUpload(
                session_id=session_id,
                document_name=document_name,
                document_type=document_type,
                file_size=file_size,
                upload_time=datetime.now()
            )

            db.add(upload_record)
            db.commit()
            db.refresh(upload_record)

            logger.info(f"文档上传记录已创建: session_id={session_id}, doc={document_name}")
            return upload_record

        except Exception as e:
            db.rollback()
            logger.error(f"创建文档上传记录失败: {e}")
            raise

    @staticmethod
    def get_session_documents(db: Session, session_id: str) -> List:
        """
        获取指定会话的所有文档上传记录

        Args:
            db: 数据库会话
            session_id: 会话ID

        Returns:
            文档记录列表
        """
        try:
            from app.models.conversation import DocumentUpload

            stmt = select(DocumentUpload).where(
                DocumentUpload.session_id == session_id
            ).order_by(DocumentUpload.upload_time.desc())

            result = db.execute(stmt).scalars().all()
            return list(result)

        except Exception as e:
            logger.error(f"获取会话文档失败: {e}")
            return []

    @staticmethod
    def has_uploaded_documents(db: Session, session_id: str) -> bool:
        """
        检查指定会话是否有上传的文档

        Args:
            db: 数据库会话
            session_id: 会话ID

        Returns:
            True表示有文档
        """
        try:
            from app.models.conversation import DocumentUpload

            stmt = select(DocumentUpload).where(
                DocumentUpload.session_id == session_id
            )

            count = db.execute(stmt).scalar() or 0
            return count > 0

        except Exception as e:
            logger.error(f"检查会话文档失败: {e}")
            return False

    @staticmethod
    def get_latest_document(db: Session, session_id: str) -> Optional:
        """
        获取指定会话最新上传的文档

        Args:
            db: 数据库会话
            session_id: 会话ID

        Returns:
            最新文档记录，如果不存在返回None
        """
        try:
            from app.models.conversation import DocumentUpload

            stmt = select(DocumentUpload).where(
                DocumentUpload.session_id == session_id
            ).order_by(DocumentUpload.upload_time.desc()).limit(1)

            result = db.execute(stmt).scalar_one_or_none()
            return result

        except Exception as e:
            logger.error(f"获取最新文档失败: {e}")
            return None

    @staticmethod
    def get_document_summary(db: Session, session_id: str) -> dict:
        """
        获取会话文档上传摘要信息

        Args:
            db: 数据库会话
            session_id: 会话ID

        Returns:
            摘要信息字典
        """
        try:
            has_documents = DocumentUploadTracker.has_uploaded_documents(db, session_id)
            latest_document = DocumentUploadTracker.get_latest_document(db, session_id)
            all_documents = DocumentUploadTracker.get_session_documents(db, session_id)

            return {
                "session_id": session_id,
                "has_documents": has_documents,
                "latest_document_name": latest_document.document_name if latest_document else None,
                "latest_document_type": latest_document.document_type if latest_document else None,
                "latest_upload_time": latest_document.upload_time if latest_document else None,
                "total_documents": len(all_documents)
            }

        except Exception as e:
            logger.error(f"获取文档摘要失败: {e}")
            return {
                "session_id": session_id,
                "has_documents": False,
                "error": str(e)
            }

# 创建全局服务实例
document_upload_tracker = DocumentUploadTracker()

def get_document_upload_tracker() -> DocumentUploadTracker:
    """获取文档上传追踪服务实例"""
    return document_upload_tracker
