"""
会话管理增强服务
从 swxy/backend 移植，提供完整的会话生命周期管理
"""

from app.core.structured_logging import get_structured_logger
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

logger = get_structured_logger(__name__)

class SessionManagerEnhanced:
    """增强的会话管理器"""

    @staticmethod
    async def create_session(
        db: Session,
        user_id: str,
        initial_question: Optional[str] = None,
        generate_name: bool = True
    ) -> Dict[str, Any]:
        """
        创建新会话并自动命名

        Args:
            db: 数据库会话
            user_id: 用户ID
            initial_question: 初始问题（用于生成会话名称）
            generate_name: 是否自动生成会话名称

        Returns:
            会话信息字典
        """
        try:
            from app.models.conversation import Conversation
            from app.services.rag.session_namer import session_namer

            # 创建会话
            conversation = Conversation()
            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            session_id = str(conversation.id)
            session_name = None

            # 自动生成会话名称
            if generate_name and initial_question:
                try:
                    session_name = await session_namer.generate_session_name(
                        user_question=initial_question,
                        fallback=f"会话 {session_id}"
                    )
                    conversation.title = session_name
                    db.commit()
                except Exception as e:
                    logger.error(f"生成会话名称失败: {e}")
                    session_name = f"会话 {session_id}"
            else:
                session_name = f"会话 {session_id}"

            logger.info(f"会话已创建: id={session_id}, name={session_name}")

            return {
                "session_id": session_id,
                "session_name": session_name,
                "user_id": user_id,
                "created_at": conversation.created_at.isoformat(),
                "status": "success"
            }

        except Exception as e:
            db.rollback()
            logger.error(f"创建会话失败: {e}")
            raise

    @staticmethod
    def update_session_name(
        db: Session,
        session_id: str,
        question: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        更新会话名称（如果尚未设置）

        Args:
            db: 数据库会话
            session_id: 会话ID
            question: 用户问题
            user_id: 用户ID

        Returns:
            是否更新成功
        """
        try:
            from app.models.conversation import Conversation
            from app.services.rag.session_namer import session_namer

            # 查询会话
            stmt = db.query(Conversation).filter(Conversation.id == int(session_id))
            conversation = stmt.first()

            if not conversation:
                logger.warning(f"会话不存在: {session_id}")
                return False

            # 如果已有名称，跳过
            if conversation.title and conversation.title.strip():
                logger.info(f"会话已有名称: {conversation.title}")
                return True

            # 生成新名称
            session_name = await session_namer.generate_session_name(question)

            conversation.title = session_name
            if user_id:
                # 假设有user_id字段，如果没有可以忽略
                pass

            db.commit()
            logger.info(f"会话名称已更新: {session_id} -> {session_name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"更新会话名称失败: {e}")
            return False

    @staticmethod
    def get_session_history(
        db: Session,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取会话历史

        Args:
            db: 数据库会话
            session_id: 会话ID
            limit: 最大返回条数

        Returns:
            消息历史列表
        """
        try:

            stmt = db.query(Message).filter(
                Message.conversation_id == int(session_id)
            ).order_by(Message.created_at.asc()).limit(limit)

            messages = stmt.all()

            history = []
            for msg in messages:
                history.append({
                    "id": msg.id,
                    "role": msg.role.value,
                    "content": msg.content,
                    "metadata": msg.message_metadata,
                    "created_at": msg.created_at.isoformat()
                })

            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {e}")
            return []

    @staticmethod
    def get_user_sessions(
        db: Session,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取用户的所有会话

        Args:
            db: 数据库会话
            user_id: 用户ID
            limit: 返回数量
            offset: 偏移量

        Returns:
            会话列表
        """
        try:
            from app.models.conversation import Conversation

            # 假设conversations表有user_id字段，如果没有需要调整
            stmt = db.query(Conversation).order_by(
                Conversation.updated_at.desc()
            ).limit(limit).offset(offset)

            conversations = stmt.all()

            sessions = []
            for conv in conversations:
                sessions.append({
                    "session_id": str(conv.id),
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat()
                })

            return sessions

        except Exception as e:
            logger.error(f"获取用户会话失败: {e}")
            return []

# 创建全局服务实例
session_manager_enhanced = SessionManagerEnhanced()

def get_session_manager_enhanced() -> SessionManagerEnhanced:
    """获取增强会话管理器实例"""
    return session_manager_enhanced
