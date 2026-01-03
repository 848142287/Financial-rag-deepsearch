"""
对话管理相关API端点
"""

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Any, Optional
from datetime import datetime

from app.core.database import get_db
from app.schemas.conversation import Conversation, Message, ConversationCreate, MessageCreate
from app.models.conversation import Conversation as ConversationModel, Message as MessageModel
import json

router = APIRouter()


@router.post("", response_model=Conversation)
async def create_conversation(
    conversation_data: Optional[ConversationCreate] = None,
    db: Session = Depends(get_db)
) -> Any:
    """创建对话"""
    try:
        # 创建新对话
        title = "新建对话"
        if conversation_data and conversation_data.title:
            title = conversation_data.title

        conversation = ConversationModel(
            title=title,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        # 添加初始欢迎消息
        welcome_message = MessageModel(
            conversation_id=conversation.id,
            role="assistant",
            content="您好！我是金融研报智能助手，可以帮助您分析和查询金融研报内容。请问有什么可以帮助您的吗？",
            created_at=datetime.utcnow()
        )

        db.add(welcome_message)
        db.commit()
        db.refresh(welcome_message)

        # 添加消息到响应
        conversation.messages = [welcome_message]

        return conversation

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/", response_model=List[Conversation])
async def list_conversations(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
) -> Any:
    """获取对话列表"""
    try:
        conversations = db.query(ConversationModel)\
            .order_by(ConversationModel.updated_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
        return conversations

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
) -> Any:
    """获取对话详情"""
    try:
        conversation = db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .first()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )


@router.get("/{conversation_id}/messages", response_model=List[Message])
async def get_conversation_messages(
    conversation_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> Any:
    """获取对话消息列表"""
    try:
        # 验证对话是否存在
        conversation = db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .first()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        # 获取消息
        messages = db.query(MessageModel)\
            .filter(MessageModel.conversation_id == conversation_id)\
            .order_by(MessageModel.created_at.asc())\
            .offset(skip)\
            .limit(limit)\
            .all()

        return messages

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get messages: {str(e)}"
        )


@router.post("/{conversation_id}/messages", response_model=Message)
async def send_message(
    conversation_id: int,
    message_data: MessageCreate,
    db: Session = Depends(get_db)
) -> Any:
    """发送消息"""
    try:
        # 验证对话是否存在
        conversation = db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .first()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        # 创建消息
        message = MessageModel(
            conversation_id=conversation_id,
            role=message_data.role,
            content=message_data.content,
            message_metadata=message_data.metadata
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        # 更新对话的更新时间
        conversation.updated_at = datetime.utcnow()
        db.commit()

        return message

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}"
        )


@router.put("/{conversation_id}")
async def update_conversation(
    conversation_id: int,
    title: str,
    db: Session = Depends(get_db)
) -> Any:
    """更新对话标题"""
    try:
        conversation = db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .first()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        conversation.title = title
        conversation.updated_at = datetime.utcnow()
        db.commit()

        return {"message": "Conversation updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation: {str(e)}"
        )


@router.post("/{conversation_id}/clear")
async def clear_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
) -> Any:
    """清空对话消息"""
    try:
        conversation = db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .first()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        # 删除所有消息
        db.query(MessageModel).filter(
            MessageModel.conversation_id == conversation_id
        ).delete()

        # 添加欢迎消息
        welcome_message = MessageModel(
            conversation_id=conversation_id,
            role="assistant",
            content="对话已清空。有什么可以帮助您的吗？",
            created_at=datetime.utcnow()
        )

        db.add(welcome_message)
        db.commit()

        # 更新对话时间
        conversation.updated_at = datetime.utcnow()
        db.commit()

        return {"message": "Conversation cleared successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
) -> Any:
    """删除对话"""
    try:
        conversation = db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .first()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        db.delete(conversation)
        db.commit()

        return {"message": "Conversation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@router.websocket("/ws/{conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """WebSocket聊天端点"""
    await websocket.accept()

    try:
        # 验证对话是否存在
        conversation = db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .first()

        if not conversation:
            await websocket.send_json({
                "type": "error",
                "message": "Conversation not found"
            })
            await websocket.close()
            return

        # 发送连接确认
        await websocket.send_json({
            "type": "connected",
            "conversation_id": conversation_id,
            "message": f"Connected to conversation {conversation_id}"
        })

        # 持续监听消息
        while True:
            try:
                # 接收消息
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # 验证消息格式
                if not message_data.get("content"):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Message content is required"
                    })
                    continue

                # 创建消息记录
                message = MessageModel(
                    conversation_id=conversation_id,
                    role=message_data.get("role", "user"),
                    content=message_data["content"],
                    message_metadata=message_data.get("metadata")
                )
                db.add(message)
                db.commit()
                db.refresh(message)

                # 更新对话时间
                conversation.updated_at = datetime.utcnow()
                db.commit()

                # 发送消息确认
                await websocket.send_json({
                    "type": "message_received",
                    "message": {
                        "id": message.id,
                        "role": message.role,
                        "content": message.content,
                        "created_at": message.created_at.isoformat()
                    }
                })

                # 集成RAG系统生成回复
                try:
                    # DEPRECATED: Use UnifiedRAGService instead - from app.services.rag.unified_rag_entry import UnifiedRAGService agentic_rag_service

                    # 获取对话历史作为上下文
                    context_messages = await agentic_rag_service._get_conversation_context(conversation_id)

                    # 生成RAG响应
                    result = await agentic_rag_service.query(
                        question=message_data["content"],
                        conversation_id=conversation_id,
                        mode="enhanced"  # 使用增强检索模式
                    )

                    # 创建助手回复消息
                    assistant_message = MessageModel(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=result.get("answer", "抱歉，我无法回答这个问题。"),
                        message_metadata={
                            "sources": result.get("sources", []),
                            "retrieval_mode": "enhanced",
                            "response_time": result.get("response_time", 0)
                        }
                    )
                    db.add(assistant_message)
                    db.commit()
                    db.refresh(assistant_message)

                    # 更新对话时间
                    conversation.updated_at = datetime.utcnow()
                    db.commit()

                    # 发送助手回复
                    await websocket.send_json({
                        "type": "assistant_response",
                        "message": {
                            "id": assistant_message.id,
                            "role": assistant_message.role,
                            "content": assistant_message.content,
                            "created_at": assistant_message.created_at.isoformat(),
                            "metadata": assistant_message.message_metadata
                        }
                    })

                except Exception as e:
                    logger.error(f"Failed to generate RAG response: {e}")

                    # 发生错误时创建默认回复
                    error_message = MessageModel(
                        conversation_id=conversation_id,
                        role="assistant",
                        content="抱歉，处理您的请求时遇到了问题。请稍后再试。",
                        message_metadata={"error": True}
                    )
                    db.add(error_message)
                    db.commit()

                    await websocket.send_json({
                        "type": "assistant_response",
                        "message": {
                            "id": error_message.id,
                            "role": error_message.role,
                            "content": error_message.content,
                            "created_at": error_message.created_at.isoformat(),
                            "metadata": error_message.message_metadata
                        }
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })

    except WebSocketDisconnect:
        # 客户端断开连接
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"WebSocket error: {str(e)}"
            })
        except Exception:
            pass
