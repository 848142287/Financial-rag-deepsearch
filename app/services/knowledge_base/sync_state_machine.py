"""
双库同步状态机
管理文档同步的状态转移流程
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import logging
import json

from ...core.redis_client import redis_client

logger = logging.getLogger(__name__)


class SyncState(Enum):
    """同步状态枚举"""
    INIT = "init"                    # 初始状态
    READY = "ready"                  # 准备同步
    VECTOR_ING = "vector_ing"        # 向量库同步中
    GRAPH_ING = "graph_ing"          # 图谱库同步中
    LINK_ING = "link_ing"            # 关联建立中
    COMPLETED = "completed"          # 同步完成
    FAILED = "failed"                # 同步失败
    ROLLBACK = "rollback"            # 回滚状态


@dataclass
class SyncStatus:
    """同步状态信息"""
    document_id: str
    state: SyncState
    version: int
    content_hash: str
    previous_state: Optional[SyncState] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    retry_count: int = 0


class SyncStateMachine:
    """同步状态机"""

    def __init__(self):
        self.state_transitions = {
            SyncState.INIT: [SyncState.READY, SyncState.FAILED],
            SyncState.READY: [SyncState.VECTOR_ING, SyncState.FAILED],
            SyncState.VECTOR_ING: [SyncState.GRAPH_ING, SyncState.FAILED],
            SyncState.GRAPH_ING: [SyncState.LINK_ING, SyncState.FAILED],
            SyncState.LINK_ING: [SyncState.COMPLETED, SyncState.FAILED],
            SyncState.COMPLETED: [SyncState.READY],  # 重新同步
            SyncState.FAILED: [SyncState.READY, SyncState.ROLLBACK],
            SyncState.ROLLBACK: [SyncState.INIT, SyncState.FAILED]
        }

        self.max_retry_count = 3
        self.state_timeout = 300  # 5分钟超时

    async def initialize_sync(
        self,
        document_id: str,
        version: int,
        content_hash: str
    ) -> SyncStatus:
        """初始化同步状态"""
        status = SyncStatus(
            document_id=document_id,
            state=SyncState.INIT,
            version=version,
            content_hash=content_hash,
            start_time=datetime.now(timezone.utc),
            metadata={}
        )

        await self._save_status(status)
        await self._transition_state(status, SyncState.READY)

        logger.info(f"初始化同步状态: {document_id}")
        return status

    async def transition_to_vector_sync(self, document_id: str) -> bool:
        """转移到向量同步状态"""
        status = await self._get_status(document_id)
        if not status:
            return False

        return await self._transition_state(status, SyncState.VECTOR_ING)

    async def transition_to_graph_sync(self, document_id: str) -> bool:
        """转移到图谱同步状态"""
        status = await self._get_status(document_id)
        if not status:
            return False

        return await self._transition_state(status, SyncState.GRAPH_ING)

    async def transition_to_link_building(self, document_id: str) -> bool:
        """转移到关联建立状态"""
        status = await self._get_status(document_id)
        if not status:
            return False

        return await self._transition_state(status, SyncState.LINK_ING)

    async def transition_to_completed(self, document_id: str) -> bool:
        """转移到完成状态"""
        status = await self._get_status(document_id)
        if not status:
            return False

        success = await self._transition_state(status, SyncState.COMPLETED)
        if success:
            status.end_time = datetime.now(timezone.utc)
            await self._save_status(status)

        return success

    async def transition_to_failed(self, document_id: str, error_message: str) -> bool:
        """转移到失败状态"""
        status = await self._get_status(document_id)
        if not status:
            return False

        status.error_message = error_message
        status.retry_count += 1

        success = await self._transition_state(status, SyncState.FAILED)
        if success:
            await self._save_status(status)

        return success

    async def retry_sync(self, document_id: str) -> bool:
        """重试同步"""
        status = await self._get_status(document_id)
        if not status or status.retry_count >= self.max_retry_count:
            return False

        return await self._transition_state(status, SyncState.READY)

    async def rollback_sync(self, document_id: str) -> bool:
        """回滚同步"""
        status = await self._get_status(document_id)
        if not status:
            return False

        return await self._transition_state(status, SyncState.ROLLBACK)

    async def get_sync_status(self, document_id: str) -> Optional[SyncStatus]:
        """获取同步状态"""
        return await self._get_status(document_id)

    async def get_all_pending_syncs(self) -> List[str]:
        """获取所有待同步的文档ID"""
        try:
            keys = await redis_client.keys("sync_status:*")
            pending_docs = []

            for key in keys:
                status_data = await redis_client.get(key)
                if status_data:
                    status = json.loads(status_data)
                    state = status.get('state')
                    if state in ['ready', 'vector_ing', 'graph_ing', 'link_ing']:
                        pending_docs.append(status.get('document_id'))

            return pending_docs

        except Exception as e:
            logger.error(f"获取待同步文档失败: {e}")
            return []

    async def cleanup_completed_syncs(self, days: int = 7) -> int:
        """清理已完成的同步状态"""
        try:
            keys = await redis_client.keys("sync_status:*")
            cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
            cleaned_count = 0

            for key in keys:
                status_data = await redis_client.get(key)
                if status_data:
                    status = json.loads(status_data)
                    state = status.get('state')
                    end_time = status.get('end_time')

                    if state == 'completed' and end_time:
                        end_timestamp = datetime.fromisoformat(end_time).timestamp()
                        if end_timestamp < cutoff_time:
                            await redis_client.delete(key)
                            cleaned_count += 1

            logger.info(f"清理完成同步状态: {cleaned_count} 个")
            return cleaned_count

        except Exception as e:
            logger.error(f"清理同步状态失败: {e}")
            return 0

    async def _transition_state(self, status: SyncStatus, new_state: SyncState) -> bool:
        """状态转移"""
        # 检查转移是否合法
        if new_state not in self.state_transitions.get(status.state, []):
            logger.error(f"非法状态转移: {status.state} -> {new_state}")
            return False

        # 更新状态
        status.previous_state = status.state
        status.state = new_state

        # 保存状态
        await self._save_status(status)

        # 记录状态转移日志
        await self._log_state_transition(status.document_id, status.previous_state, new_state)

        logger.info(f"状态转移: {status.document_id} {status.previous_state.value} -> {new_state.value}")
        return True

    async def _get_status(self, document_id: str) -> Optional[SyncStatus]:
        """获取状态信息"""
        try:
            key = f"sync_status:{document_id}"
            status_data = await redis_client.get(key)

            if status_data:
                data = json.loads(status_data)
                return SyncStatus(
                    document_id=data['document_id'],
                    state=SyncState(data['state']),
                    version=data['version'],
                    content_hash=data['content_hash'],
                    previous_state=SyncState(data['previous_state']) if data.get('previous_state') else None,
                    error_message=data.get('error_message'),
                    start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
                    end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
                    metadata=data.get('metadata', {}),
                    retry_count=data.get('retry_count', 0)
                )

            return None

        except Exception as e:
            logger.error(f"获取同步状态失败 {document_id}: {e}")
            return None

    async def _save_status(self, status: SyncStatus):
        """保存状态信息"""
        try:
            key = f"sync_status:{status.document_id}"
            status_data = {
                'document_id': status.document_id,
                'state': status.state.value,
                'version': status.version,
                'content_hash': status.content_hash,
                'previous_state': status.previous_state.value if status.previous_state else None,
                'error_message': status.error_message,
                'start_time': status.start_time.isoformat() if status.start_time else None,
                'end_time': status.end_time.isoformat() if status.end_time else None,
                'metadata': status.metadata or {},
                'retry_count': status.retry_count,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            await redis_client.setex(key, 86400, json.dumps(status_data))  # 24小时过期

        except Exception as e:
            logger.error(f"保存同步状态失败 {status.document_id}: {e}")

    async def _log_state_transition(
        self,
        document_id: str,
        from_state: SyncState,
        to_state: SyncState
    ):
        """记录状态转移日志"""
        try:
            log_key = f"sync_log:{document_id}"
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'from_state': from_state.value if from_state else None,
                'to_state': to_state.value,
                'document_id': document_id
            }

            # 使用Redis列表存储日志
            await redis_client.lpush(log_key, json.dumps(log_entry))
            await redis_client.expire(log_key, 86400 * 7)  # 7天过期

        except Exception as e:
            logger.error(f"记录状态转移日志失败: {e}")

    async def get_sync_log(self, document_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取同步日志"""
        try:
            log_key = f"sync_log:{document_id}"
            log_entries = await redis_client.lrange(log_key, 0, limit - 1)

            logs = []
            for entry in log_entries:
                logs.append(json.loads(entry))

            return logs

        except Exception as e:
            logger.error(f"获取同步日志失败 {document_id}: {e}")
            return []


# 全局状态机实例
sync_state_machine = SyncStateMachine()