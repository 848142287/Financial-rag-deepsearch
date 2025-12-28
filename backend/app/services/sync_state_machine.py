"""
同步状态机
管理文档同步过程中的状态转换
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SyncState(Enum):
    """同步状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class SyncStateMachine:
    """同步状态机"""

    def __init__(self):
        self.state_transitions = {
            SyncState.PENDING: [SyncState.PROCESSING, SyncState.FAILED],
            SyncState.PROCESSING: [SyncState.INDEXING, SyncState.FAILED],
            SyncState.INDEXING: [SyncState.COMPLETED, SyncState.FAILED, SyncState.RETRYING],
            SyncState.RETRYING: [SyncState.PROCESSING, SyncState.FAILED],
            SyncState.COMPLETED: [],  # Terminal state
            SyncState.FAILED: [SyncState.RETRYING, SyncState.PENDING]  # Can retry or restart
        }

    def can_transition(self, from_state: SyncState, to_state: SyncState) -> bool:
        """检查是否可以进行状态转换"""
        return to_state in self.state_transitions.get(from_state, [])

    def transition(self, current_state: SyncState, new_state: SyncState, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行状态转换"""
        if not self.can_transition(current_state, new_state):
            raise ValueError(f"Invalid state transition from {current_state.value} to {new_state.value}")

        transition_info = {
            "from_state": current_state.value,
            "to_state": new_state.value,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }

        logger.info(f"State transition: {current_state.value} -> {new_state.value}")
        return transition_info

    def get_next_states(self, current_state: SyncState) -> List[SyncState]:
        """获取可能的下一个状态"""
        return self.state_transitions.get(current_state, [])

    def is_terminal_state(self, state: SyncState) -> bool:
        """检查是否为终止状态"""
        return len(self.state_transitions.get(state, [])) == 0